"""
OXASL DEBLUR

Perform z-deblurring of ASL data

kernel options are:
  direct - estimate kernel directly from data
  gauss  - use gauss kernel, but estimate size from data
  manual - gauss kernel with size given by sigma
  lorentz - lorentzain kernel, estimate size from data
  lorwein - lorentzian kernel with weiner type filter

deblur methods are:
  fft - do division in FFT domain
  lucy - Lucy-Richardson (ML solution) for Gaussian noise

(c) Michael A. Chappell, University of Oxford, 2009-2018
"""
from __future__ import print_function

import sys
from math import exp, pi, ceil, floor, sqrt

import numpy as np

from scipy.fftpack import fft, ifft
from scipy.signal import tukey
from scipy.optimize import curve_fit

from fsl.data.image import Image

from oxasl import Workspace, AslImage, image, basil, mask
from oxasl.options import AslOptionParser, OptionCategory, OptionGroup, GenericOptions

from ._version import __version__, __timestamp__

def threshold(arr, thresh, useabs=False, binarise=False):
    """
    Threshold an array

    :param arr: Array to threshold
    :param thresh: Threshold value - all values below this are zeroed
    :param useabs: If True, threshold based on absolute value
    :param binarise: If True, set all non-zeroed values to 1
    """
    if useabs:
        arr = np.absolute(arr)
       
    arr[arr < thresh] = 0
    if binarise: arr[arr >= thresh] = 1
    return arr

def flattenmask(mask, thr):
    """
    Create a 2D array whose values are 1 if there are at least
    ``thr`` unmasked voxels in the z direction, 0 otherwise.
    """
    if thr > mask.shape[2]:
        raise RuntimeError("Cannot flatten mask with a threshold larger than the z dimension")

    # Set all unmasked voxels to 1
    mask = np.copy(mask)
    mask[mask > 0] = 1

    return threshold(np.sum(mask, 2), thr, binarise=True)

def Zvols2matrix(data, mask):
    """
    Takes 4D volume and 2D (xy) or 3D (xyt) mask and return 2D matrix
    (space-time x z-dimension)
    
    Just vols2matrix but choosing the Z-dimension
    """
    # Mask is 2D need to repeat by number of t points
    if mask.ndim == 2:
        mask = np.expand_dims(mask, -1)

    if mask.shape[2] == 1:
        mask = np.repeat(mask, data.shape[3], 2)
    
    # Flatten with extra z dimension
    mask = np.reshape(mask, [mask.size]) > 0
    
    # need to swap axes so 2nd dim of 2D array is Z not T
    data = np.transpose(data, [0, 1, 3, 2])
    data2 = np.reshape(data, [mask.size, data.shape[3]])
    return data2[mask, :]

def zdeblur_make_spec(resids, flatmask):
    zdata = Zvols2matrix(resids, flatmask)
    ztemp = np.zeros(zdata.shape)
    mean = zdata.mean(axis=1, dtype=np.float64)
    ztemp = zdata - mean[:, np.newaxis]
    
    thepsd = np.absolute(fft(ztemp, axis=1))
    thepsd = np.mean(thepsd, 0)
    return thepsd

def lorentzian(x, gamma):
    return 1/pi * (0.5*gamma)/(np.square(x)+(0.5*gamma)**2)

def lorentzian_kern(gamma, length, demean=True):
    half = (float(length)-1)/2
    x = range(0, int(ceil(half))+1) + range(int(floor(half)), 0, -1)
    out = lorentzian(x, gamma)
    if demean: out = out - np.mean(out) #zero mean/DC
    return out

def lorentzian_autocorr(length, gamma):
    return np.real(ifft(np.square(np.absolute(fft(lorentzian_kern(gamma, length, 1))))))

def lorentzian_wiener(length, gamma, tunef):
    thefft = np.absolute(fft(lorentzian_kern(gamma, length, True)))
    thepsd = np.square(thefft)
    tune = tunef*np.mean(thepsd)
    wien = np.divide(thepsd, thepsd+tune)
    wien[0] = 1 # we are about to dealing with a demeaned kernel
    out = np.real(ifft(np.divide(thepsd, np.square(wien))))
    return out/max(out)

def gaussian_autocorr(length, sig):
    """
    Returns the autocorrelation function for Gaussian smoothed white
    noise with length data points, where the Gaussian std dev is sigma 
    
    For now we go via the gaussian fourier transform
    (autocorr is ifft of the power spectral density)
    ideally , we should just analytically calc the autocorr
    """
    gfft = gaussian_fft(sig, length)
    x = np.real(ifft(np.square(gfft))) 

    if max(x) > 0:
        x = x/max(x)
    return x

def gaussian_fft(sig, length, demean=True):
    """
    Returns the fourier transform function for Gaussian smoothed white
    noise with len data points, where the Gaussian std dev is sigma 
    """
    tres = 1.0
    fres = 1.0/(tres*length)
    maxk = 1/tres
    krange = np.linspace(fres, maxk, length)
    
    x = [sig*exp(-(0.5*sig**2*(2*pi*k)**2))+sqrt(2*pi)*sig*exp(-(0.5*sig**2*(2*pi*((maxk+fres)-k))**2))
         for k in krange]
    if demean: x[0] = 0
    return x

def fit_gaussian_autocorr(thefft):
    """
    Fit a Gaussian autocorrelation model to the data and return the
    std dev sigma
    """

    # (autocorr is ifft of the power spectral density)
    data_raw_autocorr = np.real(ifft(np.square(np.absolute(thefft))))
    data_raw_autocorr = data_raw_autocorr/max(data_raw_autocorr)

    popt, _ = curve_fit(gaussian_autocorr, len(data_raw_autocorr), data_raw_autocorr, 1)
    return popt[0]

def create_deblur_kern(thefft, kernel, kernlen, sig=1):
    np.set_printoptions(precision=16)
    if kernel == "direct":
        slope = thefft[1]-thefft[2]
        thefft[0] = thefft[1]+slope #put the mean in for tapering of the AC
        thefft = thefft/(thefft[1]+slope) #normalise, we want DC=1, but we will have to extrapolate as we dont ahve DC
        
        # multiply AC by tukey window
        i1 = np.real(ifft(np.square(thefft)))
        t1 = 1-tukey(len(thefft), sig)
        thefft = np.sqrt(np.absolute(fft(np.multiply(i1, t1))))
        thefft[0] = 0 # back to zero mean
    elif kernel == "lorentz":
        ac = np.real(ifft(np.square(thefft))) # autocorrelation
        ac = ac/max(ac)
        popt, _ = curve_fit(lorentzian_autocorr, len(ac), ac, 2)
        gamma = popt[0]
        lozac = lorentzian_autocorr(kernlen, gamma)
        lozac = lozac/max(lozac)
        thefft = np.absolute(fft(lorentzian_kern(gamma, kernlen, True))) # when getting final spec. den. include mean        
    elif kernel == "lorwien":
        ac = np.real(ifft(np.square(thefft))) # autocorrelation
        ac = ac/max(ac)
        popt, _ = curve_fit(lorentzian_wiener, len(ac), ac, (2, 0.01))
        gamma, tunef = popt
        lozac = lorentzian_wiener(kernlen, gamma, tunef)
        thefft = np.absolute(fft(lorentzian_kern(gamma, kernlen, True))) # when getting final spec. den. include mean
        thepsd = np.square(thefft)
        tune = tunef*np.mean(thepsd)
        wien = np.divide(thepsd, thepsd+tune)
        wien[0] = 1
        thefft = np.divide(thefft, wien)
    elif kernel == "gauss":
        sigfit = fit_gaussian_autocorr(thefft)
        thefft = gaussian_fft(sigfit, kernlen, True) # When getting final spec. den. include mean
    elif kernel == "manual":
        if len(sig) != kernlen:
            raise RuntimeError("Manual deblur kernel requires signal of length %i" % kernlen)
        thefft = gaussian_fft(sig, kernlen, True)
    else:
        raise RuntimeError("Unknown kernel: %s" % kernel)

    # note that currently all the ffts have zero DC term!
    invkern = np.reciprocal(np.clip(thefft[1:], 1e-50, None))
    kern = np.real(ifft(np.insert(invkern, 0, 0)))
    
    # Code below is commented out in MATLAB original - preserving for now

    # Weiner filter
    # thepsd = thefft.^2
    # tune = 0.01*mean(thepsd)
    # invkern = 1./thefft.*(thepsd./(thepsd+tune))

    # The ffts should be already correctly normalized (unity DC)

    # normalise
    #if sum(kern)>0.01
    #   kern = kern/(sum(kern))
    #else
    #    warning('normalization of kernel skipped')
    #end

    if len(kern) < kernlen:
        # if the kernel is shorter than required pad in the middle by zeros
        n = kernlen-len(kern)
        i1 = int(len(kern)/2)
        kern = np.concatenate((kern[:i1], np.zeros(n), kern[i1:]))
    return kern
   
def zdeblur_with_kern(volume, kern, deblur_method="fft"):
    if deblur_method == "fft":

        # FIXME MATLAB code below transposes and takes complex conjugate 
        # We don't need to transpose, so just take conjugate. However not 
        # completely clear if complex conjugate is required or if this is
        # an unintentional side-effect of the transpose
        fftkern = np.conj(fft(kern))
        #if size(fftkern,2)==1
        #    fftkern = fftkern'
        #end

        # demean volume (in z) - 'cos kern is zero mean
        m = np.expand_dims(np.mean(volume, 2), 2)
        zmean = np.repeat(m, volume.shape[2], 2)
        volume = volume  - zmean
        
        fftkern = np.expand_dims(fftkern, 0)
        fftkern = np.expand_dims(fftkern, 0)
        fftkern = np.expand_dims(fftkern, -1)
        fftkern2 = np.zeros(volume.shape, dtype=complex)
        fftkern2[:, :, :, :] = fftkern
        fftvol = fft(volume, axis=2)
        volout = np.real(ifft(np.multiply(fftkern2, fftvol), axis=2))
        volout += zmean
        return volout

    elif deblur_method == "lucy":
        #volout = filter_matrix(volume, kern)
        raise RuntimeError("Lucy-Richardson deconvolution not supported in this version of ASL_DEBLUR")
    else:
        raise RuntimeError("Unknown deblur method: %s" % deblur_method)

# FIXME this code is not complete because we get numerical problems and it is not
# clear if the method is correctly implemented.
# def filter_matrix(data, kernel):
#     # This is the wrapper for the Lucy-Richardson deconvolution
#     #
#     # Filter matrix creates the different matrices before applying the
#     # deblurring algorithm
#     # Input --> original deltaM maps kernel
#     # Output --> deblurred deltaM maps
#     #
#     # (c) Michael A. Chappell & Illaria Boscolo Galazzo, University of Oxford, 2012-2014

#     # MAC 4/4/14 removed the creation of the lorentz kernel and allow to accept
#     # any kernel
#     #
#     nr, nc, ns, nt = data.shape
#     # Matrix K 
#     kernel_max = kernel/np.sum(kernel)
#     matrix_kernel = np.zeros((len(kernel), ns))
#     matrix_kernel[:, 0] = kernel_max
#     for i in range(1, ns):
#         matrix_kernel[:, i] = np.concatenate([np.zeros(i), kernel_max[:ns-i]])
    
#     # Invert with SVD
#     #U, S, V = svd(matrix_kernel)
#     #W = np.diag(np.reciprocal(np.diag(S)))
#     #W[S < (0.2*S[0])] = 0
#     #inverse_matrix = V*W*U.'
#     inverse_matrix = np.linalg.inv(matrix_kernel)
    
#     # Deblurring Algorithm
#     index = 1
#     for i in range(1, nr+1):
#         for j in range(1, nc+1):
#             for k in range(1, nt+1):
#                 index = index+1
#                 #waitbar(index/(nt*nc*nc),h)
#                 data_vettore = data[i, j, :, k]
#                 initial_estimate = np.dot(inverse_matrix, data_vettore)
#     #deblur = deconvlucy_asl(data_vettore,kernel,8,initial_estimate)
#     #deblur_image[i,j,:,k] = deblur
#     deblur_image = None
#     return deblur_image 

def get_residuals(wsp):
    """
    Run BASIL on ASL data to get residuals prior to deblurring
    """
    wsp.log.write(' - Running BASIL to generate residuals\n')
    wsp.sub("basil")
    wsp.basil_options = {
        "save-residuals" : True,
    #    "inferart" : False,
    #    "spatial" : False,
    }
    basil.basil(wsp, output_wsp=wsp.basil)
    wsp.residuals = wsp.basil.finalstep.residuals

def run(wsp, output_wsp=None):
    """
    Run deblurring on an OXASL workspace
    """
    wsp.sub("deblur")
    if output_wsp is None:
        output_wsp = wsp.deblur
    wsp.log.write('\nDeblurring data\n')

    if wsp.mask is None:
        mask.generate_mask(wsp.deblur)
        wsp.deblur.mask = wsp.deblur.rois.mask
    else:
        wsp.deblur.mask = wsp.mask

    if wsp.residuals is None:
        get_residuals(wsp.deblur)

    output_wsp.asldata = deblur_img(wsp.deblur, wsp.asldata)
    if wsp.calib is not None:
        output_wsp.calib = deblur_img(wsp.deblur, wsp.calib)
    if wsp.addimg is not None:
        output_wsp.addimg = deblur_img(wsp.deblur, wsp.addimg)
    # FIXME CATC, CBLIP...

    wsp.log.write('DONE\n')

def deblur_img(wsp, img):
    """
    Do deblurring on ASL data

    :param img: Image to deblur. If AslImage is provided, an AslImage is returned
    :return: Image or AslImage
    
    Required workspace attributes
    -----------------------------

     - ``deblur_method`` : Deblurring method name
     - ``deblur_kernel`` : Deblurring kernel name

    Optional workspace attributes
    -----------------------------

     - ``mask`` : Data mask. If not provided, will be auto generated
     - ``residuals`` : Residuals from model fit on ASL data. If not specified and ``wsp.asldata``
                       is provided, will run BASIL fitting on this data to generate residuals
    """
    # Ensure data is 4D by padding additional dimension if necessary
    deblur_data = img.data
    if deblur_data.ndim == 3:
        deblur_data = deblur_data[..., np.newaxis]

    # Pad the data - 2 slices top and bottom
    data_pad = np.pad(deblur_data, [(0, 0), (0, 0), (2, 2), (0, 0)], 'edge')
 
    if wsp.kernel is None:
        wsp.log.write(' - Using kernel: %s\n' % wsp.deblur_kernel)
        wsp.log.write(' - Deblur method: %s\n' % wsp.deblur_method)
        # Number of slices that are non zero in mask
        maskser = np.sum(wsp.mask.data, (0, 1))
        nslices = np.sum(maskser > 0)
        flatmask = flattenmask(wsp.mask.data, nslices-2)

        # Commented out in MATLAB code
        # residser = zdeblur_make_series(resids,flatmask)
        thespecd = zdeblur_make_spec(wsp.residuals.data, flatmask)

        # NB data has more slices than residuals
        sig = wsp.ifnone("sig", 1)
        wsp.kernel = create_deblur_kern(thespecd, wsp.deblur_kernel, data_pad.shape[2], sig)

    # Do deblurring
    wsp.log.write(' - Deblurring image %s\n' % img.name)
    dataout = zdeblur_with_kern(data_pad, wsp.kernel, wsp.deblur_method)
    
    # Discard padding and return
    dataout = dataout[:, :, 2:-2, :]
    if img.data.ndim == 3:
        dataout = np.squeeze(dataout, axis=3)

    if isinstance(img, AslImage):
        ret = img.derived(dataout)
    else:
        ret = Image(dataout, header=img.header)

    return ret

class Options(OptionCategory):
    """
    DEBLUR option category
    """
    def __init__(self, **kwargs):
        OptionCategory.__init__(self, "deblur", **kwargs)

    def groups(self, parser):
        group = OptionGroup(parser, "DEBLUR options")
        group.add_option("--kernel", dest="deblur_kernel", 
                         help="Deblurring kernel: Choices are 'direct' (estimate kernel directly from data), "
                              "'gauss' - Gaussian kernel but estimate size from data, "
                              "'manual' - Gaussian kernel with size given by sigma"
                              "'lorentz' - Lorentzian kernel, estimate size from data"
                              "'lorwein' - Lorentzian kernel with weiner type filter", 
                         choices=["direct", "gauss", "manual", "lorentz", "lorwein"], default="direct")
        group.add_option("--method", dest="deblur_method", 
                         help="Deblurring method: Choicess are 'fft' for division in FFT domain or 'lucy' for Lucy-Richardson (ML solution) for Gaussian noise", 
                         choices=["fft", "lucy"], default="fft")
        group.add_option("--residuals", type="image",
                         help="Image containing the residials from a model fit. If not specified, BASIL options must be given to perform model fit")
        group.add_option("--addimg", type="image", help="Additional image to deblur using same residuals. Output will be saved as <filename>_deblur")
        group.add_option("--save-kernel", help="Save deblurring kernel", action="store_true", default=False)
        return [group]

def main():
    """
    Entry point for OXASL_DEBLUR command line application
    """
    try:
        parser = AslOptionParser(usage="oxasl_deblur -i <ASL input file> [options...]", version=__version__)
        parser.add_category(image.AslImageOptions())
        parser.add_category(Options())
        parser.add_category(basil.BasilOptions())
        #parser.add_category(calib.CalibOptions())
        parser.add_category(GenericOptions())

        options, _ = parser.parse_args()
        if not options.output:
            options.output = "oxasl_deblur"

        if not options.asldata:
            sys.stderr.write("Input ASL data not specified\n")
            parser.print_help()
            sys.exit(1)
                
        print("OXASL_DEBLUR %s (%s)\n" % (__version__, __timestamp__))
        wsp = Workspace(savedir=options.output, auto_asldata=True, **vars(options))
        wsp.asldata.summary()
        wsp.sub("output")
        run(wsp, output_wsp=wsp.output)
        if wsp.save_kernel:
            wsp.output.kernel = wsp.deblur.kernel

        if not wsp.debug:
            wsp.deblur = None
            wsp.input = None

        print('\nOXASL_DEBLUR - DONE - output is %s' % options.output)

    except RuntimeError as e:
        print("ERROR: " + str(e) + "\n")
        sys.exit(1)
    
if __name__ == "__main__":
    main()
