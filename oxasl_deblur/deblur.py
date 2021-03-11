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

def threshold(arr, thresh_value, useabs=False, binarise=False):
    """
    Threshold an array

    :param arr: Array to threshold
    :param thresh_value: Threshold value - all values below this are zeroed
    :param useabs: If True, threshold based on absolute value
    :param binarise: If True, set all non-zeroed values to 1
    """
    if useabs:
        arr = np.absolute(arr)
       
    arr[arr < thresh_value] = 0
    if binarise: arr[arr >= thresh_value] = 1
    return arr

def flatten_mask(mask, thresh_value):
    """
    Create a 2D array whose values are 1 if there are at least
    ``thresh_value`` unmasked voxels in the z direction, 0 otherwise.
    """
    if thresh_value > mask.shape[2]:
        raise RuntimeError("Cannot flatten mask with a threshold larger than the z dimension")

    mask = np.copy(mask)
    mask[mask > 0] = 1
    return threshold(np.sum(mask, 2), thresh_value, binarise=True)

def zvols_to_matrix(data, mask):
    """
    :param data: 4D Numpy array
    :param mask: Mask flattened in Z dimension (may have optional T dimension)
    :return Masked data as 2D Numpy array - first dimension is XYT, second is Z
    """
    # Mask is 2D need to repeat by number of t points
    if mask.ndim == 2:
        mask = np.expand_dims(mask, -1)

    if mask.shape[2] == 1:
        mask = np.repeat(mask, data.shape[3], 2)
    
    # Flatten with extra z dimension
    mask = np.reshape(mask, [mask.size]) > 0
    
    # need to swap axes so 2nd dim of 2D array is Z not T
    data = np.transpose(data, [0, 1, 3, 2]) # Shape [X, Y, T, Z]
    data2 = np.reshape(data, [mask.size, data.shape[3]]) # Shape [X*Y*T, Z]
    return data2[mask, :]

def zdeblur_make_spec(resids, flatmask):
    # Mask and de-mean residuals
    zdata = zvols_to_matrix(resids, flatmask)
    mean = zdata.mean(axis=1, dtype=np.float64)
    ztemp = zdata - mean[:, np.newaxis]
    
    thepsd = np.absolute(fft(ztemp, axis=1))
    thepsd = np.mean(thepsd, 0)
    return thepsd

def lorentzian(x, gamma):
    return 1/pi * (0.5*gamma)/(np.square(x)+(0.5*gamma)**2)

def lorentzian_kern(gamma, length, demean=True):
    half = (float(length)-1)/2
    x = list(range(int(ceil(half))+1)) + list(range(int(floor(half)), 0, -1))
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
    noise with length data points, where the Gaussian std dev is sigma
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

    (autocorr is ifft of the power spectral density)
    """
    data_raw_autocorr = np.real(ifft(np.square(np.absolute(thefft))))
    data_raw_autocorr = data_raw_autocorr/max(data_raw_autocorr)

    popt, _ = curve_fit(gaussian_autocorr, len(data_raw_autocorr), data_raw_autocorr, 1)
    return popt[0]

def create_deblur_kern(wsp, thefft, kernel_length, sig=1):
    """
    Create the deblurring kernel

    :param kernel_name: Kernel name
    """
    np.set_printoptions(precision=16)
    if wsp.deblur_kernel == "direct":
        slope = thefft[1]-thefft[2]
        thefft[0] = thefft[1]+slope #put the mean in for tapering of the AC
        thefft = thefft/(thefft[1]+slope) #normalise, we want DC=1, but we will have to extrapolate as we dont ahve DC
        
        # multiply AC by tukey window
        i1 = np.real(ifft(np.square(thefft)))
        t1 = 1-tukey(len(thefft), sig)
        thefft = np.sqrt(np.absolute(fft(np.multiply(i1, t1))))
        thefft[0] = 0 # back to zero mean
    elif wsp.deblur_kernel == "lorentz":
        ac = np.real(ifft(np.square(thefft))) # autocorrelation
        ac = ac/max(ac)
        popt, _ = curve_fit(lorentzian_autocorr, len(ac), ac, 2)
        # Lorentzian autocorrelation function is even but for consistency
        # make gamma positive
        gamma = np.abs(popt[0])
        wsp.log.write(" - Lorentzian kernel: Gamma=%.3f\n" % gamma)
        lozac = lorentzian_autocorr(kernel_length, gamma)
        lozac = lozac/max(lozac)
        thefft = np.absolute(fft(lorentzian_kern(gamma, kernel_length, True))) # when getting final spec. den. include mean
    elif wsp.deblur_kernel == "lorwien":
        ac = np.real(ifft(np.square(thefft))) # autocorrelation
        ac = ac/max(ac)
        popt, _ = curve_fit(lorentzian_wiener, len(ac), ac, (2, 0.01))
        gamma, tunef = popt
        lozac = lorentzian_wiener(kernel_length, gamma, tunef)
        thefft = np.absolute(fft(lorentzian_kern(gamma, kernel_length, True))) # when getting final spec. den. include mean
        thepsd = np.square(thefft)
        tune = tunef*np.mean(thepsd)
        wien = np.divide(thepsd, thepsd+tune)
        wien[0] = 1
        thefft = np.divide(thefft, wien)
    elif wsp.deblur_kernel == "gauss":
        sigfit = fit_gaussian_autocorr(thefft)
        thefft = gaussian_fft(sigfit, kernel_length, True) # When getting final spec. den. include mean
    elif wsp.deblur_kernel == "manual":
        if len(sig) != kernel_length:
            raise RuntimeError("Manual deblur kernel requires signal of length %i" % kernel_length)
        thefft = gaussian_fft(sig, kernel_length, True)
    else:
        raise RuntimeError("Unknown kernel: %s" % wsp.deblur_kernel)

    # note that currently all the ffts have zero DC term!
    invkern = np.reciprocal(np.clip(thefft[1:], 1e-50, None))
    kernel = np.real(ifft(np.insert(invkern, 0, 0)))

    # Code below is commented out in MATLAB original - preserving for now

    # Weiner filter
    # thepsd = thefft.^2
    # tune = 0.01*mean(thepsd)
    # invkern = 1./thefft.*(thepsd./(thepsd+tune))

    # The ffts should be already correctly normalized (unity DC)

    # normalise
    #if sum(kernel)>0.01
    #   kernel = kernel/(sum(kernel))
    #else
    #    warning('normalization of kernel skipped')
    #end

    if len(kernel) < kernel_length:
        # if the kernel is shorter than required pad in the middle by zeros
        n = kernel_length-len(kernel)
        i1 = int(len(kernel)/2)
        kernel = np.concatenate((kernel[:i1], np.zeros(n), kernel[i1:]))
    return kernel
   
def zdeblur_with_kern(volume, kernel, deblur_method="fft"):
    """
    Deblur an image

    :param volume: 4D Numpy array containing data to be deblurred
    :param kernel: Numpy array containing deblurring kernel
    :param deblur_method: Name of method to use with deblurring
    """
    if deblur_method == "fft":
        # FIXME original MATLAB transposes and takes complex conjugate
        # We don't need to transpose, so just take conjugate. However not
        # completely clear if complex conjugate is required or if this is
        # an unintentional side-effect of the transpose
        fftkern = np.conj(fft(kernel))

        # Demean volume along Z axis - kernel is zero mean
        zmean = np.expand_dims(np.mean(volume, 2), 2)
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
        #volout = filter_matrix(volume, kernel)
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
    if wsp.residuals is not None:
        wsp.log.write(' - Residuals already supplied\n')
    else:
        wsp.log.write(' - Running BASIL to generate residuals\n')
        wsp.sub("basil")
        wsp.basil_options = {
            "save-residuals" : True,
            "inferart" : False,
            "spatial" : False,
        }
        basil.basil(wsp, output_wsp=wsp.basil)
        wsp.residuals = wsp.basil.finalstep.residuals

def get_mask(wsp):
    if wsp.mask is None:
        mask.generate_mask(wsp.deblur)
        wsp.mask = wsp.rois.mask
    else:
        wsp.mask = wsp.mask

def run(wsp, output_wsp=None):
    """
    Run deblurring on an OXASL workspace

    Required workspace attributes
    -----------------------------

     - ``deblur_method`` : Deblurring method name
     - ``deblur_kernel`` : Deblurring kernel name

    Optional workspace attributes
    -----------------------------

     - ``mask`` : Data mask. If not provided, will be auto generated
     - ``residuals`` : Residuals from model fit on ASL data. If not specified and ``wsp.asldata``
                       is provided, will run BASIL fitting on this data to generate residuals
     - ``kernel`` : Numpy array containing pre-generated kernel
    """
    wsp.log.write('\nDeblurring data\n')
    wsp.sub("deblur")
    if output_wsp is None:
        output_wsp = wsp.deblur

    get_kernel(wsp.deblur)

    wsp.log.write(' - Deblur method: %s\n' % wsp.deblur_method)
    output_wsp.asldata = deblur_img(wsp.deblur, wsp.asldata)
    if wsp.calib is not None:
        output_wsp.calib = deblur_img(wsp.deblur, wsp.calib)
    if wsp.addimg is not None:
        output_wsp.addimg = deblur_img(wsp.deblur, wsp.addimg)
    # FIXME CATC, CBLIP...

    wsp.log.write('DONE\n')

def data_pad(img, padding_slices=2):
    """
    :param img: Image in ASL data space

    :return: 4D Numpy array of data in an image, padded by 2 slices top and bottom
    """
    # Ensure data is 4D by padding additional dimension if necessary
    data = img.data
    if data.ndim == 3:
        data = data[..., np.newaxis]

    # Pad the data - 2 slices top and bottom
    return np.pad(data, [(0, 0), (0, 0), (padding_slices, padding_slices), (0, 0)], 'edge')

def data_unpad(padded_data, padding_slices=2):
    """
    :param padded_data: Numpy array of data padded by 2 slices top and bottom
    :return: 4D Numpy array with padding discarded
    """
    return padded_data[:, :, padding_slices:-padding_slices, :]

def get_kernel(wsp):
    if wsp.kernel is not None:
        wsp.log.write(' - Kernel already supplied\n')
    else:
        # Number of slices that are non zero in mask
        get_mask(wsp)
        maskser = np.sum(wsp.mask.data, (0, 1))
        nslices = np.sum(maskser > 0)
        flatmask = flatten_mask(wsp.mask.data, nslices-2)

        # Commented out in MATLAB code
        # residser = zdeblur_make_series(resids,flatmask)
        get_residuals(wsp)
        thespecd = zdeblur_make_spec(wsp.residuals.data, flatmask)

        # NB data has more slices than residuals
        wsp.log.write(' - Using kernel: %s\n' % wsp.deblur_kernel)
        data_padded = data_pad(wsp.asldata)
        sig = wsp.ifnone("sig", 1)
        wsp.kernel = create_deblur_kern(wsp, thespecd, data_padded.shape[2], sig)

def deblur_img(wsp, img):
    """
    Deblur an image

    :param wsp: Workspace
    :param img: Image to be deblurred
    :return Deblurred Image
    """
    wsp.log.write(' - Deblurring image %s\n' % img.name)
    data_padded = data_pad(img)
    data_deblurred = zdeblur_with_kern(data_padded, wsp.kernel, wsp.deblur_method)
    data_out = data_unpad(data_deblurred)
    if img.data.ndim == 3:
        data_out = np.squeeze(data_out, axis=3)

    if isinstance(img, AslImage):
        ret = img.derived(data_out)
    else:
        ret = Image(data_out, header=img.header)

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
        group.add_option("--kernel-file", dest="kernel", help="File containing pre-specified deblurring kernel data", type="matrix")
        group.add_option("--method", dest="deblur_method", 
                         help="Deblurring method: Choicess are 'fft' for division in FFT domain or 'lucy' for Lucy-Richardson (ML solution) for Gaussian noise", 
                         choices=["fft", "lucy"], default="fft")
        group.add_option("--residuals", type="image",
                         help="Image containing the residials from a model fit. If not specified, BASIL options must be given to perform model fit")
        group.add_option("--addimg", type="image", help="Additional image to deblur using same residuals. Output will be saved as <filename>_deblur")
        group.add_option("--save-kernel", help="Save deblurring kernel", action="store_true", default=False)
        group.add_option("--save-residuals", help="Save residuals used to generate kernel", action="store_true", default=False)
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
        if wsp.save_residuals:
            wsp.output.residuals = wsp.deblur.residuals

        if not wsp.debug:
            wsp.deblur = None
            wsp.input = None

        print('\nOXASL_DEBLUR - DONE - output is in %s' % options.output)

    except RuntimeError as e:
        print("ERROR: " + str(e) + "\n")
        sys.exit(1)
    
if __name__ == "__main__":
    main()
