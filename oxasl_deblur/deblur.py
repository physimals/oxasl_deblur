"""
OXASL DEBLUR

Perform Z-deblurring of ASL data

kernel options are:
  direct - estimate kernel directly from data
  gauss  - use gauss kernel, but estimate size from data
  manual - gauss kernel with size given by sigma
  lorentz - lorentzain kernel, estimate size from data
  lorwein - lorentzian kernel with weiner type filter

A kernel can also be supplied as a set of values (a 
centered PSF, not necessarily normalized)

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
from scipy.signal import tukey, convolve
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

def get_mean_fft_z(resids, flatmask):
    # Get masked data as matrix: [XYT, Z] and de-mean along Z axis
    zdata = zvols_to_matrix(resids, flatmask)
    zdata_demeaned = zdata - np.mean(zdata, axis=1)[..., np.newaxis]
    
    # Take FFT along Z axis and mean over XYT
    fft_z = np.absolute(fft(zdata_demeaned, axis=1))
    mean_fft_z = np.mean(fft_z, 0)
    return mean_fft_z

def lorentzian(x, gamma):
    return 1/pi * (0.5*gamma)/(np.square(x)+(0.5*gamma)**2)

def lorentzian_kern(gamma, length, demean=True):
    half = (float(length)-1)/2
    x = list(range(int(ceil(half))+1)) + list(range(int(floor(half)), 0, -1))
    out = lorentzian(x, gamma)
    if demean: out = out - np.mean(out) #zero mean/DC
    return out

def lorentzian_autocorr(length, gamma):
    autocorr = np.real(ifft(np.square(np.absolute(fft(lorentzian_kern(gamma, length, 1))))))
    return autocorr / np.max(autocorr)

def lorentzian_wiener(length, gamma, tunef):
    thefft = np.absolute(fft(lorentzian_kern(gamma, length, True)))
    thepsd = np.square(thefft)
    tune = tunef*np.mean(thepsd)
    wien = np.divide(thepsd, thepsd+tune)
    wien[0] = 1 # we are about to dealing with a demeaned kernel
    out = np.real(ifft(np.divide(thepsd, np.square(wien))))
    return out/max(out)

def gaussian_autocorr(length, sigma):
    """
    Returns the autocorrelation function for Gaussian smoothed white
    noise with length data points, where the Gaussian std dev is sigma
    
    For now we go via the gaussian fourier transform
    (autocorr is ifft of the power spectral density)
    ideally , we should just analytically calc the autocorr
    """
    gfft = gaussian_fft(sigma, length)
    x = np.real(ifft(np.square(gfft))) 

    if max(x) > 0:
        x = x/max(x)
    return x

def gaussian_fft(sigma, length, demean=True):
    """
    Returns the fourier transform function for Gaussian smoothed white
    noise with length data points, where the Gaussian std dev is sigma
    """
    tres = 1.0
    fres = 1.0/(tres*length)
    maxk = 1/tres
    krange = np.linspace(fres, maxk, length)
    
    x = [sigma*exp(-(0.5*sigma**2*(2*pi*k)**2))+sqrt(2*pi)*sigma*exp(-(0.5*sigma**2*(2*pi*((maxk+fres)-k))**2))
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

def create_deblur_kern(wsp, mean_fft_z, kernel_length):
    """
    Create the deblurring kernel
    """
    sig = wsp.ifnone("sig", 1)
    np.set_printoptions(precision=16)
    if wsp.deblur_kernel == "direct":
        slope = mean_fft_z[1]-mean_fft_z[2]
        mean_fft_z[0] = mean_fft_z[1]+slope #put the mean in for tapering of the AC
        mean_fft_z = mean_fft_z/(mean_fft_z[1]+slope) #normalise, we want DC=1, but we will have to extrapolate as we dont have DC
        
        # multiply AC by tukey window
        i1 = np.real(ifft(np.square(mean_fft_z)))
        t1 = 1-tukey(len(mean_fft_z), sig)
        kernel = np.multiply(i1, t1)
        kernel_fft = np.sqrt(np.absolute(fft(kernel)))
        wsp.kernel = np.absolute(ifft(kernel_fft))
    elif wsp.deblur_kernel == "lorentz":
        if wsp.deblur_gamma is None:
            ac = np.real(ifft(np.square(mean_fft_z))) # autocorrelation
            ac = ac/max(ac)
            popt, _ = curve_fit(lorentzian_autocorr, len(ac), ac, 2)
            # Autocorrelation function is even but for consistency make gamma positive
            wsp.deblur_gamma = np.abs(popt[0])
        wsp.log.write(" - Lorentzian kernel: Gamma=%.4f\n" % wsp.deblur_gamma)
        wsp.kernel = lorentzian_kern(wsp.deblur_gamma, kernel_length, False)
    elif wsp.deblur_kernel == "lorwien":
        if wsp.deblur_gamma is None:
            ac = np.real(ifft(np.square(mean_fft_z))) # autocorrelation
            ac = ac/max(ac)
            popt, _ = curve_fit(lorentzian_wiener, len(ac), ac, (2, 0.01))
            wsp.deblur_gamma = np.abs(popt[0])
            wsp.deblur_tunef = popt[1]
        wsp.log.write(" - Lorentizian-Weiner kernel: Gamma=%.4f, tunef=%.4f\n" % (wsp.deblur_gamma, wsp.deblur_tunef))
        kernel = lorentzian_kern(wsp.deblur_gamma, kernel_length, False)
        thefft = np.absolute(fft(kernel)) # when getting final spec. den. include mean
        thepsd = np.square(thefft)
        tune = wsp.deblur_tunef*np.mean(thepsd)
        wien = np.divide(thepsd, thepsd+tune)
        wien[0] = 1
        kernel_fft = np.divide(thefft, wien)
        wsp.kernel = ifft(kernel_fft)
    elif wsp.deblur_kernel == "gauss":
        sigfit = fit_gaussian_autocorr(mean_fft_z)
        kernel_fft = gaussian_fft(sigfit, kernel_length, True) # When getting final spec. den. include mean
        wsp.kernel = ifft(kernel_fft)
    elif wsp.deblur_kernel == "manual":
        if len(sig) != kernel_length:
            raise RuntimeError("Manual deblur kernel requires signal of length %i" % kernel_length)
        kernel_fft = gaussian_fft(sig, kernel_length, True)
        wsp.kernel = ifft(kernel_fft)
    else:
        raise RuntimeError("Unknown kernel: %s" % wsp.deblur_kernel)

def deblur_fft(wsp, volume):
    """
    Deconvolution implementation using division in FFT space
    """
    # Demean volume along Z axis - kernel is zero mean
    # This means the DC term in the kernel does not matter - we don't
    # need to demean the kernel
    zmean = np.expand_dims(np.mean(volume, 2), 2)
    volume = volume  - zmean

    fftkern = fft(wsp.kernel)[np.newaxis, np.newaxis, ..., np.newaxis]
    fftvol = fft(volume, axis=2)
    volout = np.real(ifft(np.divide(fftvol, fftkern), axis=2))
    volout += zmean
    return volout

def deblur_lr(wsp, volume):
    """
    Deconvolution implementation using Lucy-Richardson approach modified
    to use Gaussian noise

    Dey, N. et al. 3D Microscopy Deconvolution using Richardson-Lucy Algorithm with 
    Total Variation Regularization.” (2004).
    """
    # Step size for iteration
    alpha = wsp.ifnone("lr_alpha", 1.0)

    # Regularization parameter (0=no regularization)
    lam = wsp.ifnone("lr_lam", 0)

    # Shape PSF for broadcasting and create adjoint operator by reversing
    psf = wsp.psf[np.newaxis, np.newaxis, ..., np.newaxis]
    psf_hat = wsp.psf[::-1][np.newaxis, np.newaxis, ..., np.newaxis]

    #data_deblurred, _remainder = signal.deconvolve(data_padded, psf)

    # Main Lucy-Richardson algorithm
    # Iterate towards maximum likelihood estimate for the deblurred image
    deblurred = np.copy(volume)
    for i in range(wsp.ifnone("lr_iterations", 100)):
        h_est = convolve(deblurred, psf, 'same')
        hhat_h_est = convolve(h_est, psf_hat, 'same')
        hhat_img = convolve(volume, psf_hat, 'same') 
        diff = hhat_img - hhat_h_est
        deblurred = deblurred + alpha*diff

        # Regularisation
        if lam > 0:
            grad = np.array(np.gradient(deblurred, axis=(0, 1, 2)))
            mod_grad = np.sqrt(np.square(grad[0])+np.square(grad[1])+np.square(grad[2]))
            mod_grad[mod_grad == 0] = 1
            grad = grad / mod_grad
            reg = np.ufunc.reduce(np.add, [np.gradient(grad[i], axis=i) for i in range(3)])
            deblurred += lam*reg
        
        # Enforce positivity
        deblurred = np.clip(deblurred, 0, None)

    return deblurred

def zdeblur_with_kern(wsp, volume):
    """
    Deblur an image

    :param volume: 4D Numpy array containing data to be deblurred
    """
    deblur_method = wsp.ifnone("deblur_method", "fft")
    if deblur_method == "fft":
        return deblur_fft(wsp, volume)
    elif deblur_method == "lucy":
        return deblur_lr(wsp, volume)
    else:
        raise RuntimeError("Unknown deblur method: %s" % deblur_method)

# def deconvlucy_asl(data_vector, kernel, iterations, initial_estimate):
#     # deconvlucy_asl is partially based on the deconvlucy function in Matlab.
#     # NOTE = the matlab function implements this type of deconvolution using
#     # the specific formula for Poisson noise.
#     # deconvlucy_asl implements the deconvolution with the modified equation
#     # that considers Gaussian noise.
#     #
#     # This function deconvolves image I using Lucy-Richardson algorithm, 
#     # returning deblurred image J. The assumption is
#     # that the image I was created by convolving a true image with a
#     # point-spread function PSF and possibly by adding noise.
#     # 
#     # Based on Matlab function deconvlucy.m
#     # 27-03-2013 IBG
    
#     # Parse inputs to verify valid function calling syntaxes and arguments
#     J = data_vector
#     PSF = kernel
#     WEIGHT = np.ones(len(initial_estimate), dtype=np.float32)
#     # WEIGHT[0] = 0.7
#     WEIGHT[-1] = 1.3
#     # WEIGHT(end-1)=1.05;
#     # initial_estimate = J{1};
#     DAMPAR = 0
#     SUBSMPL = 1

#     # 1. Prepare PSF --> Our estimated Kernel
#     sizeOTF = len(initial_estimate)
#     sizeOTF(numNSdim) = SUBSMPL*len(PSF)
#     H = PSF
#     ns = length(H)
#     H = H./sum(H)

#     # Matrix K
#     matrix_kernel = np.zeros((ns, ns), dtype=np.float32)
#     matrix_kernel[:,0] = H
#     for i in range(1, ns):
#         matrix_kernel[i:,i] = H[:(ns-i)]
#     H = matrix_kernel

#     # 2. Prepare parameters for iterations
#     #
#     # Create indexes for image according to the sampling rate
#     idx = repmat({':'},[1 length(sizeI)]);
#     for k = numNSdim, # index replicates for non-singleton PSF sizes only
#         idx{k} = reshape(repmat(1:sizeI(k),[SUBSMPL 1]),[SUBSMPL*sizeI(k) 1]);
#     end

#     # J{2} = (abs(sqrt(initial_estimate)));
#     wI = max(WEIGHT.*(READOUT + J{1}),0); # at this point  - positivity constraint
#     J{2} = J{2}(idx{:});
#     scale = 1;
#     clear WEIGHT;
#     DAMPAR22 = (DAMPAR.^2)/2;

#     if SUBSMPL~=1,% prepare vector of dimensions to facilitate the reshaping
#         % when the matrix is binned within the iterations.
#         vec(2:2:2*length(sizeI)) = sizeI;
#         vec(2*numNSdim-1) = -1;
#         vec(vec==0) = [];
#         num = fliplr(find(vec==-1));
#         vec(num) = SUBSMPL;
#     else
#         vec = [];    
#         num = [];
            
#     # 3. L_R Iterations
#     lambda = 2*any(J{4}(:)~=0);
#     for k =  (1:NUMIT)

#         # 3.a Make an image predictions for the next iteration
#         if k > 2,
#             lambda = (J{4}(:,1).'*J{4}(:,2))/(J{4}(:,2).'*J{4}(:,2) + eps);
#             lambda = max(min(lambda,1),0);% stability enforcement
#         end
#     #   Y = max(J{2} + lambda*(J{2} - J{3}),0);% plus positivity constraint
#         Y = max(J{2},0);
#         # 3.b  Make core for the LR estimation
        
#         CC = corelucy_asl(Y,H,DAMPAR22,wI,READOUT,SUBSMPL,idx,vec,num);
        
#         # 3.c Determine next iteration image & apply positivity constraint
#         J{3} = J{2};
        
#         matrix_kernel_flip = (matrix_kernel(end:-1:1,end:-1:1));
#         prodotto = (matrix_kernel_flip)*CC;
#         J{2} = max(Y.*(prodotto)./scale,0); %
#         clear CC;
#         J{4} = [J{2}(:)-Y J{4}(:,1)];

#     # 4. Convert the right array (for cell it is first array, for notcell it is
#     # second array) to the original image class & output whole thing
#     num = 1 + strcmp(classI{1},'notcell');
#     if ~strcmp(classI{2},'double'),
#         J{num} = changeclass(classI{2},J{num});

#     if num==2,% the input & output is NOT a cell
#         J = J{2}

# def corelucy_asl(Y,H,DAMPAR22,wI,READOUT,SUBSMPL,idx,vec,num):

#     Hflip = H[::-1,::-1] # PSFHAT
#     ReBlurred1 = np.dot(H, Y) # PSF * IMG
#     ReBlurred = np.dot(Hflip, ReBlurred1) # PSFHAT * EST

#     # 2. An Estimate for the next step
#     ReBlurred = ReBlurred + READOUT
#     ReBlurred[ReBlurred == 0] = np.eps
#     #AnEstim = wI / ReBlurred
#     #  AnEstim = (wI - ReBlurred)

#     # 3. Damping if needed
#     if DAMPAR22 == 0: # No Damping
#         ImRatio = AnEstim
#     else:
#         # Damping of the image relative to DAMPAR22 = (N*sigma)^2
#         gm = 10
#         g = (wI*np.log(AnEstim)+ ReBlurred - wI)/DAMPAR22
#         g = np.min(g,1)
#         G = (g^(gm-1))*(gm-(gm-1)*g)
#         ImRatio = 1 + G*(AnEstim - 1)

#     return ImRatio

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

#     # Matrix K - [NZ+PAD, NZ]
#     # This seems to be set up to do convolution
#     kernel_normed = kernel/np.sum(kernel)
#     matrix_kernel = np.zeros((len(kernel), ns))
#     matrix_kernel[:, 0] = kernel_normed
#     for i in range(1, ns):
#         matrix_kernel[:, i] = np.concatenate([np.zeros(i), kernel_normed[:ns-i]])

#     # Invert with SVD
#     #U, S, V = svd(matrix_kernel)
#     #W = np.diag(np.reciprocal(np.diag(S)))
#     #W[S < (0.2*S[0])] = 0
#     #inverse_matrix = V*W*U.'
#     inverse_matrix = np.linalg.inv(matrix_kernel)

#     # Deblurring Algorithm
#     # Loop over all Z stacks in the XY plane and over all volumes
#     deblur_image = np.zeros(data.shape, dtype=np.float32)
#     index = 1
#     for i in range(1, nr+1):
#         for j in range(1, nc+1):
#             for k in range(1, nt+1):
#                 index = index+1
#                 data_vector = data[i, j, :, k]
#                 # Initial estimate is just a convolution with inverse convolution matrix?
#                 initial_estimate = np.dot(inverse_matrix, data_vector)
#                 deblur_image[i,j,:,k] = deconvlucy_asl(data_vector, kernel, 8, initial_estimate)
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
    if wsp.cblip is not None:
        output_wsp.cblip = deblur_img(wsp.deblur, wsp.cblip)
    if wsp.cref is not None:
        output_wsp.cref = deblur_img(wsp.deblur, wsp.cref)
    if wsp.cact is not None:
        output_wsp.cact = deblur_img(wsp.deblur, wsp.cact)
    if wsp.addimg is not None:
        output_wsp.addimg = deblur_img(wsp.deblur, wsp.addimg)

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
    data_padded = data_pad(wsp.asldata)
    kernel_length = data_padded.shape[2]

    if wsp.kernel is not None:
        wsp.log.write(' - Kernel already supplied\n')
        # Kernel is supplied as centered
        kernel = np.squeeze(wsp.kernel)
        half = int(len(kernel)/2)
        wsp.kernel = np.concatenate((kernel[half:], kernel[:half]))
    else:
        # Number of slices that are non zero in mask
        get_mask(wsp)
        maskser = np.sum(wsp.mask.data, (0, 1))
        nslices = np.sum(maskser > 0)
        flatmask = flatten_mask(wsp.mask.data, nslices-2)

        get_residuals(wsp)
        mean_fft_z = get_mean_fft_z(wsp.residuals.data, flatmask)

        # NB data has more slices than residuals
        wsp.log.write(' - Using kernel: %s\n' % wsp.deblur_kernel)
        create_deblur_kern(wsp, mean_fft_z, kernel_length)

    if len(wsp.kernel) < kernel_length:
        # if the kernel is shorter than required pad in the middle by zeros
        n = kernel_length-len(wsp.kernel)
        half_len = int(len(wsp.kernel)/2)
        wsp.kernel = np.concatenate((wsp.kernel[:half_len], np.zeros(n), wsp.kernel[half_len:]))
   
    half_len = int(len(wsp.kernel)/2)
    wsp.kernel_centered = np.concatenate((wsp.kernel[half_len:], wsp.kernel[:half_len]))
    #print("kernel (centered): ", wsp.kernel_centered)
    wsp.psf = np.copy(wsp.kernel_centered)
    if len(wsp.psf) % 2 == 0:
        wsp.psf = wsp.psf[1:]
    norm = sum(wsp.kernel_centered)
    wsp.psf /= norm
    #print("psf: ", wsp.psf)

def deblur_img(wsp, img):
    """
    Deblur an image

    :param wsp: Workspace
    :param img: Image to be deblurred
    :return Deblurred Image
    """
    wsp.log.write(' - Deblurring image %s\n' % img.name)
    data_padded = data_pad(img)
    data_deblurred = zdeblur_with_kern(wsp, data_padded)
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
                              "'lorwien' - Lorentzian kernel with weiner type filter", 
                         choices=["direct", "gauss", "manual", "lorentz", "lorwien"], default="direct")
        group.add_option("--kernel-file", dest="kernel", help="File containing pre-specified deblurring kernel data", type="matrix")
        group.add_option("--deblur-gamma", help="Value of gamma to use in lorentzian kernels", type="float")
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
        parser.add_category(image.Options())
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
        wsp.sub("reg")
        wsp.reg.nativeref = wsp.asldata.perf_weighted()
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
