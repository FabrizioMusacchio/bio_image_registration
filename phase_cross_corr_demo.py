# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft
from skimage.transform import SimilarityTransform, warp
from scipy.ndimage import fourier_shift
# %% PATHS
RESULTSpath = "plots/"
# %% FUNCTIONS
def plot_2Dimage(image, title, path):
    plt.close()
    plt.figure(1, figsize=(4,4))
    plt.imshow(image)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(RESULTSpath+title+".pdf")
# %% CREATE SOME TOY DATA
image = data.cells3d()[30,1,:,:]
shift_known = (-30,15)
tform = SimilarityTransform(translation=(shift_known))
image_moved = warp(image, tform, preserve_range=True)
# image_moved = fourier_shift(np.fft.fftn(image), shift_known)
# image_moved = np.fft.ifftn(image_moved)
plot_2Dimage(image, title="phase_cross_corr_image", path=RESULTSpath)
plot_2Dimage(image_moved, title="phase_cross_corr_image_moved", path=RESULTSpath)
#plot_2Dimage(image_moved.real, title="phase_cross_corr_image_moved", path=RESULTSpath)
# %% CROSS CORRELATION

shape = image.shape
# compute the cross-power spectrum of the two images:
image_product = np.fft.fft2(image) * np.fft.fft2(image_moved).conj()
# compute the (not-normalized) cross-correlation between the two images:
cc_image = np.fft.ifft2(image_product)
# for visualization reasons, shift the zero-frequency component to the center of the spectrum:
cc_image_fftshift = np.fft.fftshift(cc_image)
plot_2Dimage(cc_image_fftshift.real, title="phase_cross_corr_cross_corr", path=RESULTSpath)

# find the peak in cc_image: 
maxima = np.unravel_index(np.argmax(np.abs(cc_image)), shape)
midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])
float_dtype = image_product.real.dtype
shifts = np.stack(maxima).astype(float_dtype, copy=False)
shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]
print(f"detected shifts: {shifts[1], shifts[0]} (known shift: {shift_known})")

