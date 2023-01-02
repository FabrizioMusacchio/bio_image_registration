"""
At the moment, a simple routine that performs affine 3D image registration.

author: Fabrizio Musacchio (fabriziomusacchio.com)
date:   Nov 10, 2022

conda create -n imreg_conda python=3.9
conda activate imreg_conda
conda install -y ipykernel matplotlib numpy pandas scikit-image pooch scipy opencv nbformat plotly
"""
# %% IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import data
from skimage.io import imread
from skimage.registration import phase_cross_correlation, optical_flow_tvl1, optical_flow_ilk
from skimage.transform import SimilarityTransform, warp, AffineTransform
import tifffile
import scipy as sp
import cv2 as cv
from scipy import signal
import scipy as sp
import os
import plotly
import plotly.express as px
# %% PATHS
DATApath    = "/Users/husker/Workspace/Falko 3P Spines/Data for registration/files2register/"
RESULTSpath = "/Users/husker/Workspace/Falko 3P Spines/Data for registration/registered/"
# %% PARAMETERS

# %% FUNCTIONS
def list_specific_files(directory, extensions):
    """
        Function to search for file within a given folder and with specific file-extensions.

        The entire function can also be plugged in a list comprehension btw:

          sorted(file for file in os.listdir(Current_TL_Folder) if file.endswith(r".tif") or file.endswith(r".czi"))

        FMusacchio, 17. Feb 2021

        :param      directory       path to folder, in which function should search
        :param      extensions      list of file extensions to look for
        :return:    filelist        list of files found with the requested extensions

    DEBUG:
        extensions = ["czi", "tif"]
        directory = Current_TL_Folder

      or:
        list_specific_files(Current_TL_Folder, ["czi", "tif"])

    """
    # Main search algorithm:
    filelist = []
    for file in os.listdir(directory):
        for extension in extensions:
            if file.endswith('.' + extension):
                filelist.append(file)

    return sorted(filelist)
def filterdir_by_string(path_to_folder, search_string, search_type="file", exact=False):
    """ scans for folders (search_type="folder") or files (search_type="file")
        in path_to_folder, whos name(s) contain search_string.
        Returns MatchingFolders_Indices and folderlist_matching.

    INPUT:
        path_to_folder  the folder path wherein to search
        search_string   the file or folder name to search for
        search_type     "file" or "folder"
        exact           False: lists all files/folders containing - also partly- the
                                search_string
                        True: case sensitivity for the search_string

    RETURN:
        MatchingFolders_Indices index/indices of matching file(s) or folder(s)
        folderlist_matching     list of matching file(s) or folder(s)
        folderlist              list of all found files or folders

    DEBUG:
        path_to_folder = Current_Mouse_Folder
        search_string  = imaris_file
        exact = True
        search_type    = "file" or "folder"
    """
    #path_to_folder = os.path.join(path_to_folder)
    folderlist = sorted(os.listdir(path_to_folder))
    entry = 0
    MatchingFolders_Indices = []
    folderlist_matching = []
    for folder in folderlist:
        if exact:
            if search_string == folder:
                if search_type == "file":
                    MatchingFolders_Indices.append(entry)
                    folderlist_matching.append(folder)
                    # print(entry, folder)
                elif search_type == "folder":
                    if os.path.isdir((path_to_folder + folder)):
                        MatchingFolders_Indices.append(entry)
                        folderlist_matching.append(folder)
            entry += 1
        else:
            if search_string in folder:
                if search_type == "file":
                    MatchingFolders_Indices.append(entry)
                    folderlist_matching.append(folder)
                    # print(entry, folder)
                elif search_type == "folder":
                    if os.path.isdir((path_to_folder + folder)):
                        MatchingFolders_Indices.append(entry)
                        folderlist_matching.append(folder)
            entry += 1
    return (MatchingFolders_Indices, folderlist_matching, folderlist)
# %% READ THE IMAGE FILE AND REGISTER
tif_files = list_specific_files(DATApath, ["tif"])
#_, tif_files, _ = filterdir_by_string(DATApath, search_string="tif", search_type="file", exact=False)

for tif_file in tif_files:
    print(f"registering {tif_file}")
    # tif_file=tif_files[0]
    curr_tif_file = os.path.join(DATApath, tif_file)
    curr_tif = imread(curr_tif_file)
    """ fig = px.imshow(curr_tif, animation_frame=0, binary_string=True, binary_format='jpg')
    fig.layout.title = 'unregistered'
    plotly.io.show(fig) """
       
    #template = curr_tif.max(axis=0)
    template_orig = curr_tif.max(axis=0)
    template = sp.ndimage.median_filter(curr_tif, 3).max(axis=0)
    pearson_corr_R = np.zeros((curr_tif.shape[0], 2))
    image_reg         = np.zeros((curr_tif.shape))
    image_reg_medfilt = np.zeros((curr_tif.shape))
    image_medfilt = np.zeros((curr_tif.shape))
    for layer in range(curr_tif.shape[0]):
        moved_orig = curr_tif[layer]
        moved = sp.ndimage.median_filter(curr_tif[layer], 3)
        v, u = optical_flow_ilk(template, moved, radius=15, prefilter=False, num_warp=30, gaussian=False)
        nr, nc = template.shape
        row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
        image_reg[layer]= warp(moved_orig, np.array([row_coords + v, col_coords + u]), 
                               mode='constant', preserve_range=True).astype(template.dtype)
        image_reg_medfilt[layer] = sp.ndimage.median_filter(image_reg[layer], 3)
        image_medfilt[layer] = sp.ndimage.median_filter(curr_tif[layer], 3)
        pearson_corr_R[layer,0] = sp.stats.pearsonr(template_orig.flatten(), moved_orig.flatten())[0]
        pearson_corr_R[layer,1] = sp.stats.pearsonr(template_orig.flatten(), image_reg[layer].flatten())[0]
    """ fig = px.imshow(curr_tif, animation_frame=0, binary_string=True, binary_format='jpg')
    fig.layout.title = 'unregistered'
    plotly.io.show(fig)
    fig = px.imshow(image_reg, animation_frame=0, binary_string=True, binary_format='jpg')
    fig.layout.title = 'registered'
    plotly.io.show(fig)
    print(pearson_corr_R)
    plt.imshow(curr_tif.max(axis=0))
    plt.imshow(image_reg.max(axis=0))
    plt.imshow(sp.ndimage.median_filter(curr_tif, 3).max(axis=0))
    plt.imshow(sp.ndimage.median_filter(image_reg, 3).max(axis=0))
    plt.imshow(image_reg_medfilt.max(axis=0))
    plt.imshow(image_reg_medfilt.max(axis=0)-sp.ndimage.median_filter(image_reg, 3).max(axis=0))
    
    plt.imshow(sp.ndimage.median_filter(image_reg, 3).max(axis=0))
    plt.imshow(sp.ndimage.gaussian_filter(sp.ndimage.median_filter(image_reg, 3), 0.5).max(axis=0)) """
    print(f"done. saving files.")
    
    TIFF_path = os.path.join(RESULTSpath, os.path.splitext(tif_file)[0]+"_reg.tif")
    tifffile.imwrite(TIFF_path, image_reg.astype("float32"), 
                         resolution=(curr_tif.shape[1], curr_tif.shape[2]),
                         metadata={'spacing': 1, 'unit': 'um', 'axes': 'ZYX'},
                         imagej=True, bigtiff=False)
    TIFF_path = os.path.join(RESULTSpath, os.path.splitext(tif_file)[0]+"_reg_medfilt2D.tif")
    tifffile.imwrite(TIFF_path, image_reg_medfilt.astype("float32"), 
                         resolution=(curr_tif.shape[1], curr_tif.shape[2]),
                         metadata={'spacing': 1, 'unit': 'um', 'axes': 'ZYX'},
                         imagej=True, bigtiff=False)
    TIFF_path = os.path.join(RESULTSpath, os.path.splitext(tif_file)[0]+"_reg_medfilt3D.tif")
    tifffile.imwrite(TIFF_path, sp.ndimage.median_filter(image_reg, 3).astype("float32"), 
                         resolution=(curr_tif.shape[1], curr_tif.shape[2]),
                         metadata={'spacing': 1, 'unit': 'um', 'axes': 'ZYX'},
                         imagej=True, bigtiff=False)
    
    TIFF_path = os.path.join(RESULTSpath, os.path.splitext(tif_file)[0]+"_reg_proj.tif")
    tifffile.imwrite(TIFF_path, image_reg.max(axis=0).astype("float32"), 
                         resolution=(curr_tif.shape[1], curr_tif.shape[2]),
                         metadata={'spacing': 1, 'unit': 'um', 'axes': 'YX'},
                         imagej=True, bigtiff=False)
    TIFF_path = os.path.join(RESULTSpath, os.path.splitext(tif_file)[0]+"_reg_proj_medfilt2D.tif")
    tifffile.imwrite(TIFF_path, image_reg_medfilt.max(axis=0).astype("float32"), 
                         resolution=(curr_tif.shape[1], curr_tif.shape[2]),
                         metadata={'spacing': 1, 'unit': 'um', 'axes': 'YX'},
                         imagej=True, bigtiff=False)
    TIFF_path = os.path.join(RESULTSpath, os.path.splitext(tif_file)[0]+"_reg_proj_medfilt3D.tif")
    tifffile.imwrite(TIFF_path, sp.ndimage.median_filter(image_reg, 3).max(axis=0).astype("float32"), 
                         resolution=(curr_tif.shape[1], curr_tif.shape[2]),
                         metadata={'spacing': 1, 'unit': 'um', 'axes': 'YX'},
                         imagej=True, bigtiff=False)
    
    TIFF_path = os.path.join(RESULTSpath, os.path.splitext(tif_file)[0]+"_reg_proj_medfilt3D_gaussfilts0.75.tif")
    tifffile.imwrite(TIFF_path, sp.ndimage.gaussian_filter(sp.ndimage.median_filter(image_reg, 3), 0.75).max(axis=0).astype("float32"), 
                         resolution=(curr_tif.shape[1], curr_tif.shape[2]),
                         metadata={'spacing': 1, 'unit': 'um', 'axes': 'YX'},
                         imagej=True, bigtiff=False)
    TIFF_path = os.path.join(RESULTSpath, os.path.splitext(tif_file)[0]+"_reg_proj_medfilt3D_gaussfilts1.0.tif")
    tifffile.imwrite(TIFF_path, sp.ndimage.gaussian_filter(sp.ndimage.median_filter(image_reg, 3), 1.0).max(axis=0).astype("float32"), 
                         resolution=(curr_tif.shape[1], curr_tif.shape[2]),
                         metadata={'spacing': 1, 'unit': 'um', 'axes': 'YX'},
                         imagej=True, bigtiff=False)
    
    TIFF_path = os.path.join(RESULTSpath, os.path.splitext(tif_file)[0]+"_proj.tif")
    tifffile.imwrite(TIFF_path, curr_tif.max(axis=0).astype("float32"), 
                         resolution=(curr_tif.shape[1], curr_tif.shape[2]),
                         metadata={'spacing': 1, 'unit': 'um', 'axes': 'YX'},
                         imagej=True, bigtiff=False)
    TIFF_path = os.path.join(RESULTSpath, os.path.splitext(tif_file)[0]+"_proj_medfilt2D.tif")
    tifffile.imwrite(TIFF_path, image_medfilt.max(axis=0).astype("float32"), 
                         resolution=(curr_tif.shape[1], curr_tif.shape[2]),
                         metadata={'spacing': 1, 'unit': 'um', 'axes': 'YX'},
                         imagej=True, bigtiff=False)
    TIFF_path = os.path.join(RESULTSpath, os.path.splitext(tif_file)[0]+"_proj_medfilt3D.tif")
    tifffile.imwrite(TIFF_path, sp.ndimage.median_filter(curr_tif, 3).max(axis=0).astype("float32"), 
                         resolution=(curr_tif.shape[1], curr_tif.shape[2]),
                         metadata={'spacing': 1, 'unit': 'um', 'axes': 'YX'},
                         imagej=True, bigtiff=False)
    
    
    
    """ fig = px.imshow(image_reg, animation_frame=0, binary_string=True, binary_format='jpg')
    fig.layout.title = 'registered'
    plotly.io.show(fig)
    plt.imshow(curr_tif.max(axis=0))
    plt.imshow(image_reg.max(axis=0))
    plt.imshow(image_reg.max(axis=0)-curr_tif.max(axis=0)) """

# %% CREATE SOME TOY DATA
layers=100
image = data.cells3d()[30,1,:,:]
plt.imshow(image)

image_dichotomized = np.zeros((image.shape), dtype=image.dtype)
image_dichotomized[:128,30:] = image[:128,0:-30]
image_dichotomized[128:,:-30]= image[128:,30:]
plt.imshow(image_dichotomized)

image_dichotomized_sheared = np.zeros((image.shape), dtype=image.dtype)
tform = AffineTransform(shear=np.pi/35)
image_dichotomized_sheared[:128,30:] = warp(image[:128,0:-30], tform, preserve_range=True).astype(image.dtype)
image_dichotomized_sheared[128:,:-30]= warp(image[128:,30:], tform.inverse, preserve_range=True).astype(image.dtype)
plt.imshow(image_dichotomized_sheared)

image_shifted = np.zeros((image.shape))
tform = SimilarityTransform(translation=(-30,15))
image_shifted = warp(image, tform, preserve_range=True).astype(image.dtype)
plt.imshow(image_shifted)
# %% optical_flow_tvl1:
#image_shifted = image_dichotomized_sheared

v, u = optical_flow_tvl1(image, image_shifted, attachment=40, tightness=0.2, num_warp=15, 
                         num_iter=10, tol=0.0001, prefilter=False)
nr, nc = image.shape
row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
image_reg= warp(image_shifted, np.array([row_coords + v, col_coords + u]), 
                mode='constant', preserve_range=True).astype(image.dtype)

nvec = 20  # Number of vectors to be displayed along each image dimension
step = max(nr//nvec, nc//nvec)
y, x = np.mgrid[:nr:step, :nc:step]
u_ = -u[::step, ::step]
v_ = -v[::step, ::step]

pearson_corr_R     = sp.stats.pearsonr(image.flatten(), image_shifted.flatten())[0]
pearson_corr_R_reg = sp.stats.pearsonr(image.flatten(), image_reg.flatten())[0]
print(f"Pearson correlation coefficient image vs. moved: {pearson_corr_R}")
print(f"Pearson correlation coefficient image vs. registration: {pearson_corr_R_reg}")

fig = plt.figure(figsize=(28, 23))
ax1 = plt.subplot(1, 5, 1)
ax2 = plt.subplot(1, 5, 2, sharex=ax1, sharey=ax1)
ax3 = plt.subplot(1, 5, 3, sharex=ax1, sharey=ax1)
ax4 = plt.subplot(1, 5, 4, sharex=ax1, sharey=ax1)
ax5 = plt.subplot(1, 5, 5, sharex=ax1, sharey=ax1)
ax1.imshow(image)
ax1.set_title("original")
ax2.imshow(image_shifted)
ax2.set_title("moved")
ax3.imshow(image_reg)
ax3.set_title("registered")
ax4.imshow(image-image_reg)
ax4.set_title("original-registered")
ax5.imshow(image_shifted)
ax5.quiver(x, y, u_, v_, color='pink', units='dots',
           angles='xy', scale_units='xy', lw=3)
ax5.set_title("optical flow magnitude and vector field")
# %% optical_flow_ilk
v, u = optical_flow_ilk(image, image_shifted, radius=15, prefilter=False, num_warp=30, gaussian=False)
nr, nc = image.shape
row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
image_reg= warp(image_shifted, np.array([row_coords + v, col_coords + u]), 
                mode='constant', preserve_range=True).astype(image.dtype)

pearson_corr_R     = sp.stats.pearsonr(image.flatten(), image_shifted.flatten())[0]
pearson_corr_R_reg = sp.stats.pearsonr(image.flatten(), image_reg.flatten())[0]
print(f"Pearson correlation coefficient image vs. moved: {pearson_corr_R}")
print(f"Pearson correlation coefficient image vs. registration: {pearson_corr_R_reg}")

nvec = 20  # Number of vectors to be displayed along each image dimension
step = max(nr//nvec, nc//nvec)
y, x = np.mgrid[:nr:step, :nc:step]
u_ = -u[::step, ::step]
v_ = -v[::step, ::step]

fig = plt.figure(figsize=(28, 19))
ax1 = plt.subplot(2, 3, 1)
ax2 = plt.subplot(2, 3, 2, sharex=ax1, sharey=ax1)
ax3 = plt.subplot(2, 3, 3, sharex=ax1, sharey=ax1)
ax4 = plt.subplot(2, 3, 5, sharex=ax1, sharey=ax1)
ax5 = plt.subplot(2, 3, 6, sharex=ax1, sharey=ax1)
ax1.imshow(image)
ax1.set_title("original")
ax2.imshow(image_shifted)
ax2.set_title("moved")
ax3.imshow(image_reg)
ax3.set_title("registered")
ax4.imshow(image-image_reg)
ax4.set_title("original-registered")
ax5.imshow(image_shifted)
ax5.quiver(x, y, u_, v_, color='pink', units='dots',
           angles='xy', scale_units='xy', lw=3)
ax5.set_title("optical flow magnitude and vector field")
plt.tight_layout()
# %% calcOpticalFlowFarneback

""" flow = cv.calcOpticalFlowFarneback(prev=image, next=image_shifted, pyr_scale=0.5, levels=15, winsize=50,
                                    iterations=10, poly_n=7, poly_sigma=0, flags=0, flow=None) """
flow = cv.calcOpticalFlowFarneback(prev=image, next=image_shifted, pyr_scale=0.5, levels=15, winsize=7,
                                    iterations=10, poly_n=5, poly_sigma=10, flags=0, flow=None)
h, w = flow.shape[:2]
#flow = -flow
flow_trans = flow.copy()
flow_trans[:,:,0] += np.arange(w)
flow_trans[:,:,1] += np.arange(h)[:,np.newaxis]
image_reg = cv.remap(image_shifted, flow_trans, None, cv.INTER_NEAREST)

pearson_corr_R     = sp.stats.pearsonr(image.flatten(), image_shifted.flatten())[0]
pearson_corr_R_reg = sp.stats.pearsonr(image.flatten(), image_reg.flatten())[0]
print(f"Pearson correlation coefficient image vs. moved: {pearson_corr_R}")
print(f"Pearson correlation coefficient image vs. registration: {pearson_corr_R_reg}")


nvec = 20  # Number of vectors to be displayed along each image dimension
step = max(h//nvec, w//nvec)
u = -flow[::step, ::step, 0]
v = -flow[::step, ::step, 1]
x = np.arange(0, flow.shape[1], step)
y = np.arange(flow.shape[0], -1, -step)
#plt.quiver(x, y, u,v)

fig = plt.figure(figsize=(28, 19))
ax1 = plt.subplot(2, 3, 1)
ax2 = plt.subplot(2, 3, 2, sharex=ax1, sharey=ax1)
ax3 = plt.subplot(2, 3, 3, sharex=ax1, sharey=ax1)
ax4 = plt.subplot(2, 3, 5, sharex=ax1, sharey=ax1)
ax5 = plt.subplot(2, 3, 6, sharex=ax1, sharey=ax1)
ax1.imshow(image)
ax1.set_title("original")
ax2.imshow(image_shifted)
ax2.set_title("moved")
ax3.imshow(image_reg)
ax3.set_title("registered")
ax4.imshow(image-image_reg)
ax4.set_title("original-registered")
ax5.imshow(image_shifted)
ax5.quiver(x, y, u, v, color='pink', units='dots',
           angles='xy', scale_units='xy', lw=3)
ax5.set_title("optical flow magnitude and vector field")
# %% END
