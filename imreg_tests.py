"""
A simple routine testing different affine image registration routines.

author: Fabrizio Musacchio (fabriziomusacchio.com)
date:   Nov 10, 2022

conda create -n imreg_conda python=3.9
conda activate imreg_conda
conda install -y ipykernel matplotlib numpy pandas scikit-image pooch scipy opencv
"""
# %% IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import data
from skimage.registration import phase_cross_correlation, optical_flow_tvl1, optical_flow_ilk
from skimage.transform import SimilarityTransform, warp, warp_polar, AffineTransform, rotate
import scipy as sp
import cv2 as cv
from scipy import signal
from scipy.ndimage import median_filter
from skimage.exposure import match_histograms
# %% PATHS
RESULTSpath = "plots/"
# %% PARAMETERS

# %% FUNCTIONS
def plot_2Dimage(image, title, path):
    plt.close()
    plt.figure(1, figsize=(4,4))
    plt.imshow(image)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(RESULTSpath+title+".pdf")

def plot_phase_cross_corr_results_old(image, image_shifted,image_reg, plotname="phase_cross_corr",
                                  pearson_corr_R=1.0, pearson_corr_R_reg=1.0, corr_border=30,
                                  shifts=[0,0,0]):
    corr_border_0 = corr_border
    corr_border_1 = image.shape[0]-corr_border
    
    plt.close()
    fig = plt.figure(figsize=(12, 8))
    plt.clf()
    ax1 = plt.subplot(2, 3, 1)
    ax2 = plt.subplot(2, 3, 2, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(2, 3, 3, sharex=ax1, sharey=ax1)
    ax6 = plt.subplot(2, 3, 4, sharex=ax1, sharey=ax1)
    ax4 = plt.subplot(2, 3, 5, sharex=ax1, sharey=ax1)
    
    
    ax1.imshow(image)
    ax1.set_title("original", fontsize=14, fontweight="bold")
    ax1.plot([corr_border_0,corr_border_1], [corr_border_0, corr_border_0], '-', c="r", lw=0.75)
    ax1.plot([corr_border_0,corr_border_1], [corr_border_1, corr_border_1], '-', c="r", lw=0.75)
    ax1.plot([corr_border_0,corr_border_0], [corr_border_0, corr_border_1], '-', c="r", lw=0.75)
    ax1.plot([corr_border_1,corr_border_1], [corr_border_0, corr_border_1], '-', c="r", lw=0.75)
    
    #ax6.text(0, 200, plotname,  fontsize=9, fontweight="normal")
    ax6.text(0, 220, f"Correlation coefficients:",  fontsize=13, fontweight="bold", va="top")
    ax6.text(0, 200, f" - image vs. moved:      {np.round(pearson_corr_R, 2)}", fontsize=13, 
             fontweight="bold",  va="top")
    ax6.text(0, 180, f" - image vs. registered: {np.round(pearson_corr_R_reg, 2)}", fontsize=13, 
             fontweight="bold", va="top")
    ax6.text(0, 160, f"(account for area within the", fontsize=13, fontweight="bold", va="top")
    ax6.text(0, 140, f"red rectangle)", fontsize=13, fontweight="bold", va="top")
    ax6.text(0, 100, f"detected ", fontsize=13, fontweight="bold", va="top")
    ax6.text(0, 80, f" - translation: {np.round(shifts[0],1)}, {np.round(shifts[1],1)}", fontsize=13, fontweight="bold", va="top")
    ax6.text(0, 60, f" - rotation:      {np.round(shifts[2],1)}°", fontsize=13, fontweight="bold", va="top")
    
    ax6.spines['right'].set_visible(False)
    ax6.spines['top'].set_visible(False)
    ax6.spines['bottom'].set_visible(False)
    ax6.spines['left'].set_visible(False)
    ax6.set_xticks([])
    ax6.set_yticks([])
    
    ax2.imshow(image_shifted)
    ax2.set_title("moved", fontsize=14, fontweight="bold")
    ax2.plot([corr_border_0,corr_border_1], [corr_border_0, corr_border_0], '-', c="r", lw=0.75)
    ax2.plot([corr_border_0,corr_border_1], [corr_border_1, corr_border_1], '-', c="r", lw=0.75)
    ax2.plot([corr_border_0,corr_border_0], [corr_border_0, corr_border_1], '-', c="r", lw=0.75)
    ax2.plot([corr_border_1,corr_border_1], [corr_border_0, corr_border_1], '-', c="r", lw=0.75)
    
    ax3.imshow(image_reg)
    ax3.set_title("registered", fontsize=14, fontweight="bold")
    ax3.plot([corr_border_0,corr_border_1], [corr_border_0, corr_border_0], '-', c="r", lw=0.75)
    ax3.plot([corr_border_0,corr_border_1], [corr_border_1, corr_border_1], '-', c="r", lw=0.75)
    ax3.plot([corr_border_0,corr_border_0], [corr_border_0, corr_border_1], '-', c="r", lw=0.75)
    ax3.plot([corr_border_1,corr_border_1], [corr_border_0, corr_border_1], '-', c="r", lw=0.75)
    
    ax4.imshow(image-image_reg)
    ax4.set_title("original minus registered", fontsize=14, fontweight="bold")
    ax4.set_xlim((0, image.shape[0]))
    ax4.set_ylim((0, image.shape[1]))
    
    plt.tight_layout()
    plt.savefig(RESULTSpath+plotname+".pdf")
    plt.savefig(RESULTSpath+"jpg/"+plotname+".jpg", dpi=120)

def plot_phase_cross_corr_results(image, image_shifted,image_reg, plotname="phase_cross_corr",
                                  pearson_corr_R=1.0, pearson_corr_R_reg=1.0, corr_border=30,
                                  shifts=[0,0,0]):
    corr_border_0 = corr_border
    corr_border_1 = image.shape[0]-corr_border
    
    plt.close()
    fig = plt.figure(figsize=(8, 8))
    plt.clf()
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(2, 2, 3, sharex=ax1, sharey=ax1)
    ax4 = plt.subplot(2, 2, 4, sharex=ax1, sharey=ax1)
    
    ax4.text(0, 250, f"Correlation coefficients:",  fontsize=13, fontweight="bold", va="top")
    ax4.text(0, 233, f" - image vs. moved:      {np.round(pearson_corr_R, 2)}", fontsize=13, 
             fontweight="bold",  va="top")
    ax4.text(0, 216, f" - image vs. registered: {np.round(pearson_corr_R_reg, 2)}", fontsize=13, 
             fontweight="bold", va="top")
    ax4.text(0, 199, f"(account for area within the", fontsize=13, fontweight="bold", va="top")
    ax4.text(0, 182, f"red rectangle)", fontsize=13, fontweight="bold", va="top")
    ax4.text(0, 145, f"detected ", fontsize=13, fontweight="bold", va="top")
    ax4.text(0, 128, f" - translation: {np.round(shifts[0],1)}, {np.round(shifts[1],1)}", fontsize=13, fontweight="bold", va="top")
    ax4.text(0, 111, f" - rotation:      {np.round(shifts[2],1)}°", fontsize=13, fontweight="bold", va="top")
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.spines['bottom'].set_visible(False)
    ax4.spines['left'].set_visible(False)
    ax4.set_xticks([])
    ax4.set_yticks([])
    
    ax1.imshow(image_shifted)
    ax1.set_title("moved", fontsize=14, fontweight="bold")
    ax1.plot([corr_border_0,corr_border_1], [corr_border_0, corr_border_0], '-', c="r", lw=0.75)
    ax1.plot([corr_border_0,corr_border_1], [corr_border_1, corr_border_1], '-', c="r", lw=0.75)
    ax1.plot([corr_border_0,corr_border_0], [corr_border_0, corr_border_1], '-', c="r", lw=0.75)
    ax1.plot([corr_border_1,corr_border_1], [corr_border_0, corr_border_1], '-', c="r", lw=0.75)
    
    ax2.imshow(image_reg)
    ax2.set_title("registered", fontsize=14, fontweight="bold")
    ax2.plot([corr_border_0,corr_border_1], [corr_border_0, corr_border_0], '-', c="r", lw=0.75)
    ax2.plot([corr_border_0,corr_border_1], [corr_border_1, corr_border_1], '-', c="r", lw=0.75)
    ax2.plot([corr_border_0,corr_border_0], [corr_border_0, corr_border_1], '-', c="r", lw=0.75)
    ax2.plot([corr_border_1,corr_border_1], [corr_border_0, corr_border_1], '-', c="r", lw=0.75)
    
    ax3.imshow(image-image_reg)
    ax3.set_title("original minus registered", fontsize=14, fontweight="bold")
    ax3.set_xlim((0, image.shape[0]))
    ax3.set_ylim((0, image.shape[1]))
    
    plt.tight_layout()
    plt.savefig(RESULTSpath+plotname+".pdf")
    plt.savefig(RESULTSpath+"jpg/"+plotname+".jpg", dpi=120)

def plot_optical_flow_results_old(image, image_shifted,image_reg, x,y,u_,v_, plotname="optical_flow",
                              pearson_corr_R=1.0, pearson_corr_R_reg=1.0, corr_border=30):
    corr_border_0 = corr_border
    corr_border_1 = image.shape[0]-corr_border
    
    
    plt.close()
    fig = plt.figure(figsize=(12, 8))
    plt.clf()
    ax1 = plt.subplot(2, 3, 1)
    ax2 = plt.subplot(2, 3, 2, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(2, 3, 3, sharex=ax1, sharey=ax1)
    ax4 = plt.subplot(2, 3, 5, sharex=ax1, sharey=ax1)
    ax5 = plt.subplot(2, 3, 6, sharex=ax1, sharey=ax1)
    ax6 = plt.subplot(2, 3, 4, sharex=ax1, sharey=ax1)
    
    ax1.imshow(image)
    ax1.set_title("original", fontsize=14, fontweight="bold")
    ax1.plot([corr_border_0,corr_border_1], [corr_border_0, corr_border_0], '-', c="r", lw=0.75)
    ax1.plot([corr_border_0,corr_border_1], [corr_border_1, corr_border_1], '-', c="r", lw=0.75)
    ax1.plot([corr_border_0,corr_border_0], [corr_border_0, corr_border_1], '-', c="r", lw=0.75)
    ax1.plot([corr_border_1,corr_border_1], [corr_border_0, corr_border_1], '-', c="r", lw=0.75)
    
    #ax6.text(0, 200, plotname,  fontsize=9, fontweight="normal")
    ax6.text(0, 180, f"Correlation coefficients:",  fontsize=14, fontweight="bold", va="top")
    ax6.text(0, 160, f"  - image vs. moved:      {np.round(pearson_corr_R, 2)}", fontsize=14, 
             fontweight="bold",  va="top")
    ax6.text(0, 140, f"  - image vs. registered: {np.round(pearson_corr_R_reg, 2)}", fontsize=14, 
             fontweight="bold", va="top")
    ax6.text(0, 120, f"(account for area within the", fontsize=14, fontweight="bold", va="top")
    ax6.text(0, 100, f"red rectangle)", fontsize=14, fontweight="bold", va="top")
    ax6.spines['right'].set_visible(False)
    ax6.spines['top'].set_visible(False)
    ax6.spines['bottom'].set_visible(False)
    ax6.spines['left'].set_visible(False)
    ax6.set_xticks([])
    ax6.set_yticks([])
    
    ax2.imshow(image_shifted)
    ax2.set_title("moved", fontsize=14, fontweight="bold")
    ax2.plot([corr_border_0,corr_border_1], [corr_border_0, corr_border_0], '-', c="r", lw=0.75)
    ax2.plot([corr_border_0,corr_border_1], [corr_border_1, corr_border_1], '-', c="r", lw=0.75)
    ax2.plot([corr_border_0,corr_border_0], [corr_border_0, corr_border_1], '-', c="r", lw=0.75)
    ax2.plot([corr_border_1,corr_border_1], [corr_border_0, corr_border_1], '-', c="r", lw=0.75)
    
    ax3.imshow(image_reg)
    ax3.set_title("registered", fontsize=14, fontweight="bold")
    ax3.plot([corr_border_0,corr_border_1], [corr_border_0, corr_border_0], '-', c="r", lw=0.75)
    ax3.plot([corr_border_0,corr_border_1], [corr_border_1, corr_border_1], '-', c="r", lw=0.75)
    ax3.plot([corr_border_0,corr_border_0], [corr_border_0, corr_border_1], '-', c="r", lw=0.75)
    ax3.plot([corr_border_1,corr_border_1], [corr_border_0, corr_border_1], '-', c="r", lw=0.75)
    
    ax4.imshow(image-image_reg)
    ax4.set_title("original minus registered", fontsize=14, fontweight="bold")
    
    ax5.imshow(image_shifted)
    ax5.quiver(x, y, u_, v_, color='pink', units='dots',
            angles='xy', scale_units='xy', lw=3)
    ax5.set_xlim((0, image.shape[0]))
    ax5.set_ylim((0, image.shape[1]))
    ax5.set_title("opt. flow magnitude & vector field", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(RESULTSpath+plotname+".pdf")
    plt.savefig(RESULTSpath+"jpg/"+plotname+".jpg", dpi=120)
    
def plot_optical_flow_results(image, image_shifted,image_reg, x,y,u_,v_, plotname="optical_flow",
                              pearson_corr_R=1.0, pearson_corr_R_reg=1.0, corr_border=30):
    corr_border_0 = corr_border
    corr_border_1 = image.shape[0]-corr_border
    
    plt.close()
    fig = plt.figure(figsize=(7, 9.0))
    plt.clf()
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(2, 2, 3, sharex=ax1, sharey=ax1)
    ax4 = plt.subplot(2, 2, 4, sharex=ax1, sharey=ax1)
    #ax5 = plt.subplot(3, 2, 5, sharex=ax1, sharey=ax1)
    
    ax1.imshow(image_shifted)
    ax1.set_title("moved", fontsize=14, fontweight="bold")
    ax1.plot([corr_border_0,corr_border_1], [corr_border_0, corr_border_0], '-', c="r", lw=0.75)
    ax1.plot([corr_border_0,corr_border_1], [corr_border_1, corr_border_1], '-', c="r", lw=0.75)
    ax1.plot([corr_border_0,corr_border_0], [corr_border_0, corr_border_1], '-', c="r", lw=0.75)
    ax1.plot([corr_border_1,corr_border_1], [corr_border_0, corr_border_1], '-', c="r", lw=0.75)
    
    ax2.imshow(image_reg)
    ax2.set_title("registered", fontsize=14, fontweight="bold")
    ax2.plot([corr_border_0,corr_border_1], [corr_border_0, corr_border_0], '-', c="r", lw=0.75)
    ax2.plot([corr_border_0,corr_border_1], [corr_border_1, corr_border_1], '-', c="r", lw=0.75)
    ax2.plot([corr_border_0,corr_border_0], [corr_border_0, corr_border_1], '-', c="r", lw=0.75)
    ax2.plot([corr_border_1,corr_border_1], [corr_border_0, corr_border_1], '-', c="r", lw=0.75)
    
    ax4.imshow(image-image_reg)
    ax4.set_title("original minus registered", fontsize=14, fontweight="bold")
    
    ax3.imshow(image_shifted)
    ax3.quiver(x, y, u_, v_, color='pink', units='dots',
            angles='xy', scale_units='xy', lw=3)
    ax3.set_xlim((0, image.shape[0]))
    ax3.set_ylim((0, image.shape[1]))
    ax3.set_title("optical flow vector field", fontsize=14, fontweight="bold")
    
    text_fontsize= 12
    ax3.text(0, -8, f"Correlation coefficients:",  fontsize=text_fontsize, fontweight="bold", va="top")
    ax3.text(0, -26, f"  - image vs. moved:      {np.round(pearson_corr_R, 2)}", fontsize=text_fontsize, 
             fontweight="bold",  va="top")
    ax3.text(0, -44, f"  - image vs. registered: {np.round(pearson_corr_R_reg, 2)}", fontsize=text_fontsize, 
             fontweight="bold", va="top")
    ax3.text(0, -62, f"(account for area within the", fontsize=text_fontsize, fontweight="bold", va="top")
    ax3.text(0, -79, f"red rectangle)", fontsize=text_fontsize, fontweight="bold", va="top")
    ax3.text(0, -106, f"", fontsize=text_fontsize, fontweight="bold", va="top")
    ax3.set_xticks([])
    ax3.set_yticks([]) 
    
    plt.tight_layout()
    plt.savefig(RESULTSpath+plotname+".pdf")
    plt.savefig(RESULTSpath+"jpg/"+plotname+".jpg", dpi=120)
    
def do_optical_flow_ilk(image, image_shifted, shift_type="rigid translation", corr_border=30,
                        radius=15, prefilter=False, num_warp=30, gaussian=False, 
                        median_filter_kernel=1):
    
    v, u = optical_flow_ilk(median_filter(image, size=median_filter_kernel), 
                            median_filter(image_shifted, size=median_filter_kernel), 
                            radius=radius, prefilter=prefilter, 
                            num_warp=num_warp, gaussian=gaussian)
    nr, nc = image.shape
    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
    image_reg= warp(image_shifted, np.array([row_coords + v, col_coords + u]), 
                    mode='constant', preserve_range=True).astype(image.dtype)

    pearson_corr_R     = sp.stats.pearsonr(image[corr_border:-corr_border, corr_border:-corr_border].flatten(), 
                                           image_shifted[corr_border:-corr_border, corr_border:-corr_border].flatten())[0]
    pearson_corr_R_reg = sp.stats.pearsonr(image[corr_border:-corr_border, corr_border:-corr_border].flatten(), 
                                           image_reg[corr_border:-corr_border, corr_border:-corr_border].flatten())[0]
    print(f"Pearson correlation coefficient image vs. moved: {pearson_corr_R}")
    print(f"Pearson correlation coefficient image vs. registration: {pearson_corr_R_reg}")

    nvec = 20  # Number of vectors to be displayed along each image dimension
    step = max(nr//nvec, nc//nvec)
    y, x = np.mgrid[:nr:step, :nc:step]
    #u_ = -u[::step, ::step]
    #v_ = -v[::step, ::step]
    u = -u[::step, ::step]
    v = -v[::step, ::step]

    plot_optical_flow_results(image, image_shifted, image_reg, x,y,u,v, corr_border=corr_border,
                            plotname="optical_flow_ilk "+shift_type+" mf="+str(median_filter_kernel),
                            pearson_corr_R=pearson_corr_R, pearson_corr_R_reg=pearson_corr_R_reg)

def do_optical_flow_tvl1(image, image_shifted, shift_type="rigid translation", corr_border=30,
                         attachment=40, tightness=0.2, num_warp=15, 
                         num_iter=10, tol=0.0001, prefilter=False, median_filter_kernel=1):
    
    v, u = optical_flow_tvl1(median_filter(image, size=median_filter_kernel), 
                             median_filter(image_shifted, size=median_filter_kernel), 
                             attachment=attachment, tightness=tightness, 
                             num_warp=num_warp, num_iter=num_iter, tol=tol, prefilter=prefilter)
    nr, nc = image.shape
    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
    image_reg= warp(image_shifted, np.array([row_coords + v, col_coords + u]), 
                    mode='constant', preserve_range=True).astype(image.dtype)

    nvec = 20  # Number of vectors to be displayed along each image dimension
    step = max(nr//nvec, nc//nvec)
    y, x = np.mgrid[:nr:step, :nc:step]
    u = -u[::step, ::step]
    v = -v[::step, ::step]

    pearson_corr_R     = sp.stats.pearsonr(image[corr_border:-corr_border, corr_border:-corr_border].flatten(), 
                                           image_shifted[corr_border:-corr_border, corr_border:-corr_border].flatten())[0]
    pearson_corr_R_reg = sp.stats.pearsonr(image[corr_border:-corr_border, corr_border:-corr_border].flatten(), 
                                           image_reg[corr_border:-corr_border, corr_border:-corr_border].flatten())[0]
    print(f"Pearson correlation coefficient image vs. moved: {pearson_corr_R}")
    print(f"Pearson correlation coefficient image vs. registration: {pearson_corr_R_reg}")

    plot_optical_flow_results(image, image_shifted, image_reg, x,y,u,v, corr_border=corr_border,
                                plotname="optical_flow_tvl1 "+shift_type+" mf="+str(median_filter_kernel), 
                                pearson_corr_R=pearson_corr_R, pearson_corr_R_reg=pearson_corr_R_reg)

def do_calcOpticalFlowFarneback(image, image_shifted, shift_type="rigid translation", corr_border=30,
                                median_filter_kernel=1, pyr_scale=0.5, levels=15, winsize=7,
                                iterations=10, poly_n=5, poly_sigma=10, flags=0, flow=None):
    
    flow = cv.calcOpticalFlowFarneback(prev=median_filter(image, size=median_filter_kernel), 
                                       next=median_filter(image_shifted, size=median_filter_kernel), 
                                       pyr_scale=pyr_scale, 
                                       levels=levels, winsize=winsize, iterations=iterations, 
                                       poly_n=poly_n, poly_sigma=poly_sigma, flags=flags, flow=flow)
    h, w = flow.shape[:2]
    flow_trans = flow.copy()
    flow_trans[:,:,0] += np.arange(w)
    flow_trans[:,:,1] += np.arange(h)[:,np.newaxis]
    image_reg = cv.remap(image_shifted, flow_trans, None, cv.INTER_NEAREST)

    pearson_corr_R     = sp.stats.pearsonr(image[corr_border:-corr_border, corr_border:-corr_border].flatten(), 
                                           image_shifted[corr_border:-corr_border, corr_border:-corr_border].flatten())[0]
    pearson_corr_R_reg = sp.stats.pearsonr(image[corr_border:-corr_border, corr_border:-corr_border].flatten(), 
                                           image_reg[corr_border:-corr_border, corr_border:-corr_border].flatten())[0]
    print(f"Pearson correlation coefficient image vs. moved: {pearson_corr_R}")
    print(f"Pearson correlation coefficient image vs. registration: {pearson_corr_R_reg}")


    nvec = 20  # Number of vectors to be displayed along each image dimension
    step = max(h//nvec, w//nvec)
    u = -flow[::step, ::step, 0]
    v = -flow[::step, ::step, 1]
    x = np.arange(0, flow.shape[1], step)
    y = np.arange(flow.shape[0], -1, -step)
    #plt.quiver(x, y, u,v)

    plot_optical_flow_results(image, image_shifted, image_reg, x,y,u,v, corr_border=corr_border,
                              plotname="calcOpticalFlowFarneback "+shift_type+" mf="+str(median_filter_kernel), 
                              pearson_corr_R=pearson_corr_R, pearson_corr_R_reg=pearson_corr_R_reg)

def do_phase_cross_corr(image, image_shifted, upsample_factor=1, shift_type="translation",
                        detect_rotation=False, corr_border=30, median_filter_kernel=1, 
                        normalization="phase"):
    if detect_rotation:
        radius=800
        image_polar = warp_polar(median_filter(image, size=median_filter_kernel), 
                                 radius=radius, preserve_range=True)
        image_shifted_polar = warp_polar(median_filter(image_shifted, size=median_filter_kernel), 
                                         radius=radius, preserve_range=True)
        shift_rot, _, _ = phase_cross_correlation(image_polar, image_shifted_polar, 
                                                  upsample_factor=upsample_factor, 
                                                  normalization=normalization)
        shift_rot = shift_rot[0]
        print(f'Recovered value for counterclockwise rotation: {shift_rot}°')
        image_reg = rotate(image_shifted, -shift_rot, preserve_range=True)
    else:
        shift_rot=np.nan
        image_reg = image_shifted
    shift, _, _ = phase_cross_correlation(median_filter(image, size=median_filter_kernel), 
                                          median_filter(image_reg, size=median_filter_kernel), 
                                          upsample_factor=upsample_factor, normalization=normalization)
    tform = SimilarityTransform(translation=(-shift[1], -shift[0]))
    image_reg = warp(image_reg, tform, preserve_range=True).astype(image_shifted.dtype)
        
    shifts=[shift[1], shift[0], shift_rot]
    pearson_corr_R     = sp.stats.pearsonr(image[corr_border:-corr_border, corr_border:-corr_border].flatten(), 
                                           image_shifted[corr_border:-corr_border, corr_border:-corr_border].flatten())[0]
    pearson_corr_R_reg = sp.stats.pearsonr(image[corr_border:-corr_border, corr_border:-corr_border].flatten(), 
                                           image_reg[corr_border:-corr_border, corr_border:-corr_border].flatten())[0]
    print(f"Pearson correlation coefficient image vs. moved: {pearson_corr_R}")
    print(f"Pearson correlation coefficient image vs. registration: {pearson_corr_R_reg}")

    plot_phase_cross_corr_results(image, image_shifted, image_reg, shifts=shifts,
                                  plotname="phase_cross_corr "+shift_type+" mf="+str(median_filter_kernel),
                                  pearson_corr_R=pearson_corr_R, pearson_corr_R_reg=pearson_corr_R_reg, 
                                  corr_border=corr_border)
# %% CREATE SOME TOY DATA
image = data.cells3d()[30,1,:,:]
plot_2Dimage(image, title="original image", path=RESULTSpath)
np.random.seed(1)
noise = np.random.normal(0, 0.2, image.shape)*image.mean()
image = image+noise
plot_2Dimage(image, title="original image w noise", path=RESULTSpath)

tform = SimilarityTransform(translation=(-30,15))
image_rigid_translated = warp(image, tform, preserve_range=True).astype(image.dtype)
plot_2Dimage(image_rigid_translated, title="image rigid translated", path=RESULTSpath)

image_rigid_rotated = rotate(image, angle=5, preserve_range=True)
plot_2Dimage(image_rigid_rotated, title="image rigid rotated", path=RESULTSpath)

image_rigid_translated_rotated = rotate(image_rigid_translated, angle=5, preserve_range=True)
plot_2Dimage(image_rigid_translated_rotated, title="image rigid translated rotated", path=RESULTSpath)
plot_2Dimage(image+image_rigid_translated_rotated, title="image rigid translated rotated overlay", 
             path=RESULTSpath)

tform = SimilarityTransform(translation=(-10,5))
image_rigid_translated_rotated_less = rotate(warp(image, tform, preserve_range=True), 
                                             angle=5, preserve_range=True)

image_moved_cell = data.cells3d()[30,1,:,:]
cell = image_moved_cell[130:180, 80:140].copy()
image_moved_cell[130:180, 80:140] = np.repeat(image_moved_cell[185:190, 80:140], 10, axis=0)
image_moved_cell[112:162, 73:133] = cell.copy()
image_moved_cell = image_moved_cell+noise
plot_2Dimage(image_moved_cell, title="image with moved cell", path=RESULTSpath)

image_moved_cell_brightness = data.cells3d()[30,1,:,:]
cell = image_moved_cell_brightness[130:180, 80:140].copy()*0.5
image_moved_cell_brightness[130:180, 80:140] = np.repeat(image_moved_cell_brightness[185:190, 80:140], 10, axis=0)
image_moved_cell_brightness[112:162, 73:133] = cell.copy()
image_moved_cell_brightness = image_moved_cell_brightness+noise
plot_2Dimage(image_moved_cell_brightness, title="image with moved cell (dimmed)", path=RESULTSpath)

image_moved_cell_brightness_global = data.cells3d()[30,1,:,:]
cell = image_moved_cell_brightness_global[130:180, 80:140].copy()
image_moved_cell_brightness_global[130:180, 80:140] = np.repeat(image_moved_cell_brightness_global[185:190, 80:140], 10, axis=0)
image_moved_cell_brightness_global[112:162, 73:133] = cell.copy()
image_moved_cell_brightness_global = (image_moved_cell_brightness_global+noise)*0.60
plot_2Dimage(image_moved_cell_brightness_global, title="image with moved cell (globally dimmed)", path=RESULTSpath)

image_dichotomized = np.zeros((image.shape), dtype=image.dtype)
image_dichotomized[:128,30:] = image[:128,0:-30]
image_dichotomized[128:,:-30]= image[128:,30:]
plot_2Dimage(image_dichotomized, title="image dichotomized", path=RESULTSpath)

image_dichotomized_sheared = np.zeros((image.shape), dtype=image.dtype)
tform = AffineTransform(shear=np.pi/35)
image_dichotomized_sheared[:128,30:] = warp(image[:128,0:-30], tform, preserve_range=True).astype(image.dtype)
image_dichotomized_sheared[128:,:-30]= warp(image[128:,30:], tform.inverse, preserve_range=True).astype(image.dtype)
plot_2Dimage(image_dichotomized_sheared, title="image dichotomized sheared", path=RESULTSpath)
# %% REGISTER TRANSLATED
image_shifted = image_rigid_translated
shift_type    = "translation"

do_optical_flow_ilk(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=1,
                    radius=30, prefilter=False, num_warp=10, gaussian=False)
do_optical_flow_ilk(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=3,
                    radius=30, prefilter=False, num_warp=10, gaussian=False)

do_optical_flow_tvl1(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=1,
                     attachment=60, tightness=0.99, num_warp=15,
                     num_iter=15, tol=0.0001, prefilter=False)
do_optical_flow_tvl1(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=3,
                     attachment=60, tightness=0.99, num_warp=15,
                     num_iter=15, tol=0.0001, prefilter=False)

do_calcOpticalFlowFarneback(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=1,
                            pyr_scale=0.5, levels=15, winsize=7, iterations=10, poly_n=5, 
                            poly_sigma=10, flags=0, flow=None)
do_calcOpticalFlowFarneback(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=3,
                            pyr_scale=0.5, levels=15, winsize=7, iterations=10, poly_n=5, 
                            poly_sigma=10, flags=0, flow=None)

do_phase_cross_corr(image, image_shifted, upsample_factor=30, shift_type=shift_type,
                    detect_rotation=False, corr_border=30, median_filter_kernel=1)
do_phase_cross_corr(image, image_shifted, upsample_factor=30, shift_type=shift_type,
                    detect_rotation=False, corr_border=30, median_filter_kernel=3)
# %% REGISTER ROTATED
image_shifted = image_rigid_rotated
shift_type    = "rotation"

do_optical_flow_ilk(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=1,
                    radius=30, prefilter=False, num_warp=10, gaussian=False)
do_optical_flow_ilk(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=3,
                    radius=30, prefilter=False, num_warp=10, gaussian=False)

do_optical_flow_tvl1(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=1,
                     attachment=60, tightness=0.99, num_warp=15,
                     num_iter=15, tol=0.0001, prefilter=False)
do_optical_flow_tvl1(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=3,
                     attachment=60, tightness=0.99, num_warp=15,
                     num_iter=15, tol=0.0001, prefilter=False)

do_calcOpticalFlowFarneback(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=1,
                            pyr_scale=0.5, levels=15, winsize=7, iterations=10, poly_n=5, 
                            poly_sigma=10, flags=0, flow=None)
do_calcOpticalFlowFarneback(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=3,
                            pyr_scale=0.5, levels=15, winsize=7, iterations=10, poly_n=5, 
                            poly_sigma=10, flags=0, flow=None)

do_phase_cross_corr(image, image_shifted, upsample_factor=30, shift_type=shift_type,
                    detect_rotation=True, corr_border=30, median_filter_kernel=1)
do_phase_cross_corr(image, image_shifted, upsample_factor=30, shift_type=shift_type,
                    detect_rotation=True, corr_border=30, median_filter_kernel=3)

""" flow = cv.calcOpticalFlowFarneback(prev=image, next=image_shifted, pyr_scale=0.5, levels=15, winsize=50,
                                    iterations=10, poly_n=7, poly_sigma=0, flags=0, flow=None) """
# %% REGISTER TRANSLATED ROTATED
image_shifted = image_rigid_translated_rotated
shift_type    = "translation rotation"

do_optical_flow_ilk(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=1,
                    radius=30, prefilter=False, num_warp=10, gaussian=False)
do_optical_flow_ilk(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=3,
                    radius=30, prefilter=False, num_warp=10, gaussian=False)

do_optical_flow_tvl1(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=1,
                     attachment=60, tightness=0.99, num_warp=15,
                     num_iter=15, tol=0.0001, prefilter=False)
do_optical_flow_tvl1(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=3,
                     attachment=60, tightness=0.99, num_warp=15,
                     num_iter=15, tol=0.0001, prefilter=False)

do_calcOpticalFlowFarneback(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=1,
                            pyr_scale=0.5, levels=15, winsize=7, iterations=10, poly_n=5, 
                            poly_sigma=10, flags=0, flow=None)
do_calcOpticalFlowFarneback(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=3,
                            pyr_scale=0.5, levels=15, winsize=7, iterations=10, poly_n=5, 
                            poly_sigma=10, flags=0, flow=None)

do_phase_cross_corr(image, image_shifted, upsample_factor=30, shift_type=shift_type,
                    detect_rotation=True, corr_border=30, median_filter_kernel=1)
do_phase_cross_corr(image, image_shifted, upsample_factor=30, shift_type=shift_type,
                    detect_rotation=True, corr_border=30, median_filter_kernel=3)

image_shifted = image_rigid_translated_rotated_less
shift_type    = "translation rotation less"

do_phase_cross_corr(image, image_shifted, upsample_factor=30, shift_type=shift_type,
                    detect_rotation=True, corr_border=30, median_filter_kernel=17, normalization=None)
# %% REGISTER TRANSLATED ROTATED (LK TESTS)
image_shifted = image_rigid_translated_rotated
shift_type    = "translation rotation LK"

image2 = data.cells3d()[45,1,:,:]
image2 = image2+noise
plot_2Dimage(image2, title="original image 2 w noise", path=RESULTSpath)

do_optical_flow_ilk(image, image2, shift_type, corr_border=30, median_filter_kernel=1,
                    radius=30, prefilter=False, num_warp=10, gaussian=False)

do_phase_cross_corr(image, image2, upsample_factor=30, shift_type=shift_type,
                    detect_rotation=True, corr_border=30, median_filter_kernel=1, normalization="phase")

# %% REGISTER DICHOTOMIZED
image_shifted = image_dichotomized
shift_type    = "dichotomized"

do_optical_flow_ilk(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=1,
                    radius=30, prefilter=False, num_warp=10, gaussian=False)
do_optical_flow_ilk(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=3,
                    radius=30, prefilter=False, num_warp=10, gaussian=False)

do_optical_flow_tvl1(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=1,
                     attachment=60, tightness=0.99, num_warp=15,
                     num_iter=15, tol=0.0001, prefilter=False)
do_optical_flow_tvl1(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=3,
                     attachment=60, tightness=0.99, num_warp=15,
                     num_iter=15, tol=0.0001, prefilter=False)

do_calcOpticalFlowFarneback(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=1,
                            pyr_scale=0.75, levels=15, winsize=8, iterations=10, poly_n=3, 
                            poly_sigma=15, flags=0, flow=None)
do_calcOpticalFlowFarneback(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=3,
                            pyr_scale=0.75, levels=15, winsize=8, iterations=10, poly_n=3, 
                            poly_sigma=15, flags=0, flow=None)
# %% REGISTER DICHOTOMIZED SHEARED
image_shifted = image_dichotomized_sheared
shift_type    = "dichotomized sheared"

do_optical_flow_ilk(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=1,
                    radius=10, prefilter=False, num_warp=15, gaussian=True)
do_optical_flow_ilk(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=3,
                    radius=10, prefilter=False, num_warp=15, gaussian=True)

do_optical_flow_tvl1(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=1,
                     attachment=100, tightness=1.20, num_warp=15,
                     num_iter=15, tol=0.1, prefilter=True)
do_optical_flow_tvl1(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=3,
                     attachment=160, tightness=0.8, num_warp=15,
                     num_iter=15, tol=0.1, prefilter=True)

do_calcOpticalFlowFarneback(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=1,
                            pyr_scale=0.6, levels=15, winsize=8, iterations=10, poly_n=7, 
                            poly_sigma=0, flags=0, flow=None)
do_calcOpticalFlowFarneback(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=3,
                            pyr_scale=0.6, levels=15, winsize=8, iterations=10, poly_n=7, 
                            poly_sigma=0, flags=0, flow=None)
# %% REGISTER MOVED CELL
image_shifted = image_moved_cell
shift_type    = "moved cell"

do_optical_flow_ilk(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=1,
                    radius=10, prefilter=False, num_warp=15, gaussian=True)
do_optical_flow_ilk(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=3,
                    radius=10, prefilter=False, num_warp=15, gaussian=True)

do_optical_flow_tvl1(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=1,
                     attachment=100, tightness=1.20, num_warp=15,
                     num_iter=15, tol=0.1, prefilter=True)
do_optical_flow_tvl1(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=3,
                     attachment=160, tightness=0.8, num_warp=15,
                     num_iter=15, tol=0.1, prefilter=True)

do_calcOpticalFlowFarneback(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=1,
                            pyr_scale=0.6, levels=15, winsize=8, iterations=10, poly_n=7, 
                            poly_sigma=0, flags=0, flow=None)
do_calcOpticalFlowFarneback(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=3,
                            pyr_scale=0.6, levels=15, winsize=8, iterations=10, poly_n=7, 
                            poly_sigma=0, flags=0, flow=None)
# %% REGISTER MOVED CELL BRIGHTNESS
image_shifted = image_moved_cell_brightness
shift_type    = "moved cell brightness"

do_optical_flow_ilk(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=1,
                    radius=10, prefilter=False, num_warp=15, gaussian=True)
do_optical_flow_ilk(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=3,
                    radius=10, prefilter=False, num_warp=15, gaussian=True)

do_optical_flow_tvl1(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=1,
                     attachment=100, tightness=1.20, num_warp=15,
                     num_iter=15, tol=0.1, prefilter=True)
do_optical_flow_tvl1(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=3,
                     attachment=160, tightness=0.8, num_warp=15,
                     num_iter=15, tol=0.1, prefilter=True)

do_calcOpticalFlowFarneback(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=1,
                            pyr_scale=0.6, levels=15, winsize=8, iterations=10, poly_n=7, 
                            poly_sigma=0, flags=0, flow=None)
do_calcOpticalFlowFarneback(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=3,
                            pyr_scale=0.6, levels=15, winsize=8, iterations=10, poly_n=7, 
                            poly_sigma=0, flags=0, flow=None)

shift_type    = "moved cell brightness hist_adjusted"
image_shifted_histadjusted = match_histograms(image_shifted, image)
do_optical_flow_ilk(image, image_shifted_histadjusted, shift_type, corr_border=30, median_filter_kernel=1,
                    radius=10, prefilter=False, num_warp=15, gaussian=True)
do_optical_flow_ilk(image, image_shifted_histadjusted, shift_type, corr_border=30, median_filter_kernel=3,
                    radius=10, prefilter=False, num_warp=15, gaussian=True)

do_optical_flow_tvl1(image, image_shifted_histadjusted, shift_type, corr_border=30, median_filter_kernel=1,
                     attachment=100, tightness=1.20, num_warp=15,
                     num_iter=15, tol=0.1, prefilter=True)
do_optical_flow_tvl1(image, image_shifted_histadjusted, shift_type, corr_border=30, median_filter_kernel=3,
                     attachment=160, tightness=0.8, num_warp=15,
                     num_iter=15, tol=0.1, prefilter=True)

do_calcOpticalFlowFarneback(image, image_shifted_histadjusted, shift_type, corr_border=30, median_filter_kernel=1,
                            pyr_scale=0.6, levels=15, winsize=8, iterations=10, poly_n=7, 
                            poly_sigma=0, flags=0, flow=None)
do_calcOpticalFlowFarneback(image, image_shifted_histadjusted, shift_type, corr_border=30, median_filter_kernel=3,
                            pyr_scale=0.6, levels=15, winsize=8, iterations=10, poly_n=7, 
                            poly_sigma=0, flags=0, flow=None)
# %% REGISTER MOVED CELL GLOBAL BRIGHTNESS
image_shifted = image_moved_cell_brightness_global
shift_type    = "moved cell brightness global"

do_optical_flow_ilk(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=1,
                    radius=10, prefilter=False, num_warp=15, gaussian=True)
do_optical_flow_ilk(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=3,
                    radius=10, prefilter=False, num_warp=15, gaussian=True)

do_optical_flow_tvl1(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=1,
                     attachment=100, tightness=1.20, num_warp=15,
                     num_iter=15, tol=0.1, prefilter=True)
do_optical_flow_tvl1(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=3,
                     attachment=160, tightness=0.8, num_warp=15,
                     num_iter=15, tol=0.1, prefilter=True)

do_calcOpticalFlowFarneback(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=1,
                            pyr_scale=0.6, levels=15, winsize=8, iterations=10, poly_n=7, 
                            poly_sigma=0, flags=0, flow=None)
do_calcOpticalFlowFarneback(image, image_shifted, shift_type, corr_border=30, median_filter_kernel=3,
                            pyr_scale=0.6, levels=15, winsize=8, iterations=10, poly_n=7, 
                            poly_sigma=0, flags=0, flow=None)

shift_type    = "moved cell brightness global hist_adjusted"
image_shifted_histadjusted = match_histograms(image_shifted, image)
do_optical_flow_ilk(image, image_shifted_histadjusted, shift_type, corr_border=30, median_filter_kernel=1,
                    radius=10, prefilter=False, num_warp=15, gaussian=True)
do_optical_flow_ilk(image, image_shifted_histadjusted, shift_type, corr_border=30, median_filter_kernel=3,
                    radius=10, prefilter=False, num_warp=15, gaussian=True)

do_optical_flow_tvl1(image, image_shifted_histadjusted, shift_type, corr_border=30, median_filter_kernel=1,
                     attachment=100, tightness=1.20, num_warp=15,
                     num_iter=15, tol=0.1, prefilter=True)
do_optical_flow_tvl1(image, image_shifted_histadjusted, shift_type, corr_border=30, median_filter_kernel=3,
                     attachment=160, tightness=0.8, num_warp=15,
                     num_iter=15, tol=0.1, prefilter=True)

do_calcOpticalFlowFarneback(image, image_shifted_histadjusted, shift_type, corr_border=30, median_filter_kernel=1,
                            pyr_scale=0.6, levels=15, winsize=8, iterations=10, poly_n=7, 
                            poly_sigma=0, flags=0, flow=None)
do_calcOpticalFlowFarneback(image, image_shifted_histadjusted, shift_type, corr_border=30, median_filter_kernel=3,
                            pyr_scale=0.6, levels=15, winsize=8, iterations=10, poly_n=7, 
                            poly_sigma=0, flags=0, flow=None)
# %% END
