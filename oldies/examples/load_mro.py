# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 09:17:15 2016

@author: courbot
"""


import numpy as np 
import sys 
import matplotlib.pyplot as plt
import time
#import scipy.stats as st
#import scipy.cluster.vq as cvq
#import multiprocessing as mp
#import image_tools as it
#from scipy.ndimage import imread
#import matplotlib.mlab as mlab
#import numpy.ma as ma
import parameters
#import image_tools as it
import gibbs_sampler as gs
import fields_tools as ft
import seg_OTMF as sot
import SEM as sem

from scipy.ndimage import zoom
from scipy.ndimage.filters import gaussian_filter 
from scipy.ndimage.filters import median_filter 

import gdal

import spectral.io.envi as envi


img2 = envi.open('./data/donnees_ipag/FRT39DF/psp1981_Red_2p5m_proj_warp_p2_subset.hdr','./data/donnees_ipag/FRT39DF/psp1981_Red_2p5m_proj_warp_p2_subset')
data2 = np.copy(img2.asarray()).astype(float)
Y = data2[:,:,0]
Y[Y==0] +=np.nan
fig = plt.figure(figsize=(15,15))
ax = plt.Axes(fig, [0.,0.,1.,1.])
fig.add_axes(ax)
ax.imshow(Y,origin='lower', interpolation='nearest', cmap=plt.cm.copper_r,vmin =0,)
plt.axis("off")
plt.savefig('./figures/illu_mro.png', format='png',dpi=200)
