# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 13:58:56 2016

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
import scipy.ndimage.filters as fi

#%%
def plot_directions(angle, intensite,pas,taille=1):
    
    S0 = angle.shape[0]
    S1 = angle.shape[1]
    
    y,x = np.ogrid[0:S0,0:S1]
 
    angle2 = angle

    deb_x = np.tile(x,(S0,1)) - taille*np.sin(angle2) * intensite
    deb_y = np.tile(y,(1,S1)) - taille*np.cos(angle2) * intensite
    
    fin_x = np.tile(x,(S0,1)) + taille*np.sin(angle2) * intensite
    fin_y = np.tile(y,(1,S1)) + taille*np.cos(angle2) * intensite
    
    
    for i in range(int(pas/2.),S0,pas):
        for j in range(int(pas/2.),S1,pas):
#            if angle[i,j] != 0:
        
                plt.plot((deb_x[i,j],fin_x[i,j]), (deb_y[i,j],fin_y[i,j]) ,'k', linewidth=2)
    plt.xlim((-0.5,S1-0.5))
    plt.ylim((-0.5,S0-0.5))    
    
#%% 
numexp = '2'
dat = np.load('./data/res_pami'+numexp+'.npz')

Y, X_mpm_est, Ux_map, X_mpm_hmf, V_mpm_est, Uv_map = dat['Y'], dat['X_mpm_est'], dat['Ux_map'], dat['X_mpm_hmf'], dat['V_mpm_est'], dat['Uv_map']


cm_gris = plt.cm.Greys

#%%
plt.close('all')
fig = plt.figure(figsize=(6,6))
ax = plt.Axes(fig, [0.,0.,1.,1.])
fig.add_axes(ax)
#if nom == 'mars':
ax.imshow(Y.mean(axis=2), interpolation='nearest', origin='lower', cmap=plt.cm.copper)#,vmin = -1,vmax=1); 
#else:
#    ax.imshow(Y.mean(axis=2), interpolation='nearest', origin='lower', cmap=plt.cm.Greens_r,vmax=0.75)
   
plt.axis('off')
#plt.savefig('./figures/'+nom+'2a.png', format='png',dpi=200)


#%%
fig = plt.figure(figsize=(6,6))
ax = plt.Axes(fig, [0.,0.,1.,1.])
fig.add_axes(ax)
ax.imshow(X_mpm_est, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r); 
plt.axis('off')
#plt.savefig('./figures/'+nom+'2b.png', format='png',dpi=200)

#%%

fig = plt.figure(figsize=(6,6))
ax = plt.Axes(fig, [0.,0.,1.,1.])
fig.add_axes(ax)
ax.imshow(Ux_map, interpolation='nearest', origin='lower', cmap=plt.cm.gray,vmin=0,vmax=1); 
plt.axis('off')
#plt.savefig('./figures/'+nom+'2c.png', format='png',dpi=200)
#%%
fig = plt.figure(figsize=(6,6))
ax = plt.Axes(fig, [0.,0.,1.,1.])
fig.add_axes(ax)
ax.imshow(X_mpm_hmf, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r); 
plt.axis('off')
#plt.savefig('./figures/'+nom+'2d.png', format='png',dpi=200)

#%%
fig = plt.figure(figsize=(6,6))
ax = plt.Axes(fig, [0.,0.,1.,1.])
fig.add_axes(ax)
if nom == 'mars':
    ax.imshow(Y.mean(axis=2), interpolation='nearest', origin='lower', cmap=plt.cm.copper, alpha=0.25)#,vmin = -1,vmax=1); 
else:
    ax.imshow(Y.mean(axis=2), interpolation='nearest', origin='lower', cmap=plt.cm.Greens_r,vmax=0.75,alpha=0.25)
# 
#ax.imshow(X_mpm_hmf, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r,alpha=0.); 
#ax.imshow(V_mpm_est, interpolation='nearest', origin='lower', cmap=plt.cm.hsv,alpha=0.5)
    
V_fi = fi.median_filter(V_mpm_est, size=6)
plot_directions(V_fi, np.ones_like(V_mpm_est),pas=6,taille=2)
plt.axis('off')
#plt.savefig('./figures/'+nom+'2e.png', format='png',dpi=200)
#%%
##%%
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
##if nom == 'mars':
##    ax.imshow(Y.mean(axis=2), interpolation='nearest', origin='lower', cmap=plt.cm.copper, alpha=0.25)#,vmin = -1,vmax=1); 
##else:
##    ax.imshow(Y.mean(axis=2), interpolation='nearest', origin='lower', cmap=plt.cm.Greens_r,vmax=0.75,alpha=0.25)
### 
##ax.imshow(X_mpm_hmf, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r,alpha=0.); 
#ax.imshow(V_mpm_est, interpolation='nearest', origin='lower', cmap=plt.cm.Spectral,alpha=0.5)
#V_fi = fi.median_filter(V_mpm_est, size=6)
#plot_directions(V_fi, np.ones_like(V_mpm_est),pas=6,taille=2)
#%%



#
#
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(V_mpm_est, interpolation='nearest', origin='lower', cmap=plt.cm.Spectral,alpha=0.5)
##plot_directions(V_mpm_est, np.ones_like(V_mpm_est),pas=6,taille=2)

#%%
fig = plt.figure(figsize=(6,6))
ax = plt.Axes(fig, [0.,0.,1.,1.])
fig.add_axes(ax)
ax.imshow(Uv_map, interpolation='nearest', origin='lower', cmap=plt.cm.gray,vmin=0,vmax=1); 
plt.axis('off')
plt.savefig('./figures/'+nom+'2f.png', format='png',dpi=200)

#np.savez('./data/res_vine2',X_mpm_est = X_mpm_est, X_mpm_hmf = X_mpm_hmf, V_mpm_est = V_mpm_est, Y = Y, parsem = parsem, parsem_hmf = parsem_hmf)
