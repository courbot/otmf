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
                plt.plot((deb_x[i,j],fin_x[i,j]), (deb_y[i,j],fin_y[i,j]) ,'k',linewidth=2)
    plt.xlim((-0.5,S1-0.5))
    plt.ylim((-0.5,S0-0.5))    
    
    
cm_gris = plt.cm.Greys
plt.close('all')
  
dat = np.load('./data/exp_pami1.npz')
X1=dat['X']
V1=dat['V']
dat = np.load('./data/exp_pamiA.npz')
X2=dat['X']
V2 = np.pi/4 * np.ones(shape=(128,128))
V2[64:,:] = 3*np.pi/4
V2[:,64:] = (V2[:,64:] + np.pi/2)%np.pi
dat = np.load('./data/exp_pami3v3.npz')
X3=dat['X']>0

#%%


#%%
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(X1, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r); 
#plt.axis('off')
##plt.savefig('./figures/exp_pami1a.png', format='png',dpi=200)
#
##%%
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(V1, interpolation='nearest', origin='lower', cmap=plt.cm.Spectral, vmin=0, vmax=np.pi,alpha=0.25); 
#plot_directions(V1, np.ones_like(V1),pas=4,taille=1.5)
#plt.axis('off')
#plt.savefig('./figures/exp_pami1b.png', format='png',dpi=200)


fig = plt.figure(figsize=(6,6))
ax = plt.Axes(fig, [0.,0.,1.,1.])
fig.add_axes(ax)
ax.imshow(X2, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r); 
plt.axis('off')
plt.savefig('./figures/exp_pamiAx.png', format='png',dpi=200)

#%%
Y2 = X2 + np.random.normal(loc=0,scale=1,size=(128,128))
fig = plt.figure(figsize=(6,6))
ax = plt.Axes(fig, [0.,0.,1.,1.])
fig.add_axes(ax)
ax.imshow(Y2, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r,vmin=-1,vmax=2); 
plt.axis('off')
#plt.savefig('./figures/exp_pamiAx.png', format='png',dpi=200)
#%%
fig = plt.figure(figsize=(6,6))
ax = plt.Axes(fig, [0.,0.,1.,1.])
fig.add_axes(ax)
ax.imshow(V2, interpolation='nearest', origin='lower', cmap=plt.cm.Spectral, vmin=0, vmax=np.pi,alpha=0.25); 
plot_directions(V2, np.ones_like(V2),pas=4,taille=1.5)
plt.axis('off')
plt.savefig('./figures/exp_pamiAv.png', format='png',dpi=200)
#%%

fig = plt.figure(figsize=(6,6))
ax = plt.Axes(fig, [0.,0.,1.,1.])
fig.add_axes(ax)
ax.imshow(X3, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r); 
plt.axis('off')
plt.savefig('./figures/exp_pamiBx.png', format='png',dpi=200)

#%%
Y3 = X3 + np.random.normal(loc=0,scale=1,size=(128,128))
fig = plt.figure(figsize=(6,6))
ax = plt.Axes(fig, [0.,0.,1.,1.])
fig.add_axes(ax)
ax.imshow(Y3, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r,vmin=-1,vmax=2); 
plt.axis('off')
plt.savefig('./figures/exp_pamiBy.png', format='png',dpi=200)