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
def plot_directions(angle, intensite,pas,taille=1,couleur='b'):
    
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
        
                plt.plot((deb_x[i,j],fin_x[i,j]), (deb_y[i,j],fin_y[i,j]) ,couleur, linewidth=0.75)
    plt.xlim((-0.5,S1-0.5))
    plt.ylim((-0.5,S0-0.5))    
 
def plot_stream(angle,dens,couleur):
    taille = 1.
    intensite = 1.
    S0 = angle.shape[0]
    S1 = angle.shape[1]

    y,x = np.ogrid[0:S0,0:S1]

    angle2 = angle

    deb_x = np.tile(x,(S0,1)) - taille*np.sin(angle2) * intensite
    deb_y = np.tile(y,(1,S1)) - taille*np.cos(angle2) * intensite

    fin_x = np.tile(x,(S0,1)) + taille*np.sin(angle2) * intensite
    fin_y = np.tile(y,(1,S1)) + taille*np.cos(angle2) * intensite



    plt.streamplot(x,y,fin_x-deb_x,fin_y-deb_y,color=couleur,arrowsize=0.0001,density=dens,linewidth=0.75)

    plt.xlim((-0.5,S1-0.5))
    plt.ylim((-0.5,S0-0.5)) 

#%%
#
#nom = 'cycl'
##if nom=='cycl':
#dat = np.load('./results/res_sp_cyclone_6cl_mpm.npz')
#dat_map = np.load('./results/res_sp_cyclone_6cl_map.npz')
#
##%%
#nom = 'vine'
#dat=np.load('./results/res_pami_vine.npz')
#dat_map = np.load('./results/res_pami_'+nom+'(map).npz')
#
##if nom=='cycl':
##dat = np.load('./results/res_sp_cyclone_6cl_mpm.npz')
##dat_map = np.load('./results/res_sp_cyclone_6cl_map.npz')
#
##res_sp_cyclone_6cl_mpm.npz
#Y, X_mpm_est, Ux_map, X_mpm_hmf, V_mpm_est, Uv_map = dat['Y'], dat['X_mpm_est'], dat['Ux_map'], dat['X_mpm_hmf'], dat['V_mpm_est'], dat['Uv_map']
#
##dat_map = np.load('./results/res_sp_cyclone_6cl_map.npz')
#
#X_map_est,V_map_est, X_map_hmf = dat_map['X_mpm_est'], dat_map['V_mpm_est'], dat['X_mpm_hmf']


nom='expA'
dat = np.load('./results/manuscrit/expa_snr-6_unknown2.npz')

#nom='expB'
#dat = np.load('./results/manuscrit/expb_snr-6_unknown13.npz')

Y = dat.f.Y
X_mpm_est = dat.f.X_mpm_est
X_mpm_hmf = dat.f.X_mpm_hmf
V_mpm_est = dat.f.V_mpm_est
Ux_map = dat.f.Ux_map
Ux_hmf = dat.f.Ux_hmf
Uv_map = dat.f.Uv_map
#                        parsem = dat.f.parsem,

X_hmf = dat.f.X_mpm_hmf


V2 = np.pi/4 * np.ones(shape=(128,128))
V2[64:,:] = 3*np.pi/4
V2[:,64:] = (V2[:,64:] + np.pi/2)%np.pi



#cm_gris = plt.cm.Greys
#
##%%
plt.close('all')
fig = plt.figure(figsize=(6,6))
ax = plt.Axes(fig, [0.,0.,1.,1.])
fig.add_axes(ax)
im = Y.mean(axis=2)
ax.imshow(im, interpolation='nearest', origin='lower', cmap=plt.cm.gray,vmin = im.mean() - 3*im.std(), vmax = im.mean() + 3*im.std())#,vmin = -1,vmax=1); 
plt.axis('off')

plt.axis('off')
plt.savefig('./figures/'+nom+'_y.png', format='png',dpi=128/6.)

#
##%%
fig = plt.figure(figsize=(6,6))
ax = plt.Axes(fig, [0.,0.,1.,1.])
fig.add_axes(ax)
ax.imshow(X_mpm_est, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r); 
plt.axis('off')
plt.savefig('./figures/'+nom+'_x_otmf_mpm.png', format='png',dpi=128/6.)

##%%
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(1-X_map_est, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r); 
#plt.axis('off')
#plt.savefig('./figures/'+nom+'_x_otmf_map.png', format='png',dpi=200)
#
##%%
#
fig = plt.figure(figsize=(6,6))
ax = plt.Axes(fig, [0.,0.,1.,1.])
fig.add_axes(ax)
ax.imshow(Ux_map, interpolation='nearest', origin='lower', cmap=plt.cm.gray,vmin=0,vmax=1); 
plt.axis('off')
plt.savefig('./figures/'+nom+'_ux_otmf_mpm.png', format='png',dpi=128/6.)


fig = plt.figure(figsize=(6,6))
ax = plt.Axes(fig, [0.,0.,1.,1.])
fig.add_axes(ax)
ax.imshow(Uv_map, interpolation='nearest', origin='lower', cmap=plt.cm.gray,vmin=0,vmax=1); 
plt.axis('off')
plt.savefig('./figures/'+nom+'_uv_otmf_mpm.png', format='png',dpi=128/6.)
##%%
fig = plt.figure(figsize=(6,6))
ax = plt.Axes(fig, [0.,0.,1.,1.])
fig.add_axes(ax)

ax.imshow(X_mpm_hmf, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r); 
plt.axis('off')
plt.savefig('./figures/'+nom+'_x_hmf_mpm.png', format='png',dpi=128/6.)


fig = plt.figure(figsize=(6,6))
ax = plt.Axes(fig, [0.,0.,1.,1.])
fig.add_axes(ax)
ax.imshow(Ux_hmf, interpolation='nearest', origin='lower', cmap=plt.cm.gray,vmin=0,vmax=1); 
plt.axis('off')
plt.savefig('./figures/'+nom+'_ux_hmf_mpm.png', format='png',dpi=128/6.)

##%%
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#
#ax.imshow(X_map_hmf, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r); 
#plt.axis('off')
#plt.savefig('./figures/'+nom+'_x_hmf_map.png', format='png',dpi=200)
#
##%%
#import cmocean
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
##if nom == 'mars':
##    ax.imshow(Y.mean(axis=2), interpolation='nearest', origin='lower', cmap=plt.cm.copper, alpha=0.25)#,vmin = -1,vmax=1); 
##else:
##    ax.imshow(Y.mean(axis=2), interpolation='nearest', origin='lower', cmap=plt.cm.Greens_r,vmax=0.75,alpha=0.25)
### 
##ax.imshow(Xbis, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r,alpha=0.25); 
##ax.imshow(V_mpm_est, interpolation='nearest', origin='lower', cmap=plt.cm.hsv,alpha=0.5)
##ax.imshow(V_mpm_est, vmin=0, vmax=np.pi,interpolation='nearest', origin='lower',cmap=cmocean.cm.phase,alpha=0.25)
##ax.imshow(V_mpm_est, vmin=0, vmax=np.pi,interpolation='nearest', origin='lower',cmap=plt.cm.inferno,alpha=0.5)
#ax.imshow(Y.mean(axis=2),interpolation='nearest', origin='lower',cmap=plt.cm.gray,alpha=0.75)
#
#V_fi = fi.median_filter(V_mpm_est, size=4)
#plot_directions(V_fi, np.ones_like(V_mpm_est),pas=8,taille=2)
#plt.axis('off')
#plt.savefig('./figures/'+nom+'_v_otmf_mpm.png', format='png',dpi=200)
##%%
#
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(Y.mean(axis=2),interpolation='nearest', origin='lower',cmap=plt.cm.gray,alpha=0.75)
#
#%%
fig = plt.figure(figsize=(2.5,2.5))
ax = plt.Axes(fig, [0.,0.,1.,1.])
fig.add_axes(ax)
ax.imshow(V_mpm_est, vmin=0, vmax=np.pi,interpolation='nearest', origin='lower',cmap=plt.cm.inferno,alpha=0.5)
plot_directions(V_mpm_est, np.ones_like(V_mpm_est),pas=8,taille=2,couleur='k')#firebrick')
#ax.imshow(Y.mean(axis=2),interpolation='nearest', origin='lower',cmap=plt.cm.gray,alpha=0.5)
#
#plot_stream(V_mpm_est,dens=0.55*128./30,couleur='firebrick') 
plt.axis('off')
#plt.savefig('./figures/'+nom+'2e1.png', format='png',dpi=200)
plt.savefig('./figures/'+nom+'_v_otmf_mpm2.png', format='png',dpi=100)

#%%
fig = plt.figure(figsize=(2.5,2.5))
ax = plt.Axes(fig, [0.,0.,1.,1.])
fig.add_axes(ax)
ax.imshow(V2, vmin=0, vmax=np.pi,interpolation='nearest', origin='lower',cmap=plt.cm.inferno,alpha=0.5)
plot_directions(V2, np.ones_like(V_mpm_est),pas=8,taille=2,couleur='k')
#ax.imshow(Y.mean(axis=2),interpolation='nearest', origin='lower',cmap=plt.cm.gray,alpha=0.5)
#
#plot_stream(V_mpm_est,dens=0.55*128./30,couleur='firebrick') 
plt.axis('off')
#plt.savefig('./figures/'+nom+'2e1.png', format='png',dpi=200)
plt.savefig('./figures/'+nom+'_v_orig2.png', format='png',dpi=100)

#%%
##%%
#fig = plt.figure(figsize=(2.5,2.5))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(Y.mean(axis=2),interpolation='nearest', origin='lower',cmap=plt.cm.gray,alpha=0.5)
##
#plot_stream(V_mpm_est,dens=0.7*128./30,couleur='firebrick') 
#plt.axis('off')
##plt.savefig('./figures/'+nom+'2e1.png', format='png',dpi=200)
#plt.savefig('./figures/'+nom+'_v_otmf_mpm2.png', format='png',dpi=200)
#
##%%
#fig = plt.figure(figsize=(2.5,2.5))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(V_map_est, vmin=0, vmax=np.pi,interpolation='nearest', origin='lower',cmap=plt.cm.inferno,alpha=0.5)
#plot_directions(V_mpm_est, np.ones_like(V_mpm_est),pas=4,taille=1,couleur='firebrick')
##ax.imshow(Y.mean(axis=2),interpolation='nearest', origin='lower',cmap=plt.cm.gray,alpha=0.5)
##
##plot_stream(V_mpm_est,dens=0.55*128./30,couleur='firebrick') 
#plt.axis('off')
##plt.savefig('./figures/'+nom+'2e1.png', format='png',dpi=200)
#plt.savefig('./figures/'+nom+'_v_otmf_map.png', format='png',dpi=200)
#
##%%
#fig = plt.figure(figsize=(2.5,2.5))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(Y.mean(axis=2),interpolation='nearest', origin='lower',cmap=plt.cm.gray,alpha=0.5)
##
#plot_stream(V_map_est,dens=0.7*128./30,couleur='firebrick') 
#plt.axis('off')
##plt.savefig('./figures/'+nom+'2e1.png', format='png',dpi=200)
#plt.savefig('./figures/'+nom+'_v_otmf_map2.png', format='png',dpi=200)
#
#


   
#%% 
#nom = 'vine'
#dat = np.load('./results/res_pami_'+nom+'.npz')
#
#Y, X_mpm_est, Ux_map, X_mpm_hmf, V_mpm_est, Uv_map = dat['Y'], dat['X_mpm_est'], dat['Ux_map'], dat['X_mpm_hmf'], dat['V_mpm_est'], dat['Uv_map']
#
#
#cm_gris = plt.cm.Greys
#
##%%
#plt.close('all')
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#im = Y.mean(axis=2)
##ax.imshow(im, interpolation='nearest', origin='lower', cmap=plt.cm.gray,vmin = im.mean() - 3*im.std(), vmax = im.mean() + 3*im.std())#,vmin = -1,vmax=1); 
#ax.imshow(im, interpolation='nearest', origin='lower', cmap=plt.cm.gray, vmax = 0.8)#im.mean() + 3*im.std())#,vmin = -1,vmax=1); 
#
#plt.axis('off')
##if nom == 'mars':
##    ax.imshow(Y.mean(axis=2), interpolation='nearest', origin='lower', cmap=plt.cm.copper)#,vmin = -1,vmax=1); 
##else:
##    ax.imshow(Y.mean(axis=2), interpolation='nearest', origin='lower', cmap=plt.cm.Greens_r,vmax=0.75)
#   
#plt.axis('off')
#plt.savefig('./figures/'+nom+'2a.png', format='png',dpi=200)
#
#
##%%
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
##if nom == 'mars':
##    Xbis = np.copy(X_mpm_est)
##    Xbis[X_mpm_est==0]=0.5
##    Xbis[X_mpm_est==0.5]=1.0
##    Xbis[X_mpm_est==1.0]=0.
##    ax.imshow(Xbis, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r); 
##else:
#ax.imshow(1-X_mpm_est, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r); 
#plt.axis('off')
#plt.savefig('./figures/'+nom+'2b.png', format='png',dpi=200)
#
##%%
#
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(Ux_map, interpolation='nearest', origin='lower', cmap=plt.cm.gray,vmin=0,vmax=1); 
#plt.axis('off')
#plt.savefig('./figures/'+nom+'2c.png', format='png',dpi=200)
##%%
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
##if nom == 'mars':
##    Xhbis = np.copy(X_mpm_hmf)
###    Xhbis[X_mpm_hmf==0]=0.5
##    Xhbis[X_mpm_hmf==0.5]=1.
##    Xhbis[X_mpm_hmf==1.0]=0.5
##    ax.imshow(Xhbis, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r); 
##else:
#ax.imshow(1-X_mpm_hmf, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r); 
#plt.axis('off')
#plt.savefig('./figures/'+nom+'2d.png', format='png',dpi=200)
#
##%%
#import cmocean
#fig = plt.figure(figsize=(2,2))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(Y.mean(axis=2),interpolation='nearest', origin='lower',cmap=plt.cm.gray,alpha=0.5,vmax=0.8)
##
#plot_stream(V_mpm_est,dens=0.5*128./30,couleur='firebrick') 
#plt.axis('off')
#plt.savefig('./figures/'+nom+'2e1.png', format='png',dpi=200)
#
##%%
#
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(V_mpm_est, vmin=0, vmax=np.pi,interpolation='nearest', origin='lower',cmap=plt.cm.inferno,alpha=0.5)
##V_fi = fi.median_filter(V_mpm_est, size=4)
#plot_directions(V_mpm_est, np.ones_like(V_mpm_est),pas=4,taille=1)
#plt.axis('off')
#plt.savefig('./figures/'+nom+'2e2.png', format='png',dpi=200)
#
##%%
###%%
##fig = plt.figure(figsize=(6,6))
##ax = plt.Axes(fig, [0.,0.,1.,1.])
##fig.add_axes(ax)
###if nom == 'mars':
###    ax.imshow(Y.mean(axis=2), interpolation='nearest', origin='lower', cmap=plt.cm.copper, alpha=0.25)#,vmin = -1,vmax=1); 
###else:
###    ax.imshow(Y.mean(axis=2), interpolation='nearest', origin='lower', cmap=plt.cm.Greens_r,vmax=0.75,alpha=0.25)
#### 
###ax.imshow(X_mpm_hmf, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r,alpha=0.); 
##ax.imshow(V_mpm_est, interpolation='nearest', origin='lower', cmap=plt.cm.Spectral,alpha=0.5)
##V_fi = fi.median_filter(V_mpm_est, size=6)
##plot_directions(V_fi, np.ones_like(V_mpm_est),pas=6,taille=2)
##%%
#
#
#
##
##
##fig = plt.figure(figsize=(6,6))
##ax = plt.Axes(fig, [0.,0.,1.,1.])
##fig.add_axes(ax)
##ax.imshow(V_mpm_est, interpolation='nearest', origin='lower', cmap=plt.cm.Spectral,alpha=0.5)
###plot_directions(V_mpm_est, np.ones_like(V_mpm_est),pas=6,taille=2)
#
##%%
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(Uv_map, interpolation='nearest', origin='lower', cmap=plt.cm.gray,vmin=0,vmax=1); 
#plt.axis('off')
#plt.savefig('./figures/'+nom+'2f.png', format='png',dpi=200)
#
#
#
#
#
#
#
#
##==============================================================================
##  résultats MAP vignes
##==============================================================================
##
#
#nom = 'vine'
#dat = np.load('./results/res_pami_'+nom+'(map).npz')
#
#Y, X_mpm_est, Ux_map, X_mpm_hmf, V_mpm_est, Uv_map = dat['Y'], dat['X_mpm_est'], dat['Ux_map'], dat['X_mpm_hmf'], dat['V_mpm_est'], dat['Uv_map']
#
#
#cm_gris = plt.cm.Greys
#
##%% OTMF
##plt.close('all')
#
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#
#ax.imshow(1-X_mpm_est, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r); 
#plt.axis('off')
#plt.savefig('./figures/'+nom+'2b(map).png', format='png',dpi=200)
#
##%%
#
##%% HMF
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#
#ax.imshow(X_mpm_hmf, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r); 
#plt.axis('off')
#plt.savefig('./figures/'+nom+'2d(map).png', format='png',dpi=200)
#
##%%
##
##fig = plt.figure(figsize=(6,6))
##ax = plt.Axes(fig, [0.,0.,1.,1.])
##fig.add_axes(ax)
##
###ax.imshow(V_mpm_est, vmin=0, vmax=np.pi,interpolation='nearest', origin='lower',cmap=plt.cm.inferno,alpha=0.5)
##ax.imshow(Y.mean(axis=2),interpolation='nearest', origin='lower',cmap=plt.cm.gray,alpha=0.75,vmax=0.8)
##V_fi = fi.median_filter(V_mpm_est, size=4)
##plot_directions(V_fi, np.ones_like(V_mpm_est),pas=4,taille=1)
##plt.axis('off')
##plt.savefig('./figures/'+nom+'2e(map).png', format='png',dpi=200)
#
#
##%%
#
#from matplotlib2tikz import save as tikz_save
#
#fig = plt.figure(figsize=(2,2))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(Y.mean(axis=2),interpolation='nearest', origin='lower',cmap=plt.cm.gray,alpha=0.5,vmax=0.8)
##
#plot_stream(V_mpm_est,dens=0.5*128./30,couleur='firebrick') 
#plt.axis('off')
#plt.savefig('./figures/'+nom+'2e1(map).png', format='png',dpi=200)
#
##tikz_save('./figures/'+nom+'2e1(map).tex')
#
##%%
#
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(V_mpm_est, vmin=0, vmax=np.pi,interpolation='nearest', origin='lower',cmap=plt.cm.inferno,alpha=0.5)
##V_fi = fi.median_filter(V_mpm_est, size=4)
#plot_directions(V_mpm_est, np.ones_like(V_mpm_est),pas=4,taille=1)
#plt.axis('off')
#plt.savefig('./figures/'+nom+'2e2(map).png', format='png',dpi=200)

#%%

#==============================================================================
#  résultats MAP mars
#==============================================================================
#
#
#nom = 'mars'
#dat = np.load('./results/res_pami_'+nom+'(map).npz')
#
#Y, X_mpm_est, Ux_map, X_mpm_hmf, V_mpm_est, Uv_map = dat['Y'], dat['X_mpm_est'], dat['Ux_map'], dat['X_mpm_hmf'], dat['V_mpm_est'], dat['Uv_map']
#
#
#cm_gris = plt.cm.Greys
#
##%% OTMF
#plt.close('all')
#
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#
#ax.imshow(1-X_mpm_est, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r); 
#plt.axis('off')
#plt.savefig('./figures/'+nom+'2b(map).png', format='png',dpi=200)
#
##%%
#
##%% HMF
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#
#ax.imshow(1-X_mpm_hmf, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r); 
#plt.axis('off')
#plt.savefig('./figures/'+nom+'2d(map).png', format='png',dpi=200)
#
##%%
#
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#
#ax.imshow(V_mpm_est, vmin=0, vmax=np.pi,interpolation='nearest', origin='lower',cmap=plt.cm.inferno,alpha=0.5)
#
#V_fi = fi.median_filter(V_mpm_est, size=8)
#plot_directions(V_fi, np.ones_like(V_mpm_est),pas=8,taille=2)
#plt.axis('off')
#plt.savefig('./figures/'+nom+'2e(map).png', format='png',dpi=200)

