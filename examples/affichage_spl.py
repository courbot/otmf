# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 10:53:17 2017

@author: courbot
"""
import numpy as np 
import matplotlib.pyplot as plt
import time
#err = ((im_res > 0) != (X[:,:,np.newaxis,np.newaxis,np.newaxis]>0)).mean(axis=(0,1)) 

# 
#dat = np.load('../results/spl/images_snr_cible_vs_eff_tous.npz')
#im_res = dat['images']
#X = dat['src']
##%%   
#err1 = ((im_res[:,:,:,:,0,1] > 0) != (X[:,:,np.newaxis,np.newaxis]>0)).mean(axis=(0,1)) 
#
#f=plt.figure(figsize=(5,5))  
#ax = plt.Axes(f,[0.,0.,1.,1.])
#
#f.add_axes(ax)
#ax.imshow(err.T, origin='lower',vmin=0, vmax = 0.5,cmap=plt.cm.inferno,interpolation='bicubic',extent = [-15,0,-15,0]) ; 
#ax.set_axis_off()
#plt.ylim((-15,0))
#plt.xlim((-15,0))
#plt.savefig('../results/figures/himalya_snr_cible_vs_eff_K2.png',format='png',dpi=100)
#
#
##%%   
#err2 = ((im_res[:,:,:,:,0,2] > 0) != (X[:,:,np.newaxis,np.newaxis]>0)).mean(axis=(0,1)) 
#
#f=plt.figure(figsize=(5,5))  
#ax = plt.Axes(f,[0.,0.,1.,1.])
#
#f.add_axes(ax)
#ax.imshow(err2.T, origin='lower',vmin=0, vmax = 0.5,cmap=plt.cm.inferno,interpolation='bicubic',extent = [-15,0,-15,0]) ; 
#ax.set_axis_off()
#plt.ylim((-15,0))
#plt.xlim((-15,0))
#plt.savefig('../results/figures/himalya_snr_cible_vs_eff_K3.png',format='png',dpi=100)
###plt.colorbar()
###plt.colorbar()
###
##np.savez('../results/spl/images_snr_cible_vs_eff_tous.npz',images=im_res,src = X)
###plt.ylabel('Cible')
##plt.xlabel('Effectif')
#
##%%
#err3 = ((im_res[:,:,:,:,0,3] > 0) != (X[:,:,np.newaxis,np.newaxis]>0)).mean(axis=(0,1)) 
#
#f=plt.figure(figsize=(5,5))  
#ax = plt.Axes(f,[0.,0.,1.,1.])
#
#f.add_axes(ax)
#ax.imshow(err3.T, origin='lower',vmin=0, vmax = 0.5,cmap=plt.cm.inferno,interpolation='bicubic',extent = [-15,0,-15,0]) ; 
#ax.set_axis_off()
#plt.ylim((-15,0))
#plt.xlim((-15,0))
#plt.savefig('../results/figures/himalya_snr_cible_vs_eff_K4.png',format='png',dpi=100)

#%%


#err = ((im_res > 0) != (X[:,:,np.newaxis,np.newaxis,np.newaxis]>0)).mean(axis=(0,1)) 

# 
dat = np.load('../results/spl/images_sans_cible_tous.npz')
im_res = dat['images']
#err = ((im_res > 0) != (X[:,:,np.newaxis,np.newaxis,np.newaxis]>0)).mean(axis=(0,1)) 
#X = dat['src']
##%%   
#err1 = ((im_res[:,:,:,:,0,1] > 0) != (X[:,:,np.newaxis,np.newaxis]>0)).mean(axis=(0,1)) 
#
#f=plt.figure(figsize=(5,5))  
#ax = plt.Axes(f,[0.,0.,1.,1.])
#
#f.add_axes(ax)
#ax.imshow(err.T, origin='lower',vmin=0, vmax = 0.5,cmap=plt.cm.inferno,interpolation='bicubic',extent = [-15,0,-15,0]) ; 
#ax.set_axis_off()
#plt.ylim((-15,0))
#plt.xlim((-15,0))
#plt.savefig('../results/figures/himalya_snr_cible_vs_eff_K2.png',format='png',dpi=100)
#
#
##%%   
#err2 = ((im_res[:,:,:,:,0,2] > 0) != (X[:,:,np.newaxis,np.newaxis]>0)).mean(axis=(0,1)) 
#
#f=plt.figure(figsize=(5,5))  
#ax = plt.Axes(f,[0.,0.,1.,1.])
#
#f.add_axes(ax)
#ax.imshow(err2.T, origin='lower',vmin=0, vmax = 0.5,cmap=plt.cm.inferno,interpolation='bicubic',extent = [-15,0,-15,0]) ; 
#ax.set_axis_off()
#plt.ylim((-15,0))
#plt.xlim((-15,0))
#plt.savefig('../results/figures/himalya_snr_cible_vs_eff_K3.png',format='png',dpi=100)
###plt.colorbar()
###plt.colorbar()
###
##np.savez('../results/spl/images_snr_cible_vs_eff_tous.npz',images=im_res,src = X)
###plt.ylabel('Cible')
##plt.xlabel('Effectif')
#
##%%
#err3 = ((im_res[:,:,:,:,0,3] > 0) != (X[:,:,np.newaxis,np.newaxis]>0)).mean(axis=(0,1)) 
#
#f=plt.figure(figsize=(5,5))  
#ax = plt.Axes(f,[0.,0.,1.,1.])
#
#f.add_axes(ax)
#ax.imshow(err3.T, origin='lower',vmin=0, vmax = 0.5,cmap=plt.cm.inferno,interpolation='bicubic',extent = [-15,0,-15,0]) ; 
#ax.set_axis_off()
#plt.ylim((-15,0))
#plt.xlim((-15,0))
#plt.savefig('../results/figures/himalya_snr_cible_vs_eff_K4.png',format='png',dpi=100)

