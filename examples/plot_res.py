# -*- coding: utf-8 -*-
"""
Created on Fri May 27 14:54:53 2016

@author: courbot
"""


import numpy as np 
import sys 
import matplotlib.pyplot as plt
import time
import scipy.stats as st
import scipy.cluster.vq as cvq
import multiprocessing as mp
import image_tools as it
from scipy.ndimage import imread
import matplotlib.mlab as mlab

import parameters
#import image_tools as it
import gibbs_sampler as gs
import fields_tools as ft
import seg_OTMF as sot
import SEM as sem

from scipy.ndimage import zoom
from scipy.ndimage.filters import gaussian_filter 
from scipy.ndimage.filters import median_filter 

import matplotlib


def plot_directions(angle, intensite,pas):
    
    S0 = angle.shape[0]
    S1 = angle.shape[1]
    
    y,x = np.ogrid[0:S0,0:S1]
    
    angle2 = angle

    deb_x = np.tile(x,(S0,1)) - 0.5*np.sin(angle2) * intensite
    deb_y = np.tile(y,(1,S1)) - 0.5*np.cos(angle2) * intensite
    
    fin_x = np.tile(x,(S0,1)) + 0.5*np.sin(angle2) * intensite
    fin_y = np.tile(y,(1,S1)) + 0.5*np.cos(angle2) * intensite
    
    
    for i in range(int(pas/2.),S0,pas):
        for j in range(int(pas/2.),S1,pas):
#            if angle[i,j] != 0:
                plt.plot((deb_x[i,j],fin_x[i,j]), (deb_y[i,j],fin_y[i,j]) ,'k')
    plt.xlim((-0.5,S1-0.5))
    plt.ylim((-0.5,S0-0.5))     

def erreur(A,B):
    return (A[~np.isnan(B)] != B[~np.isnan(B)] ).mean()   
    
v_range = np.array([np.pi/4,np.pi/2, 3*np.pi/4,np.pi])

dat_hmf = np.load('./results/exray_seg_hmf.npz')
X_hmf = dat_hmf['X_mpm_hierarch']


dat_tmf = np.load('./results/exray_seg_uncert4.npz')
Y = dat_tmf['Y']
X_tmf = dat_tmf['X_mpm_hierarch']
V_tmf = dat_tmf['V_mpm_hierarch']

X = dat_tmf['X']
V = dat_tmf['V']



#plt.close('all')
nb_li = 2
nb_col =3

plt.rc('text', usetex=True)
plt.rc('font',family='serif')

cm_gris = matplotlib.cm.bone
cm_angl = matplotlib.cm.Spectral

cm_gris.set_bad('r',1.)
cm_angl.set_bad('r',1.)

plt.figure(figsize=(3.5*nb_col,3*nb_li))


plt.subplot(nb_li,nb_col,1)
plt.imshow(Y.mean(axis=2), interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin = -1,vmax=1); 
plt.axis('off')
plt.title('$\mathbf{y}$')

plt.subplot(nb_li,nb_col,2)
plt.imshow(X, interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin=0,vmax=1); 
plt.axis('off')
plt.title('$\mathbf{x}$')

plt.subplot(nb_li,nb_col,3)
plt.imshow(V, interpolation='nearest', origin='lower', cmap=plt.cm.Spectral,vmin=0,vmax=np.pi); 
plot_directions(V, np.ones_like(V),pas=8)
plt.axis('off')
plt.title('$\mathbf{v}$')




    
plt.subplot(nb_li,nb_col,nb_col+2)
plt.imshow(X_tmf[:,:,0], interpolation='nearest', origin='lower', cmap=cm_gris,vmin=0,vmax=1); 
plt.axis('off')
plt.title('$\\hat{\mathbf{x}}_{\mathrm{MPM}}$ - erreur de %.2f '%(erreur(X,X_tmf[:,:,0])*100))


plt.subplot(nb_li,nb_col,nb_col+3)
plt.imshow(V_tmf[:,:,0], interpolation='nearest', origin='lower',cmap=cm_angl,vmin=0,vmax=np.pi); 
plot_directions(V_tmf[:,:,0], np.ones_like(V_tmf[:,:,0]),pas=8)
plt.axis('off')
plt.title('$\\hat{\mathbf{v}}_{\mathrm{MPM}}$ - erreur de %.2f '%(erreur(V,V_tmf[:,:,0])*100))

#plt.subplot(nb_li,nb_col,3)
#plt.imshow(X_hmf[:,:,0], interpolation='nearest', origin='lower', cmap=cm_gris,vmin=0,vmax=1); 
#plt.axis('off')
#plt.title('Champ Cache : $\\hat{\mathbf{x}}_{\mathrm{MPM}}$ - erreur de %.2f '%(erreur(X,X_hmf[:,:,0])*100))





plt.tight_layout()

plt.savefig('./figures/exray_seg_tmf.pdf', format='pdf',dpi=100)
