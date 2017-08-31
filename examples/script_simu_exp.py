# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 16:22:59 2016

@author: courbot
"""



import numpy as np 
#import sys 
import matplotlib.pyplot as plt
#import time
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
#import fields_tools as ft
#import seg_OTMF as sot
import SEM as sem
import scipy.stats as st
#
#from scipy.ndimage import zoom
from scipy.ndimage.filters import gaussian_filter 
#from scipy.ndimage.filters import median_filter 

import gdal

#import spectral.io.envi as envi

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
                plt.plot((deb_x[i,j],fin_x[i,j]), (deb_y[i,j],fin_y[i,j]) ,'k')
    plt.xlim((-0.5,S1-0.5))
    plt.ylim((-0.5,S0-0.5))     
    
    




#pas = np.pi/2
#v_range = np.arange(pas, np.pi + pas, pas)
#v_range = np.array([np.pi/2,np.pi])
v_range = np.array([0,np.pi/4,3*np.pi/4,])
###################
nb_level_x =2
x_range = np.arange(0, 1+1./nb_level_x, 1./nb_level_x)

S0 = 256
S1 = 256
W = 1



#==============================================================================
# On définit V ici !
#==============================================================================

#
V = np.pi/4 * np.ones(shape=(S0,S1))
V[S0/2:,:] = 3*np.pi/4
V[:,S1/2:] = (V[:,S1/2:] + np.pi/2)%np.pi

print('---------------------------------------')
pargibbs = parameters.ParamsGibbs(S0 = S0,
                             S1 = S1,
                             type_pot = 'potts',
                             phi_uni = 0.,
                             thr_conv=0.001,
                             nb_iter=100,
                             fuzzy=False,
                             anisotropic=True,   
                             angle=np.zeros(shape=(S0,S1)),
                             beta = 1.,
                             phi_theta_0 = 0.,
                             alpha =1,
                             alpha_v = 1,
                             delta = 0.,
                             init_method = 'std',
                             nb_fuzzy = 256. ,
                             v_range = v_range,
                             x_range = x_range
                             )# beta=1.25,


# des valeurs estimees sur de vraies images
#[[  2.e-04,   1.0e-03,   3.3e-03,
#          1e-02,   7e-02,   1.4e-01,
#          1.1e-01,   1.4e-01,   5.0e-01],

#pi_x = np.exp(np.arange(9.))#*0.8)
#pi_x = np.ones_like(pi_x)
#pi_x/=pi_x.sum()
#pi_x[0] = 0
#pi_x[1] = 0
##pi_x[2] = 0.2
#pi_x[3] = 0.2
#pi_x[4] = 0.4
#pi_x[5] = 0.7
#pi_x[6] = 0.9
#pi_x[7]=  0.4
#pi_x[8] = 0.2
pargibbs.autoconv=False
pi_x = np.ones(shape=9)*0.1
pi_x[8] = 0.1
pi_x[7] = 0.1
pi_x[6] = 0.28
pi_x[5] = 0.35
pi_x[4] = 0.4
pi_x[3] = 0.1
pi_x[2] = 0.06
pi_x[1] = 0.00
pi_x[0] = 0.


pi_v = np.exp(20*np.arange(9.))#*0.8)

#pargibbs.pi = np.array([pi_x,
#       [  0,0,0,0,0.1,0.1,0.1,0.1,1]])

pargibbs.pi = np.array([pi_x, pi_v])
#pargibbs.pi =  sem.est_pi(X,V,pargibbs)

#dat = np.load('./data/exp_pami1.npz')
##X1=dat['X']
#V=dat['V']
#V[:20,:20] = np.pi/4.
pargibbs.V = V

use_y = False
normal = False
use_pi = True


pargibbs.nb_iter = 150
generate_v = False
generate_x = True
parv = gs.gen_champs_fast(pargibbs,generate_v,generate_x,use_y,normal,use_pi)   
#%%
print str(parv.X_res.shape[2]) + ' iterations'

#V,dumb = st.mode(parv.V_res[:,:,-50:-1],axis=2)
#V = V[:,:,0]
#%%
#X_new = gaussian_filter(X.astype(float), sigma=(1,1))
#X_final = np.zeros_like(X_new)
#
#for id_x in range(x_range.size):
#    x_msk = gaussian_filter((X==x_range[id_x]).astype(float), sigma=(2,2))
#    X_final[x_msk] = x_range[id_x]
#
#X_final[X_new <=0.25] = 0
#X_final[X_new>0.75] = 1
#
#plt.imshow(X_final, interpolation='nearest', origin='lower', cmap=plt.cm.gray,vmin=0,vmax=1); 

#pargibbs.nb_iter = 300
#pargibbs.V = V
#generate_v = False
#generate_x = True
#parvx = gs.gen_champs_fast(pargibbs,generate_v,generate_x,use_y,normal,use_pi)   
#print str(parvx.X_res.shape[2]) + ' iterations'
#%%


#X = parvx.X_res[:,:,-50:-1].mean(axis=2)>0.5

#V = np.pi/4 + np.pi/2 * (parvx.V_res[:,:,-50:-1].mean(axis=2)>np.pi/2)

#%%
#np.savez('./data/exp_pami2',X =X, X_final = X_final, V = V, pi = pargibbs.pi)
#

#%%
#==============================================================================
# Affichage 
#==============================================================================
#%%
X,dumb = st.mode(parv.X_res[:,:,-50:-1],axis=2)
X = X[:,:,0]



#%%
#X_new = gaussian_filter(X.astype(float), sigma=(1,1))
X_final = np.zeros_like(X)

for id_x in range(x_range.size):
    x_msk = gaussian_filter((X==x_range[id_x]).astype(float), sigma=(1,1))
    X_final[x_msk>0.5] = x_range[id_x]
#
#X_final[X_new <=0.25] = 0
#X_final[X_new>0.75] = 1

#plt.imshow(X_final, interpolation='nearest', origin='lower', cmap=plt.cm.gray,vmin=0,vmax=1); 


plt.figure(figsize=(9,9))
nb_li = 2
nb_col = 2
plt.subplot(nb_li,nb_col,1)
plt.imshow(X, interpolation='nearest', origin='lower', cmap=plt.cm.gray,vmin=0,vmax=1); 
#plt.title('$\\hat{\mathbf{x}}_{\mathrm{MPM}}$ ')

plt.subplot(nb_li,nb_col,3)
plt.imshow(X_final, interpolation='nearest', origin='lower', cmap=plt.cm.gray,vmin=0,vmax=1); 
plt.title('$\\hat{\mathbf{x}}_{\mathrm{MPM}}$ ')


plt.subplot(nb_li,nb_col,2)
plt.imshow(V, interpolation='nearest', origin='lower', cmap=plt.cm.Spectral, vmin=0, vmax=np.pi);#plt.imshow(V_mpm_est, interpolation='nearest', origin='lower', cmap=cm_angl,vmin=0,vmax=np.pi); 
plot_directions(V, np.ones_like(V),pas=8,taille=0.75)
plt.tight_layout()