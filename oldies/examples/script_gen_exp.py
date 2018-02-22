# -*- coding: utf-8 -*-
"""
Created on Tue May 17 10:10:40 2016

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

S0 = 128
S1 = 128

v_range = np.array([np.pi/4.,3*np.pi/4.])
    
x,y = np.ogrid[0:S0,0:S1]
##########################################"
# Cas 1
#
centre1 = np.array([S0/2-15,S1/2-5])
theta1 = np.arctan2(y-centre1[1],x-centre1[0])+np.pi/6
ray1 = np.sqrt((x-centre1[0])**2 + (y-centre1[1])**2)

centre2 = np.array([S0/2-15,S1/2-5])
theta2 = np.arctan2(y-centre2[1],x-centre2[0])+np.pi/4

#xx = x.astype(float)/1000. - 0.5
#yy = y.astype(float)/1000.-0.5
#pol = np.zeros_like(y)+np.exp(yy-(6*xx**5  +  xx))


#X = pol

#theta = (theta%np.pi)


X1 = (np.sin(20*theta1)>0.)# * (ray1>10)
X2 = (np.cos(18*theta2)>0.)#* (ray1>10)
X3 = (500-y)<x

#X = X1- 0.5*X2
#X *= (X>=0)

#X = np.cos(0.01*(ray1+40)**(1.85))>=0

X = np.cos(0.005*(np.abs(y-x)+50)**(2.0))>0.
#X = np.cos((y-x)*0.5)>0
Xa = X[:64,:64]
X[:64,64:] = Xa[::-1,:]
X[64:,:64] = Xa[::-1,:]

V = np.zeros_like(X)
V = (np.arctan2(x-centre1[1],y-centre1[0])%np.pi)#+np.pi/2
#np.savez('./data/jap.npz', X=X, V=V) 


np.savez('./data/exp_pamiC',X = X, V = V)


##########################################"
# Cas 2
#
#
#
#xp = (x-200)/80.
#yp = (y-300)/120.
#gauss1 = 10*np.exp(-(xp**2 - xp*yp + yp**2))
##gauss1 = gauss1/np.float(*gauss1.max())
#
#
#xp = (x-250)/80.
#yp = (y-750)/120.
#gauss2 = 12*np.exp(-(xp**2 + xp*yp + yp**2))
#
#
#xp = (x-750)/80.
#yp = (y-750)/120.
#gauss3 = 14*np.exp(-(xp**2 - xp*yp + yp**2))
#
#
#xp = (x-750)/60.
#yp = (y-250)/120.
#gauss4 = 16*np.exp(-(xp**2 + xp*yp + yp**2))
#
#
#
#
##centre2 = np.array([S0/2-250,S1/2+250])
##gauss2 = 12*np.exp(-(((x-centre2[0])**2)/1500. +(x-centre2[0])*(y-centre2[1])/500.+ ((y-centre2[1])**2)/500.))
#
##centre2 = np.array([S0/2+250,S1/2+250])
##gauss3 = 14*np.exp(-(((x-centre2[0])**2)/1000. -(x-centre2[0])*(y-centre2[1])/500.+ ((y-centre2[1])**2)/500.))
#
##centre2 = np.array([S0/2+250,S1/2-250])
##gauss4 = 16*np.exp(-(((x-centre2[0])**2)/1500. +0.5*(x-centre2[0])*(y-centre2[1])/500.+ ((y-centre2[1])**2)/500.))
##gauss2 = 1.-gauss2/np.float(gauss2.max())
#
#X = gauss1+gauss2+gauss3+gauss4



##########################################"
# Cas 3
#
#
#
#xp1 = (x)/250.
#yp1 = (y)/200.
#gauss1 = np.exp(-(xp1**2 -1.5*xp1*yp1 + yp1**2))
#xp1 = (x-250)/50.
#yp1 = (y-250)/50.
#gauss1b = np.exp(-(xp1**2 -1.5*xp1*yp1 + yp1**2))
#
##gauss1 = gauss1/np.float(*gauss1.max())
#
#
#xp2 = (x-750)/200.
#yp2 = (y-750)/200.
#gauss2 = np.exp(-(xp2**2 + 1.5*xp2*yp2 + yp2**2))
#
#xp2 = (x-750)/70.
#yp2 = (y-750)/70.
#gauss2b= np.exp(-(xp2**2 +1.5*xp2*yp2 + yp2**2))

#gauss3 = np.exp(-(xp1+yp2)**2)
#
#centre2 = np.array([S0/2-250,S1/2+250])
#gauss2 = 12*np.exp(-(((x-centre2[0])**2)/1500. +(x-centre2[0])*(y-centre2[1])/500.+ ((y-centre2[1])**2)/500.))
#
#centre2 = np.array([S0/2+250,S1/2+250])
#gauss3 = 14*np.exp(-(((x-centre2[0])**2)/1000. -(x-centre2[0])*(y-centre2[1])/500.+ ((y-centre2[1])**2)/500.))
#
#centre2 = np.array([S0/2+250,S1/2-250])
#gauss4 = 16*np.exp(-(((x-centre2[0])**2)/1500. +0.5*(x-centre2[0])*(y-centre2[1])/500.+ ((y-centre2[1])**2)/500.))
#gauss2 = 1.-gauss2/np.float(gauss2.max())
#
#X = gauss1 +gauss2
#V = np.zeros_like(X)
#V = v_range[0] * np.ones(shape=(S0,S1))
#V[1000-x<y] = v_range[1]
#%%
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(X, interpolation="nearest", origin="lower",cmap=plt.cm.gray)
#plt.colorbar()

#V[S0/2:,S1/2:] = v_range[0]
plt.subplot(122)
plt.imshow(V, interpolation='nearest', origin='lower',cmap=plt.cm.Spectral,vmin=0,vmax=np.pi); 
plot_directions(V, np.ones_like(V),pas=10)
plt.axis('off')


#np.savez('./data/atten1.npz', X=X, V=V) 
#plt.subplot(122)
#plt.imshow(V, interpolation="nearest", origin="lower",cmap=plt.cm.Spectral,vmin=0,vmax=np.pi); 
#plot_directions(V, np.ones_like(V),pas=50)

#np.savez('./data/atten1.npz', X=X, V=V) 
#%%


