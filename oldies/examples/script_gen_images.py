# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 09:00:12 2016

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



S0 = 128
S1 = 128

y,x = np.ogrid[0:S0,0:S1]

v_range = np.array([np.pi/3, 2*np.pi/3])

X = np.zeros(shape=(S0,S1))

reg_ne = (x >= S0/2) * (y>= S1/2) 
reg_nw = (x <= S0/2) * (y>= S1/2) 
reg_se = (x >= S0/2) * (y<= S1/2) 
reg_sw = (x <= S0/2) * (y<= S1/2) 


freq0 = 1/30. #frequence spatiale
freq1 = 1/15.
#X = (np.cos(2*np.pi*freq*(y - x*np.cos(v_range[0])))>0)*reg_sw 
X +=  (np.cos(2*np.pi*freq0*(y - x*np.cos(v_range[0])))>0)*(reg_sw)
X +=  (np.cos(2*np.pi*freq1*(y - x*np.cos(v_range[1])))>0)*(reg_nw)

X +=  (np.cos(2*np.pi*freq0*(y - x*np.cos(v_range[0])))>0.5)*(reg_ne)
X +=  (np.cos(2*np.pi*freq1*(y - x*np.cos(v_range[1])))>0.5)*(reg_se)

V = np.zeros(shape=(S0,S1))

V[reg_sw+reg_ne] = v_range[0]
V[reg_nw+reg_se] = v_range[0]
#X[reg_nw] =  X[reg_ne].T
plt.close('all')
plt.figure()
plt.imshow(X, interpolation='nearest', origin='lower',cmap=plt.cm.gray,vmin=0,vmax=1); 
plt.axis('off')