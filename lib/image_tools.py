# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 17:39:42 2016

@author: courbot
"""
import numpy as np

def get_num_voisins(x,y,image):
    """ Récupération des voisins d'un point dans une image, en tenant compte des bords
    
        The following numbering is used :
        
           --------------
        y+1 | 6 | 5 | 4 |
           --------------
          y | 7 |   | 3 |
           --------------
        y-1 | 0 | 1 | 2 |
           --------------
            x-1 | x | x+1
        Il doit être possible de stocker ca une bonne fois pour toute dans une structure !
    
    """

    S0 = image.shape[0]-1
    S1 = image.shape[1]-1
    


    if x <  S0 and x > 0 and y <  S1 and y > 0 :
        voisins = np.array([0,1,2,3,4,5,6,7])            
        
    if x == 0:
        if y==0:
            voisins = np.array([-1,-1,-1,3,4,5,-1,-1])   
        elif y==S1:
            voisins = np.array([-1,1,2,3,-1,-1,-1,-1])
        else:
            voisins = np.array([-1,1,2,3,4,5,-1,-1])
    elif x == S0:
        if y == 0:
            voisins=np.array([-1,-1,-1,-1,-1,5,6,7])
        elif y == S1:
            voisins = np.array([0,1,-1,-1,-1,-1,-1,7])
        else:
            voisins = np.array([0,1,-1,-1,-1,5,6,7])
    elif y==0:
        voisins = np.array([-1,-1,-1,3,4,5,6,7])
    elif y == S1:
        voisins = np.array([0,1,2,3,-1,-1,-1,7])
    
#    if x == 0:
#        if y==0:
#            voisins = np.array([3,4,5,-1,-1,-1,-1,-1])   
#        elif y==S1:
#            voisins = np.array([1,2,3,-1,-1,-1,-1,-1])
#        else:
#            voisins = np.array([1,2,3,4,5,-1,-1,-1])
#    elif x == S0:
#        if y == 0:
#            voisins=np.array([5,6,7,-1,-1,-1,-1,-1])
#        elif y == S1:
#            voisins = np.array([7,0,1,-1,-1,-1,-1,-1])
#        else:
#            voisins = np.array([5,6,7,0,1,-1,-1,-1])
#    elif y==0:
#        voisins = np.array([3,4,5,6,7,-1,-1,-1])
#    elif y == S1:
#        voisins = np.array([7,0,1,2,3,-1,-1,-1])
    
    # passer en 4-voisinage
#    voisins[0] = -1    
#    voisins[2] = -1
#    voisins[4] = -1
#    voisins[6] = -1
    
    return voisins
    
def get_vals_voisins_tout(image):
    """
           --------------
        y+1 | 6 | 5 | 4 |
           --------------
          y | 7 |   | 3 |
           --------------
        y-1 | 0 | 1 | 2 |
           --------------
            x-1 | x | x+1
    """
    
    S0 = image.shape[0]
    S1 = image.shape[1]
    
    vals = np.zeros(shape=(S0,S1,8)) # 8-voisinage
    
    im = np.zeros(shape=(S0+2,S1+2)) # image with 1-px 0-padding
    im[1:S0+1,1:S1+1] = image
    
    # On duplique les bords :

    im[0,:] = im[1,:] 
    im[-1,:] = im[-2,:]
    im[:,0] = im[:,1] 
    im[:,-1] = im[:,-2]
        
        
    
    
    vals[:,:,0] = im[0:S0,   0:S1]
    vals[:,:,1] = im[1:S0+1, 0:S1]
    vals[:,:,2] = im[2:S0+2, 0:S1]
    vals[:,:,3] = im[2:S0+2, 1:S1+1]
    vals[:,:,4] = im[2:S0+2, 2:S1+2]
    vals[:,:,5] = im[1:S0+1, 2:S1+2]
    vals[:,:,6] = im[0:S0,   2:S1+2]
    vals[:,:,7] = im[0:S0,   1:S1+1]
    
    
    
    return vals
#
#
#
#def get_theta_isocontour(Im):
#    dx, dy = np.gradient(Im)
#
#    theta = np.zeros_like(Im)
#    
#    theta[dx!=0] = np.arctan(dy[dx!=0]/dx[dx!=0])
#    theta[dx==0] = np.sign(dy[dx==0]) * np.pi/2
#    theta = theta + np.pi * (dx< 0)
#
#    theta_iso = theta#%np.pi
#    
#    return theta_iso