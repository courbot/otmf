# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 09:23:11 2016

:author: Jean-Baptiste Courbot - jb.courbot@unistra.fr
:date: Sep 01, 2017 (created Feb 01, 2016)
"""
import numpy as np
from numpy import cos

def phi_theta(a,b):
    """ 
    Weighting function to account for orientation in Ising models.
    
    :param float a: first parameter
    :param floas b: second parameter
    """
    
    return np.abs(cos(a-b))
    
def gen_beta(vois, angle):
    """ 
    Computation of the outputs of the weighting function given neighbor position
    and values of V (angle).
    
    :param ndarray vois: stack of neighbor number
    :param ndarray angle: priviledged directions / values of V
    
     :param ndarray beta: values generated in the lookuptable.
        """
        
#        The following numbering is used :
#        
#           --------------
#        y+1 | 6 | 5 | 4 |
#           --------------
#          y | 7 |   | 3 |
#           --------------
#        y-1 | 0 | 1 | 2 |
#           --------------
#            x-1 | x | x+1
        

    beta = np.ones_like(vois) ; #beta = beta.astype(float)
    pi = np.pi

    beta[(vois==3)+(vois==7)] = phi_theta(pi/2,angle) 
    beta[(vois==4)+(vois==0)] = phi_theta(3*pi/4.,angle)
    beta[(vois==5)+(vois==1)] = phi_theta(0,angle)
    beta[(vois==6)+(vois==2)] = phi_theta(pi/4.,angle)


    if angle==0:
        beta=np.ones_like(vois)
       
    #beta[vois==-1] = 0.   
    if np.ndim(beta) ==3:
        beta /= beta.sum(axis=2)[:,:,np.newaxis]#beta_sum
    elif np.ndim(beta) == 1:
        beta /=beta.sum()

    return beta
    
 
def psi_ising(x_1,x_2,alpha):
    """ Ising potential function
    
    :param float x_1: first argument of the potential, eventually ndarray.
    :param float x_2: second argument, eventually ndarray of the same size that x_1.
    :param float alpha: granularity parameter
    
    :returns: **res** *(ndarray)* - output of the potential, eventually ndarray.      
    """
    
    res = alpha * (1.-2.*(x_2==x_1))
        
    return res       
    
    
def init_champs(par):
    """ Set a random intialization for the class field X.
    
    :param parameter par: parameter set of the Gibbs sampling
    
    :returns: **X_init** *(ndarray)* Initialization for X.    
    """

    X_init = np.random.choice(par.x_range,size=(par.S0,par.S1))
      
    
    return X_init  

def get_num_voisins(x,y,image):
    """ Retrieving of local pixel neighborhood numbering, accounting for borders.
    
        The following numbering is used :
        

        y+1 | 6 | 5 | 4 |

          y | 7 |   | 3 |

        y-1 | 0 | 1 | 2 |

            x-1 | x | x+1
            
     **Note :** by convention, non-existing neighbor are labeled '-1'.

    :param float x: x-position of pixel in image
    :param float y: y-position of pixel in image
    :param ndarray image: concerned image, actually used for its size only.

    :returns: **voisins** *(ndarray)* - set of neighbor number    
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
    
    
    return voisins
    
def get_vals_voisins_tout(image):
    """
    Retrieving of local pixel neighborhood values, accounting for borders.

    The following numbering is used :
        

        y+1 | 6 | 5 | 4 |

          y | 7 |   | 3 |

        y-1 | 0 | 1 | 2 |

            x-1 | x | x+1
     
    Note that along borders, pixels values are duplicated.       
            
    :param ndarray image: concerned image, actually used for its size only.

    :returns: **vals** *(ndarray)* - set of neighboring values aranges in (xdim, ydim, 9) array.  
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
      