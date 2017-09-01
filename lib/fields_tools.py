# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 09:23:11 2016

@author: courbot
"""
import numpy as np

#from numpy import abs
from numpy import cos
#from numpy import log
#from image_tools import get_num_voisins, get_vals_voisins


def phi_theta(a_1,a_2):
    
    return np.abs(cos(a_1-a_2))
    
def gen_beta(vois, angle,phi_theta_0, beta_sum=1):
    """ vois : stack of neighbor number
        angle : priviledged directions (image)
        gen_beta(Vois,v_range[id_v],phi_theta_0)
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
#   
    beta[(vois==3)+(vois==7)] = phi_theta(pi/2,angle) 
    beta[(vois==4)+(vois==0)] = phi_theta(3*pi/4.,angle)
    beta[(vois==5)+(vois==1)] = phi_theta(0,angle)
    beta[(vois==6)+(vois==2)] = phi_theta(pi/4.,angle)
    
    
#    beta[(vois==3)+(vois==7)] = phi_theta(0,angle) 
#    beta[(vois==4)+(vois==0)] = phi_theta(pi/4.,angle)
#    beta[(vois==5)+(vois==1)] = phi_theta(pi/2.,angle)
#    beta[(vois==6)+(vois==2)] = phi_theta(3.*pi/4.,angle)

    

#    if angle!=0:
#        beta = phi_theta_0 + (1-phi_theta_0)*beta
#    else:
#        beta=np.ones_like(vois)

    if angle==0:
        beta=np.ones_like(vois)
       
    #beta[vois==-1] = 0.   
    if np.ndim(beta) ==3:
        beta /= beta.sum(axis=2)[:,:,np.newaxis]#beta_sum
    elif np.ndim(beta) == 1:
        beta /=beta.sum()
        
#    beta -=0.5
#    beta /= beta.sum(axis=2)[:,:,np.newaxis]#beta_sum
#    beta/=(2+np.cos(angle))
    
#    if angle ==0:
#        beta /=1.5
    #beta /= 2.#((beta**2).sum())**(0.5)
#    beta = phi_theta_0*np.ones_like(beta) + beta*(1-phi_theta_0)
    return beta
    
    
def psi_ising(x_1,x_2,fuzzy,alpha,delta,phi_uni):
    if fuzzy == False:
        
        #alpha = par.alpha
        #res = phi_uni*x_1 + alpha*(x_1 != x_2) - alpha*(x_1==x_2)#  np.abs(x_1-x_2)
#        res = alpha*(x_2 != x_1) - alpha*(x_2==x_1)
        res = alpha * (1.-2.*(x_2==x_1))
#    elif fuzzy==True:
#        # bonne combinaison : a=3, b=1.5,c=0
#
#        ha_eq =   ((x_1==0)*(x_2==0) + (x_1==1)*(x_2==1)) > 0
#        ha_diff = ((x_1==1)*(x_2==0) + (x_1==0)*(x_2==1)) > 0 #stands for "or"
#          
#        fuzz_1 = (x_1 != 0)  * (x_1 !=1)        
#        fuzz_2 = (x_2 != 0)  * (x_2 !=1)
#        fuzz = (fuzz_1 + fuzz_2) > 0 
#        
#        res = phi_uni*x_1 -alpha * ha_eq + alpha* ha_diff + (-beta* (1-2*abs(x_2-x_1)) + delta) * fuzz

        
    return res    
    
    
    
    
def init_champs(par):
    
    S0 = par.S0
    S1 = par.S1

    if par.init_method == 'astro':
        centre1 = np.array([S0/2,S1/2])
        r1 = 5
        y,x = np.ogrid[0:S0,0:S1]
        
        cercle1 = 1-( (x-centre1[0])**2+(y-centre1[1])**2 ).astype(float)/r1**2 
        dist_centre = np.sqrt((x-centre1[0])**2+(y-centre1[1])**2 )


    if par.fuzzy == False:
#        X_init = np.random.choice(np.array([0.,1.]),p=np.array([0.52,0.48]),size=(par.S0,par.S1))
        X_init = np.random.choice(par.x_range,size=(par.S0,par.S1))
        
        if par.init_method=='astro':
            X_init[cercle1>0] = 1.# += 0.5* (cercle1>0)
            X_init[dist_centre > dist_centre[int(S0/2),0]]  = 0
        
    elif par.fuzzy == True:
        
        ran_fuzz = np.arange(0,par.nb_fuzzy+1)/par.nb_fuzzy
        if par.init_method=='astro':
                        
            X_init = np.random.choice(ran_fuzz, size=(par.S0,par.S1)) - .25/(par.nb_fuzzy+1) 
                
            X_init += 0.5* (cercle1>0)
            X_init[dist_centre > 0.9*dist_centre[int(S0/2),0]]  -= 0.5
                    
            X_init[X_init>1] = 1
            X_init[X_init<0] = 0 
        else:
            X_init = np.random.choice(ran_fuzz, size=(par.S0,par.S1)) #- .25/(par.nb_fuzzy+1) 


    
    return X_init  

      