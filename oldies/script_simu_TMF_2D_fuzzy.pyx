# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 09:45:48 2015

@author: courbot
"""



import numpy as np 
#import numpy.ma as ma
import sys 
import matplotlib.pyplot as plt
import time
import scipy.stats as st
import scipy.ndimage.morphology as morph

from matplotlib import animation

import parameters
import image_tools as it
from image_tools import get_num_voisins, get_vals_voisins

from numpy import abs
from numpy import cos

def psi(x_1,x_2,fuzzy,alpha,beta,delta):
    if fuzzy == False:
        #alpha = par.alpha
        res = alpha*(x_1 != x_2) - alpha*(x_1==x_2)#  np.abs(x_1-x_2)
        

    elif fuzzy==True:
        # bonne combinaison : a=3, b=1.5,c=0
        #alpha = par.alpha # ceci pourra servir a creer une asymetrie entre blanc et noir
        #beta = par.beta
        #delta = par.delta
        ha_eq =   ((x_1==0)*(x_2==0) + (x_1==1)*(x_2==1)) > 0
        ha_diff = ((x_1==1)*(x_2==0) + (x_1==0)*(x_2==1)) > 0 #stands for "or"
          
        fuzz = (ha_eq==0) *  (ha_diff==0)
        res = -alpha * ha_eq + alpha* ha_diff + (-beta* (-2*abs(x_2-x_1)+1) + delta) * fuzz
        
    return res

def phi(a_1,a_2):
    
    return (cos(a_1-a_2))**2

def gen_beta(vois, angle,phi_theta_0):

    beta = np.zeros_like(vois) ; beta = beta.astype(float)
    pi = np.pi
   
    beta[(vois==3)+(vois==7)] = phi(0,angle) 
    beta[(vois==4)+(vois==0)] = phi(pi/4,angle)
    beta[(vois==5)+(vois==1)] = phi(pi/2,angle)
    beta[(vois==6)+(vois==2)] = phi(3*pi/4,angle)
    
    #beta_0 = 0.1
    beta = phi_theta_0 + (1-phi_theta_0)*beta

    return beta
      
def gen_champs(par):
    """ Generation d'un champs de Gibbs a partir d'une initialisation
    angle en radians    
    """
    S0 = par.S0
    S1 = par.S1    
    fuzzy = par.fuzzy
    alpha = par.alpha # ceci pourra servir a creer une asymetrie entre blanc et noir
    beta = par.beta
    delta = par.delta
    phi_theta_0 = par.phi_theta_0 
    X_init = init_champs(par)    
    
    X_courant = np.copy(X_init)
    #proba_courant = np.zeros_like(X_init)
    #num_ecart = np.zeros(shape=(par.nb_iter))
    X_res = np.zeros(shape=(S0,S1,par.nb_iter))

    # Retrieving locals neighborhoods:
    Vois = np.zeros(shape=(S0,S1,8))
    Beta = np.ones_like(Vois)
    for i in xrange(S0):
        for j in xrange(S1):
            Vois[i,j,:] = get_num_voisins(i,j,X_init)
            
            if par.anisotropic == True:
                    if np.isnan(par.angle[i,j])==0:
                         Beta[i,j,:] =  gen_beta(Vois[i,j],par.angle[i,j],phi_theta_0)

    # Generating all candidates:

    if par.fuzzy==True:
        X_new_tout = np.random.choice(np.arange(0,par.nb_fuzzy+1)/par.nb_fuzzy, size=(S0,S1,par.nb_iter)) # est-ce juste ? ou faire deux tirages ?
    elif par.fuzzy==False:
        X_new_tout = np.random.choice(np.array([0,1]), size=(S0,S1,par.nb_iter))
        
    for k in xrange(par.nb_iter):
        # Proposition de candidats
        X_new = X_new_tout[:,:,k]
        #print 'iteration...%2.f'%k

        for i in xrange(S0):
            for j in xrange(S1):
                
                vals = get_vals_voisins(i,j,X_courant)
                vois = Vois[i,j,:]


                energie_courant = Beta[i,j,:] * psi( X_courant[i,j], vals,fuzzy,alpha,beta,delta) * (vois >=0) 
                #proba_courant = np.exp(-(energie_courant.sum()))#/cst_norm
                
                energie_new = Beta[i,j,:] * psi( X_new[i,j], vals,fuzzy,alpha,beta,delta) * (vois >=0)
                #proba_new = np.exp(-(energie_new.sum()))
                
                en_tot_courant = energie_courant.sum()
                en_tot_new = energie_new.sum()
                if par.fuzzy==True:
                    if ((X_courant[i,j]==0) + (X_courant[i,j]==1)) ==0:
                        #proba_courant /=par.nb_fuzzy  
                        en_tot_courant += np.log(par.nb_fuzzy)
                        
                    if ((X_new[i,j]==0) + (X_new[i,j]==1)) == 0:
                        en_tot_new += np.log(par.nb_fuzzy)
                        #proba_new /=par.nb_fuzzy                
                
                #q = proba_new / proba_courant
                #if q > 1:
                if en_tot_new < en_tot_courant:                
                    X_courant[i,j] = X_new[i,j]
                    #num_ecart[k] +=1

        X_res[:,:,k] = X_courant        
                 
    return X_courant, X_res


def get_theta_isocontour(Im):
    dx, dy = np.gradient(Im)

    theta = np.zeros_like(Im)
    
    theta[dx!=0] = np.arctan(dy[dx!=0]/dx[dx!=0])
    theta[dx==0] = np.sign(dy[dx==0]) * np.pi/2
    theta = theta + np.pi * (dx< 0)

    theta_iso = theta#%np.pi
    
    return theta_iso

def init_champs(par):
    
    
    if par.fuzzy==False:
        X_init = np.random.choice(np.array([0,1]),size=(par.S0,par.S1))
    elif par.fuzzy==True:
        X_init = np.random.choice(np.arange(0,par.nb_fuzzy+1)/par.nb_fuzzy, size=(par.S0,par.S1))
    
    return X_init


# parametres utiles:
fuzzy = True
nb_iter = 20

S0 = 50
S1 = 50
phi_theta_0 = 0.1
###################
centre1 = np.array([S0/2,S1/2])
r1 = 5
y,x = np.ogrid[0:S0,0:S1]
cercle1 = 1-( (x-centre1[0])**2+(y-centre1[1])**2 ).astype(float)/r1**2 
angle = get_theta_isocontour(cercle1)
###################

plt.close('all')
nb_li = 1
nb_col = 1


#%%
plt.figure(figsize=(5*nb_col,5*nb_li))
start=time.time()
X_iso, X_iso_tout = gen_champs(parameters.Params(S0 = S0,
                                                 S1 = S1,
                                                 nb_iter=nb_iter,
                                                 fuzzy=fuzzy,
                                                 anisotropic=False,
                                                 beta=1,
                                                 angle=angle,
                                                 phi_theta_0 = phi_theta_0)
                                )
#plt.subplot(nb_li,nb_col,3)
plt.imshow(X_iso.T, interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin=0,vmax=1); 
plt.axis('off')
plt.tight_layout()

print time.time()-start
##%%
#plt.figure(figsize=(5*nb_col,5*nb_li))
#angle[25,25] = np.nan
#plt.quiver(-np.sin(angle[::5,::5]),-np.cos(angle[::5,::5]),pivot='mid')
#plt.xlim((0.5,9.5))
#plt.ylim((0.5,9.5))
#plt.tight_layout()
#plt.axis('off')
#plt.savefig('ani_ray.eps', format='eps',dpi=100)
#%%
#
#
#nb_fuzzy = 32.
#nb_iter_gen = 10
#
##%%
## Generation du processus auxiliaire
#U = np.zeros(shape=(S0,S1))
#
#< r1**2
#dist1 = np.sqrt((x-centre1[0])**2+(y-centre1[1])**2 )
#centre2 = np.array([50,50])
#r2 = 5
#cercle2 = 1-( (x-centre2[0])**2+(y-centre2[1])**2 ).astype(float) /r2**2 #< r2**2
#dist2 = np.sqrt((x-centre2[0])**2+(y-centre2[1])**2 )
#cercles = ((cercle1>0)) + ((cercle2>0))
#
##%%
#V_0 = np.random.choice(np.arange(0,nb_fuzzy+1)/nb_fuzzy, size=(S0,S1))#+ 0.5* (cercle1>0)
##V_0 = st.norm.rvs(0.4,0.25, size=(S0,S1)) + 0.5* (cercle1>0)
#V_0[V_0>1] = 1
#V_0[V_0<0] = 0
#
#V_0 = np.round(V_0 * nb_fuzzy) / nb_fuzzy
##%%
#
#
##
#start = time.time()
#V_iso, V_iso_tout = gen_champs(V_0, S0,S1,nb_iter_gen,nb_fuzzy=nb_fuzzy,angle=0,type_psi='abs',isotropic=True)
#temps = time.time()-start
#print temps

# Orientations des isocontours
# Ce bloc a l'air interessant. mis de cote pour plus tard.

#%%
#angle = get_theta_isocontour(cercle1) #* (dist1<dist2) + get_theta_isocontour(cercle2) * (dist1> dist2)
#angle[dist1==dist2] = np.NAN
##
#start = time.time()
#V_ani, V_ani_tout = gen_champs(V_0, S0,S1,nb_iter_gen,angle=angle,nb_fuzzy=nb_fuzzy,type_psi='fuzzy',isotropic=False)
#temps = time.time()-start
#print temps


#%%
# Affichages
#plt.close('all')
#nb_li = 1;
#nb_col = 3;
#
#
#def animate1(i):
#    plt.cla()
#    plt.imshow(V_iso_tout[:,:,i].T, interpolation='nearest', origin='lower', cmap=plt.cm.hot); 
#    
#
#
#fig=plt.figure(figsize=(6*nb_col,6*nb_li))
#
#plt.subplot(nb_li,nb_col,1)
#plt.imshow(V_0.T, interpolation='nearest', origin='lower', cmap=plt.cm.afmhot,vmin=0,vmax=1); #plt.colorbar()
#plt.title('Initialisation')
#
#
#
#
#plt.subplot(nb_li,nb_col,2)
#plt.imshow(V_iso.T, interpolation='nearest', origin='lower', cmap=plt.cm.afmhot,vmin=0,vmax=1); #plt.colorbar()
#plt.title('Isotropic')
#
#plt.subplot(nb_li,nb_col,3)
#plt.imshow((V_ani).T, interpolation='nearest', origin='lower', cmap=plt.cm.afmhot,vmin=0,vmax=1); #plt.colorbar()
#
#plt.title('Anisotropic')
#

#plt.subplot(nb_li,nb_col,4)
#plt.imshow((V_ani).T, interpolation='nearest', origin='lower', cmap=plt.cm.gray,vmin=0,vmax=1); #plt.colorbar()
#plt.contour(V_ani.T,np.arange(0.1,1,0.1))
##plt.imshow((V_ani>V_ani.mean()).T, interpolation='nearest', origin='lower', cmap=plt.cm.afmhot,vmin=0,vmax=1); #plt.colorbar()
#plt.title('Anisotropic : contour')


#plt.subplot(nb_li,nb_col,8)
#anim2 = animation.FuncAnimation(fig, animate2, frames=V_ani_tout.shape[2])



#plt.subplot(nb_li,nb_col,3)
#plt.imshow(theta_isocontour,interpolation='nearest', origin='lower', cmap=plt.cm.jet)
#plt.title('Theta - init')
#
#plt.subplot(nb_li,nb_col, 4)
#plt.imshow(V, cmap=plt.cm.Blues,interpolation='nearest', origin='lower')
#plt.quiver(V*np.cos(theta_isocontour), V*np.sin(theta_isocontour))



#plt.tight_layout()

#plt.savefig('current.eps', format='eps',dpi=100)
#%%
#plt.figure(figsize=(5,5))
#plt.imshow(X_iso.T, interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin=0,vmax=1); 
#plt.axis('off')
#plt.tight_layout()

#%%%
#
#def animate2(i):
#    plt.cla()
#    plt.imshow(V_ani_tout[:,:,i].T, interpolation='nearest', origin='lower', cmap=plt.cm.gray,vmin=0,vmax=1); 
#    #plt.contour(V_ani_tout[:,:,i].T,np.arange(0.1,1,0.1))
#    plt.title('Iteration %0.f'%i)
#
#
#fig2=plt.figure(figsize=(10,10))
#anim = animation.FuncAnimation(fig2, animate2, frames=V_ani_tout.shape[2])
#

