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
import fields_tools as ft
import gibbs_sampler as gs


def plot_directions(angle, intensite):
    
    S0 = angle.shape[0]
    S1 = angle.shape[1]
    
    y,x = np.ogrid[0:S0,0:S1]
    

    deb_x = np.tile(x,(S0,1)) - 0.5*np.sin(angle) * intensite
    deb_y = np.tile(y,(1,S1)) - 0.5*np.cos(angle) * intensite
    
    fin_x = np.tile(x,(S0,1)) + 0.5*np.sin(angle) * intensite
    fin_y = np.tile(y,(1,S1)) + 0.5*np.cos(angle) * intensite
    
    
    for i in range(0,S0,2):
        for j in range(0,S1,2):
            if angle[i,j] != 0:
                plt.plot((deb_x[i,j],fin_x[i,j]), (deb_y[i,j],fin_y[i,j]) ,'k')
    plt.xlim((-0.5,S1-0.5))
    plt.ylim((-0.5,S0-0.5))  
    
def get_nom(par):
    
    if par.fuzzy==True:
        deb = 'fuzzy-'
    else:
        deb = 'hard-'
    
    nom = deb + 'a' + str(par.alpha) +'-' + 'b' + str(par.beta) +'-' + 'd' + str(par.delta) +'-' + 'p' + str(par.phi_theta_0) 
           
    # remplacer les '.' par des '_'
    nom = nom.translate('_', '.')
    return nom    
#
#def get_vals_voisins_tout(image):
#    """
#           --------------
#        y+1 | 6 | 5 | 4 |
#           --------------
#          y | 7 |   | 3 |
#           --------------
#        y-1 | 0 | 1 | 2 |
#           --------------
#            x-1 | x | x+1
#    """
#    
#    S0 = image.shape[0]
#    S1 = image.shape[1]
#    
#    vals = np.zeros(shape=(S0,S1,8)) # 8-voisinage
#    
#    im = np.zeros(shape=(S0+2,S1+2)) # image with 1-px 0-padding
#    im[1:S0+1,1:S0+1] = image
#    
#    vals[:,:,0] = im[0:S0,   0:S1]
#    vals[:,:,1] = im[1:S0+1, 0:S1]
#    vals[:,:,2] = im[2:S0+2, 0:S1]
#    vals[:,:,3] = im[2:S0+2, 1:S1+1]
#    vals[:,:,4] = im[2:S0+2, 2:S1+2]
#    vals[:,:,5] = im[1:S0+1, 2:S1+2]
#    vals[:,:,6] = im[0:S0,   2:S1+2]
#    vals[:,:,7] = im[0:S0,   1:S1+1]
#    
#    return vals
    
    
#def psi_ising_tout(x_1,x_2,fuzzy,alpha,beta,delta,phi_uni):
#    """
#    x_1 : image
#    x_2 : image stack
#    
#    """
#    if fuzzy == False:
#        #alpha = par.alpha
#    
#        res = phi_uni*x_1 + alpha*(x_1 != x_2) - alpha*(x_1==x_2)#  np.abs(x_1-x_2)
#        
#
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
#
#        
#    return res    
        
    
#def calc_proba_vois_tout(x,vals_tr,fuzzy,nb_fuzzy, beta_tr, alpha,beta,delta,phi_uni,vois_tr):
#    energie_courant = (beta_tr * ft.psi_ising( x, vals_tr,fuzzy,alpha,beta,delta,phi_uni)    )*(vois_tr >=0) 
#    #print energie_courant.shape
#    proba_courant = np.exp(-(energie_courant.sum(axis=2)))
##    
#    isfuzzy = (x!=0)*(x!=1)
#    
#    proba_courant[isfuzzy] /=(nb_fuzzy)
#        
#    return proba_courant
#    
 
#%%
    

def get_dir(X,par):
    
    S0 = par.S0
    S1 = par.S1 
    v_range = par.v_range
    #v_range = v_range[:-1]
    nb_nn = 19# number of nearest neighbor
    dx,dy = np.gradient(X)
    an = np.arctan2(dy,dx)

    ux, uy = np.ogrid[0:S0,0:S1]
    ux = np.tile(ux,(1,S1))
    uy = np.tile(uy,(S0,1))
    
    # removing unknown values
    an[an==0] +=np.nan
    
    an_flat = an.flatten()
    an_interp = np.copy(an)
    
    for i in range(an.shape[0]):
        for j in range(an.shape[1]):
            #if np.isnan(an[i,j]):
                dist = np.sqrt((i-ux)**2 + (j-uy)**2)
                dist_flat = dist.flatten()
                
                dist_flat[np.isnan(an_flat)] = 10000
                ind_min = np.argmin(dist_flat)
                ind_sort = np.argsort(dist_flat)
                an_interp[i,j] = 0
                for nn in range(nb_nn):
                    an_interp[i,j] += an_flat[ind_sort[nn]]/nb_nn
      
    #an_interp[an_interp==0] +=np.nan
    # recasting into the known range         
    an_interp = (an_interp+np.pi/2)%np.pi      
  
    v_range_new = v_range[v_range!=0]
    for i in range(an.shape[0]):
        for j in range(an.shape[1]):
            if np.isnan(an_interp[i,j])==0:
                ecart = np.abs(an_interp[i,j] - v_range_new)
                ind_min = np.argmin(ecart)
        
                an_interp[i,j] =v_range_new[ind_min]
    an_interp[np.isnan(an_interp)] = 0              
            
    return an_interp        
#%%    
    
# parametres utiles:
fuzzy = False
nb_iter = 50

S0 = 64
S1 = 64
phi_theta_0 = 0.
###################
#centre1 = np.array([S0/2,S1/2])
#r1 = 5
#y,x = np.ogrid[0:S0,0:S1]
#cercle1 = 1-( (x-centre1[0])**2+(y-centre1[1])**2 ).astype(float)/r1**2 
#angle = it.get_theta_isocontour(cercle1)
####################
v_range = np.array([3*np.pi/4,np.pi/4,np.pi/2,np.pi])
v_range = np.array([3*np.pi/4,np.pi/4])
V=np.zeros(shape=(S0,S1))
V[:,:int(S1/2)] = np.pi/4
V[:,int(S1/2):] = 3*np.pi/4

#%%

par = parameters.ParamsGibbs(S0 = S0,
                             S1 = S1,
                             type_pot = 'ising',
                             phi_uni = 0.,
                             nb_iter=nb_iter,
                             fuzzy=False,
                             anisotropic=True,   
                             angle=V,
                             beta = 1.,
                             phi_theta_0 = 0.,
                             alpha = 1.,
                             alpha_v = 20.,
                             delta = 0.,
                             init_method = 'std',
                             v_range = v_range,
                             nb_fuzzy = 256.                             
                             )# beta=1.25,

#
#start=time.time()                             
#X,X_tout,energie = gs.gen_champs(par)
#temps =  (time.time()-start)
#print 'Champ de Gibbs : %.0f iterations et %.2f s - %.3f s/iter.'%(X_tout.shape[2]-1, temps,temps/(X_tout.shape[2]-1)  )

##
start=time.time()  

#par.V = V
#parx = gs.gen_champs_fast(par, generate_v=False, generate_x=True, use_y=False)
#temps =  (time.time()-start)
#print 'Champ de Gibbs rapide : %.0f iterations et %.2f s - %.3f s/iter.'%(parx.nb_iter_conv, temps,temps/(parx.nb_iter_conv)  )
#
#
#start=time.time()  
#par.X = np.copy(parx.X_res[:,:,-1])
#
#parv = gs.gen_champs_fast(par, generate_v=True, generate_x=False, use_y=False)
#temps =  (time.time()-start)
#print 'Champ de Gibbs rapide : %.0f iterations et %.2f s - %.3f s/iter.'%(parv.nb_iter_conv, temps,temps/(parv.nb_iter_conv)  )
      
start=time.time()  

parvx = gs.gen_champs_fast(par, generate_v=True, generate_x=True, use_y=False)
temps =  (time.time()-start)
print 'Champ de Gibbs rapide : %.0f iterations et %.2f s - %.3f s/iter.'%(parvx.nb_iter_conv, temps,temps/(parvx.nb_iter_conv)  )
      


#%%  
nb_li = 2
nb_col = 2
plt.close('all')
plt.figure(figsize=(5*nb_col,5*nb_li))
#
plt.subplot(nb_li,nb_col,1)
plt.imshow(parx.X_res[:,:,-1].T, interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin=0,vmax=1); 
plt.axis('off')



plt.subplot(nb_li,nb_col,2)
plt.imshow(parv.V_res[:,:,-1].T, interpolation='nearest', origin='lower',  cmap=plt.cm.Spectral,vmin=0)
plot_directions(parv.V_res[:,:,-1].T, np.ones_like(parv.V_res[:,:,-1].T))
plt.axis('off')


plt.subplot(nb_li,nb_col,3)
plt.imshow(parvx.X_res[:,:,-1].T, interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin=0,vmax=1); 
plt.axis('off')



plt.subplot(nb_li,nb_col,4)
plt.imshow(parvx.V_res[:,:,-1].T, interpolation='nearest', origin='lower',  cmap=plt.cm.Spectral,vmin=0)
plot_directions(parvx.V_res[:,:,-1].T, np.ones_like(parvx.V_res[:,:,-1].T))
plt.axis('off')





plt.tight_layout()    
##%%
#nb_li = 1
#nb_col = 1
#plt.close('all')
#plt.figure(figsize=(5*nb_col,5*nb_li))
#
#plt.subplot(nb_li,nb_col,1)
#plt.imshow(V.T, interpolation='nearest', origin='lower',  cmap=plt.cm.Spectral,vmin=0)
#plot_directions(V.T, np.ones_like(parv.V_res[:,:,-1].T))
#plt.axis('off')
#
#
#plt.tight_layout()    
    
#%%

#plt.close('all')
#nb_li = 1
#nb_col = 2
#
#plt.figure(figsize=(5*nb_col,5*nb_li))
#
#plt.subplot(nb_li,nb_col,1)
#plt.imshow(X_tout[:,:,-1].T, interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin=0,vmax=1); 
#plt.axis('off')
#
##plt.savefig('fuzzy_ray_astro.eps', format='eps',dpi=100)
#
#
#plt.subplot(nb_li,nb_col,2)
#
#plt.plot(energie.mean(axis=(0,1)))
#
#
#plt.tight_layout()
#%%
#def animate1(i):
#    plt.cla()
#    plt.imshow(X_tout[:,:,i].T, interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin=0,vmax=1); 
#    
#fig2=plt.figure(figsize=(10,10))
#anim = animation.FuncAnimation(fig2, animate1, frames=X_tout.shape[2])
#%%
