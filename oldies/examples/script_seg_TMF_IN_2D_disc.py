# -*- coding: utf-8 -*-
"""
Ce script simule et segmente un champ de Markov Triplet. Les images sont 2D et 
les classes sont discretes pour les classes et le processus auxilliaire. Le 
bruit est independant ("TMF-IN").
Le processus auxilliaire represente, a travers un angle, une direction
privilegiee pour les intensites du champ de Markov.

:date 01 dec 2015
:author JBC
"""



import numpy as np 
#import numpy.ma as ma
import sys 
import matplotlib.pyplot as plt
import time
import scipy.stats as st
import scipy.ndimage.morphology as morph

from matplotlib import animation
import scipy.cluster.vq as cvq


def psi(x_1,x_2,nom='abs'):
    if nom=='abs':
        res = np.abs(x_1-x_2)
    
    elif nom=='sq':
        res = (x_1-x_2)**2
    
    elif nom=='kro':
        res =1-(x_1==x_2)
        
    elif nom=='kro2':
        res =1-2*(x_1==x_2)
    
    return res

def phi(a_1,a_2):
    
    return (np.cos(a_1-a_2))**2

def gen_beta(vois, angle):
    
    beta = np.zeros_like(vois) ; beta = beta.astype(float)
    pi = np.pi
    # This diminish the values on the perpendicular axis
#    angle_utile = (angle+np.pi/2)%(2*pi)
#   
#    beta[(vois==3)+(vois==7)] = psi(0,angle_utile, type_psi) 
#    beta[(vois==4)+(vois==0)] = psi(pi/4,angle_utile,type_psi)
#    beta[(vois==5)+(vois==1)] = psi(pi/2,angle_utile,type_psi)
#    beta[(vois==6)+(vois==2)] = psi(3*pi/4,angle_utile,type_psi)
#    

   
    beta[(vois==3)+(vois==7)] = phi(0,angle) 
    beta[(vois==4)+(vois==0)] = phi(pi/4,angle)
    beta[(vois==5)+(vois==1)] = phi(pi/2,angle)
    beta[(vois==6)+(vois==2)] = phi(3*pi/4,angle)
    
    beta_0 = 0
    beta = beta_0 + (1-beta_0)*beta
    #beta/=beta.max()
    #beta = np.sin(beta)
    return beta

def get_num_voisins(pos,image):
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
    
    """
    if pos.size==2:
        S0 = image.shape[0]-1
        S1 = image.shape[1]-1
        
        # convenience/brevity notation
        x = pos[0]
        y = pos[1]
        if x <  S0 and x > 0 and y <  S1 and y > 0 :
            voisins = np.array([0,1,2,3,4,5,6,7])            
            
        if x == 0:
            if y==0:
                voisins = np.array([3,4,5,-1,-1,-1,-1,-1])   
            elif y==S1:
                voisins = np.array([1,2,3,-1,-1,-1,-1,-1])
            else:
                voisins = np.array([1,2,3,4,5,-1,-1,-1])
        elif x == S0:
            if y == 0:
                voisins=np.array([5,6,7,-1,-1,-1,-1,-1])
            elif y == S1:
                voisins = np.array([7,0,1,-1,-1,-1,-1,-1])
            else:
                voisins = np.array([5,6,7,0,1,-1,-1,-1])
        elif y==0:
            voisins = np.array([3,4,5,6,7,-1,-1,-1])
        elif y == S1:
            voisins = np.array([7,0,1,2,3,-1,-1,-1])
        
        return voisins
        
def get_vals_voisins(pos,image):
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
    
    """
    if pos.size==2:
        S0 = image.shape[0]-1
        S1 = image.shape[1]-1
        
        # convenience/brevity notation
        x = pos[0]
        y = pos[1]
        if x <  S0 and x > 0 and y <  S1 and y > 0 :
            vals = np.array([image[x-1,y-1],image[x,y-1],image[x+1,y-1],image[x+1,y], image[x+1,y+1],image[x,y+1],image[x-1,y+1],image[x-1,y]])           
            
        if x == 0:
            if y==0:
                vals = np.array([image[x+1,y],image[x+1,y+1],image[x+1,y+1],0,0,0,0,0])  
            elif y==S1:
                vals = np.array([image[x,y-1],image[x+1,y-1],image[x+1,y],0,0,0,0,0])
            else:
                vals = np.array([image[x,y-1],image[x+1,y-1],image[x+1,y], image[x+1,y+1],image[x,y+1],0,0,0])
        elif x == S0:
            if y == 0:
                vals = np.array([image[x,y+1],image[x-1,y+1], image[x-1,y],0,0,0,0,0])
            elif y == S1:
                vals = np.array([image[x-1,y],image[x-1,y-1], image[x,y-1],0,0,0,0,0])
            else:
                vals = np.array([image[x,y+1],image[x-1,y+1], image[x-1,y],image[x-1,y-1], image[x,y-1] ,0,0,0])
        elif y==0:
            vals = np.array([image[x+1,y],image[x+1,y+1],image[x+1,y+1],image[x-1,y+1], image[x-1,y],0,0,0])
        elif y == S1:
            vals = np.array([image[x-1,y],image[x-1,y-1], image[x,y-1],image[x+1,y-1],image[x+1,y] ,0,0,0])
        
        return vals

def calc_proba_gibbs(x, vals, u, vals_u, vois, beta, type_psi):
    
        psi_x = psi( x, vals,type_psi)
        psi_u = psi( u, vals_u,type_psi)                    
        
        energie = ( beta * (psi_x + psi_u)  )* (vois >=0)
        
        nb_elt = (vois >=0).sum()
        cst_norm = np.exp(-nb_elt*(2))
        proba = np.exp(-(energie.sum()))/cst_norm 
        
        return proba
        
        
def calc_proba_gibbs_xy(x, vals, y,sig, vois, type_psi):
    
        psi_x = psi( x, vals,type_psi)
        
        likelihood = st.norm.pdf(y, loc=x,scale=sig)                   
        # faux ? log_likelihood n'intervient q'une fois, pas 8 fois.
        energie = (psi_x - np.log(likelihood) ) * (vois >=0)
        
        nb_elt = (vois >=0).sum()
        cst_norm = np.exp(-nb_elt)
        proba = np.exp(-(energie.sum()))/cst_norm 
        
        return proba
        
        
def calc_proba_gibbs_xyu(x, vals, y,sig,u, vals_u, vois,beta, type_psi):
    
        psi_x = psi( x, vals,type_psi)
        psi_u = psi( u, vals_u,type_psi)  
        likelihood = st.norm.pdf(y, loc=x,scale=sig)                   
        
        energie = ( beta * (psi_x + psi_u) - np.log(likelihood)  )* (vois >=0)
        
        nb_elt = (vois >=0).sum()
        cst_norm = np.exp(-nb_elt*2)
        proba = np.exp(-(energie.sum()))/cst_norm 
        
        return proba
        
        
def gen_champs_gibbs(X_init,S0,S1,nb_iter, xmin,xmax,U,type_psi='abs',isotropic=True):
    """ Generation d'un champs de Gibbs a partir d'une initialisation.
    Genere x selon p(X | U), U étant l'angle
    """
    X_courant = np.copy(X_init)
    proba_courant = np.zeros_like(X_init)
    num_ecart = np.zeros(shape=(nb_iter))
    X_res = np.zeros(shape=(S0,S1,nb_iter))
    
    # Retrieving locals neighborhoods:
    Vois = np.zeros(shape=(S0,S1,8))
    Vals_u = np.zeros_like(Vois)
    Beta = np.ones_like(Vois)
    for i in range(S0):
        for j in range(S1):
            Vois[i,j,:] = get_num_voisins(np.array([i,j]),X_init)
            
            if isotropic == False:
                    if np.isnan(U[i,j])==0:
                         Beta[i,j,:] =  gen_beta(Vois[i,j],U[i,j])
                         Vals_u[i,j,:] = get_vals_voisins(np.array([i,j]),U)
    
    #energie_totale = np.sqrt((X_init**2).sum())
    
    # Generating all candidates:
    X_new_tout = np.random.uniform(low = xmin, high = xmax, size=(S0,S1,nb_iter)) > 0.5   
    
    for k in range(nb_iter):
        # Proposition de candidats
        X_new = X_new_tout[:,:,k]

        for i in range(S0):
            for j in range(S1):
                
                vals = get_vals_voisins(np.array([i,j]),X_courant)
                vois = Vois[i,j,:]
                vals_u = Vals_u[i,j,:]

                if k == 0:
                    proba_courant[i,j] = calc_proba_gibbs(X_courant[i,j], vals, U[i,j], vals_u, vois,Beta[i,j,:], type_psi)

                proba_new = calc_proba_gibbs(X_new[i,j], vals, U[i,j], vals_u, vois,Beta[i,j,:], type_psi)


                q = proba_new / proba_courant[i,j]
                if q > 1:
                    X_courant[i,j] = X_new[i,j]
                    proba_courant[i,j] = proba_new
                    num_ecart[k] +=1


        X_res[:,:,k] = X_courant        
                                  
                    
    return X_courant, X_res
    
    
def gen_u_gibbs(U_init,S0,S1,nb_iter, u_range,X,Y,sig,type_psi='abs'):
    """ Generation d'un champs de Gibbs a partir d'une initialisation
    Genere u selon p(U | X), U étant l'angle 
    """
    U_courant = np.copy(U_init)
    proba_courant = np.zeros_like(U_init)
    num_ecart = np.zeros(shape=(nb_iter))
    U_res = np.zeros(shape=(S0,S1,nb_iter))
    
    # Retrieving locals neighborhoods:
    Vois = np.zeros(shape=(S0,S1,8))
    Vals = np.zeros_like(Vois)
    Beta_courant = np.zeros_like(Vois)
    Beta_possible = np.zeros(shape=(S0,S1,8,u_range.size))
    for i in range(S0):
        for j in range(S1):
            Vois[i,j,:] = get_num_voisins(np.array([i,j]),U_init)
            Vals[i,j,:] = get_vals_voisins(np.array([i,j]),X)
            for u_inst in range(u_range.size):
                Beta_possible[i,j,:,u_inst] = gen_beta(Vois[i,j,:],u_range[u_inst]) 
            


    
    #energie_totale = np.sqrt((X_init**2).sum())
    
    # Generating all candidates:
    U_new_tout = np.random.choice(u_range, size=(S0,S1,nb_iter))  
    
    for k in range(nb_iter):
        # Proposition de candidats
        U_new = U_new_tout[:,:,k]

        for i in range(S0):
            for j in range(S1):
                
                vals = Vals[i,j,:]
                vois = Vois[i,j,:]
                
                vals_u = get_vals_voisins(np.array([i,j]),U_courant)

                if k == 0:
                    Beta_courant[i,j,:] = Beta_possible[i,j,:,np.where(u_range==U_courant[i,j])]
                    proba_courant[i,j] = calc_proba_gibbs_xyu(X[i,j], vals,Y[i,j],sig, U_courant[i,j], vals_u, vois,Beta_courant[i,j,:], type_psi)


                Beta_new = Beta_possible[i,j,:,np.where(u_range==U_new[i,j])]
                proba_new = calc_proba_gibbs_xyu(X[i,j], vals,Y[i,j],sig, U_new[i,j], vals_u, vois, Beta_new, type_psi)


                q = proba_new / proba_courant[i,j]
                if q > 1:
                    U_courant[i,j] = U_new[i,j]
                    Beta_courant[i,j,:] = Beta_new
                    proba_courant[i,j] = proba_new
                    num_ecart[k] +=1


        U_res[:,:,k] = U_courant        
                                  
                    
    return U_courant, U_res, num_ecart   


def gen_x_gibbs(X_init,S0,S1,nb_iter, x_range,Y,sig,type_psi='abs',isotropic=True):
    """ Generation d'un champs de Gibbs a partir d'une initialisation.
    Genere x selon p(X | Y), Y etant l'observation 
    """
    X_courant = np.copy(X_init)
    proba_courant = np.zeros_like(X_init)
    num_ecart = np.zeros(shape=(nb_iter))
    X_res = np.zeros(shape=(S0,S1,nb_iter))
    
    # Retrieving locals neighborhoods:
    Vois = np.zeros(shape=(S0,S1,8))
    #Vals_u = np.zeros_like(Vois)
    #Beta = np.ones_like(Vois)
    for i in range(S0):
        for j in range(S1):
            Vois[i,j,:] = get_num_voisins(np.array([i,j]),X_init)
#            
#            if isotropic == False:
#                    if np.isnan(U[i,j])==0:
#                         Beta[i,j,:] =  gen_beta(Vois[i,j],U[i,j])
#                         Vals_u[i,j,:] = get_vals_voisins(np.array([i,j]),U)
    
    #energie_totale = np.sqrt((X_init**2).sum())
    
    # Generating all candidates:
    X_new_tout = np.random.choice(x_range, size=(S0,S1,nb_iter))  
    
    for k in range(nb_iter):
        # Proposition de candidats
        X_new = X_new_tout[:,:,k]

        for i in range(S0):
            for j in range(S1):
                
                vals = get_vals_voisins(np.array([i,j]),X_courant)
                vois = Vois[i,j,:]
                #vals_u = Vals_u[i,j,:]

                if k == 0:
                    proba_courant[i,j] = calc_proba_gibbs_xy(X_courant[i,j], vals, Y[i,j],sig, vois, type_psi)  

                proba_new = calc_proba_gibbs_xy(X_new[i,j], vals, Y[i,j],sig, vois, type_psi)


                q = proba_new / proba_courant[i,j]
                if q > 1:
                    X_courant[i,j] = X_new[i,j]
                    proba_courant[i,j] = proba_new
                    num_ecart[k] +=1


        X_res[:,:,k] = X_courant        
                                  
                    
    return X_courant, X_res,num_ecart 
def gen_xu_gibbs(X_init,U_init,S0,S1,nb_iter, x_range,u_range,Y,sig,type_psi='kro'):
    """ Generation d'un champs de Gibbs a partir d'une initialisation.
    Genere x selon p(X, U | Y), Y etant l'observation 
    """

    proba_courant = np.zeros_like(X_init)
    num_ecart = np.zeros(shape=(nb_iter))
    
    X_courant = np.copy(X_init)
    X_res = np.zeros(shape=(S0,S1,nb_iter))
    
    U_courant = np.copy(U_init)  
    U_res = np.zeros(shape=(S0,S1,nb_iter))
    
    
    # Retrieving locals neighborhoods:
    Vois = np.zeros(shape=(S0,S1,8))
    #Vals_u = np.zeros_like(Vois)
    Beta_courant = np.zeros_like(Vois)
    Beta_possible = np.zeros(shape=(S0,S1,8,u_range.size))
    for i in range(S0):
        for j in range(S1):
            Vois[i,j,:] = get_num_voisins(np.array([i,j]),X_init)
                 
            for u_inst in range(u_range.size):
                Beta_possible[i,j,:,u_inst] = gen_beta(Vois[i,j,:],u_range[u_inst]) 
            
    
    #energie_totale = np.sqrt((X_init**2).sum())
    
    # Generating all candidates:
    X_new_tout = np.random.choice(x_range, size=(S0,S1,nb_iter))  
    U_new_tout = np.random.choice(u_range, size=(S0,S1,nb_iter))  
    
    for k in range(nb_iter):
        # Proposition de candidats
        X_new = X_new_tout[:,:,k]
        U_new = U_new_tout[:,:,k]

        for i in range(S0):
            for j in range(S1):
                
                vals = get_vals_voisins(np.array([i,j]),X_courant)
                beta =  gen_beta(Vois[i,j],U_courant[i,j])
                vals_u = get_vals_voisins(np.array([i,j]),U_courant)                
                
                
                vois = Vois[i,j,:]
                #vals_u = Vals_u[i,j,:]

                if k == 0:
                    Beta_courant[i,j,:] = Beta_possible[i,j,:,np.where(u_range==U_courant[i,j])]
                    proba_courant[i,j] = calc_proba_gibbs_xyu(X_courant[i,j], vals, Y[i,j],sig,U_courant[i,j], vals_u, vois,beta, type_psi)

                Beta_new = Beta_possible[i,j,:,np.where(u_range==U_new[i,j])]
                proba_new = calc_proba_gibbs_xyu(X_new[i,j], vals,Y[i,j],sig, U_new[i,j], vals_u, vois, Beta_new, type_psi)




                q = proba_new / proba_courant[i,j]
                if q > 1:
                    X_courant[i,j] = X_new[i,j]
                    U_courant[i,j] = U_new[i,j]
                    Beta_courant[i,j,:] = Beta_new
                    proba_courant[i,j] = proba_new
                    num_ecart[k] +=1



        X_res[:,:,k] = X_courant 
        U_res[:,:,k] = U_courant    
                                  
                    
    return X_courant, X_res, U_courant, U_res, num_ecart 
# parametres utiles:
S0 = 30
S1 = 30

val_min = 0
val_max = 1


######################### Generation observations
#%%
#Initialisation
#
#
##%%
#nb_iter_gen = 5
#
#
##%% Choix deterministe de U
#
#U = np.ones(shape=(S0,S1)).astype(float)
#u_range = np.array([np.pi/4, 3*np.pi/4])
#U[:S0/2,:] *= u_range[0] ; 
#U[S0/2:,:] *= u_range[1] ; 
#
#
##%% Generation de X conditionnellement a U
#start = time.time()
#x_range = np.array([0,1])
#init = np.random.choice(x_range, size=(S0,S1))
#X, X_tous = gen_champs_gibbs(init, S0,S1,nb_iter_gen,val_min,xmax=val_max,U=U,type_psi='kro',isotropic=False)
#temps = time.time()-start ; print 'Generation de la simu :%.2f s'%temps
#
##%% Generation observation y
## Valable dans le cas ou le bruit est independant !
#m = 0
#sig = 0.25
#Y = X + st.norm.rvs(loc=m,scale=sig,size=(S0,S1))

######################## EM simplifie pour l'estimation des parametres
centroid, X_init_flat = cvq.kmeans2(Y.flatten(),x_range.size)
X_km = X_init_flat.reshape((S0,S1))
if (X_km==X).mean() < 0.5:
    X_km = 1 - X_km

X_courant = X_km

Y_manip = np.copy(Y)
for x_inst in x_range:
    
    Y_manip  = Y_manip - x_inst * (X_courant==x_inst)
    
sig_0 = np.std(Y_manip)
print 'Ecart type estime apres pseudo-EM : %.3f (reel : %.3f)'%(sig_0,sig)
########################
######## Une instance de simulation selon p(X,U | Y)
#start = time.time()
#nb_iter = 10
#U_init = np.random.choice(u_range,(S0,S1))
#X_init = np.random.choice(x_range,(S0,S1))
#
#X_cond, X_cond_tous, U_cond, U_cond_tous, num_ecart = gen_xu_gibbs(X_init,U_init,S0,S1,nb_iter, x_range,u_range,Y,sig,type_psi='kro')
#temps = time.time()-start ; print 'Simulation selon p( X,U | Y=y) : %.2f s'%temps

########################
######## MPM :  plusieurs instance de simulation selon p(X,U | Y)
start = time.time()
nb_iter = 30
nb_sim = 101 # Impair pour eviter les cas d'egalite dans le mpm !

X_cond = np.zeros(shape=(S0,S1,nb_sim))
U_cond = np.zeros(shape=(S0,S1,nb_sim))

for sim in range(nb_sim):
    print 'Iteration %.0f...'%sim
    U_init = np.random.choice(u_range,(S0,S1))
    X_init = np.random.choice(x_range,(S0,S1))
    
    X_cond[:,:,sim], X_cond_tous, U_cond[:,:,sim], U_cond_tous, num_ecart = gen_xu_gibbs(X_init,U_init,S0,S1,nb_iter, x_range,u_range,Y,sig,type_psi='kro')
temps = time.time()-start ; print '%.0f simulations selon p(X,U|Y=y) : %.2f s'%(nb_sim,temps)

###### X selon le MPM :
X_mpm = np.zeros(shape=(S0,S1))
proba_post_x = np.zeros(shape=(S0,S1,x_range.size))

for x_num in range(x_range.size):
    proba_post_x[:,:,x_num] = (X_cond==x_range[x_num]).mean(axis=2)

pp_xmax = proba_post_x.max(axis=2)
for x_num in range(x_range.size):
    X_mpm += x_range[x_num] * (proba_post_x[:,:,x_num] == pp_xmax)
    
###### U selon le MPM :
U_mpm = np.zeros(shape=(S0,S1))
proba_post_u = np.zeros(shape=(S0,S1,u_range.size))

for u_num in range(u_range.size):
    proba_post_u[:,:,u_num] = (U_cond==u_range[u_num]).mean(axis=2)

pp_umax = proba_post_u.max(axis=2)
for u_num in range(u_range.size):
    U_mpm += u_range[u_num] * (proba_post_u[:,:,u_num] == pp_umax)

#%% Generation de U conditionnellement a X
#u_range = np.array([np.pi/4, 3*np.pi/4])
##
##start = time.time()
#nb_sim=10
#U_cond = np.zeros(shape=(S0,S1,nb_sim))
#for sim in range(nb_sim):
##    
#    U_init = np.random.choice(u_range,(S0,S1))
#    U_cond[:,:,sim], U_cond_tous,num_ecart = gen_u_gibbs(U_init,S0,S1,10, u_range,X,Y,sig,type_psi='kro')
##    
#temps = time.time()-start ; print temps
#
##%%% Generation de X conditionnellement a Y
#x_range = np.array([0,1])
##
#start = time.time()
#nb_sim=10
#X_cond = np.zeros(shape=(S0,S1,nb_sim))
#for sim in range(nb_sim):
#    
#    X_init = np.random.choice(x_range,(S0,S1))
#    X_cond[:,:,sim], X_cond_tous,num_ecart = gen_x_gibbs(X_init,S0,S1,10, x_range,Y,sig,type_psi='kro2')
#    print sim
#    
#temps = time.time()-start ; print temps
#proba_x0 = (X_cond==x_range[0]).mean(axis=2)
#proba_x1 = (X_cond==x_range[1]).mean(axis=2)
###%%
#proba_u0 = (U_cond==u_range[0]).mean(axis=2)
#proba_u1 = (U_cond==u_range[1]).mean(axis=2)
#%%
# Affichages
#plt.close('all')
nb_li = 2;
nb_col = 3;


fig=plt.figure(figsize=(6*nb_col,6*nb_li))

plt.subplot(nb_li,nb_col,1)
plt.imshow(X.T, interpolation='nearest', origin='lower', cmap=plt.cm.afmhot,vmin=0,vmax=1); #plt.colorbar()
plt.title('Anisotropic - $x$')

plt.subplot(nb_li,nb_col,2)
plt.imshow(U.T, interpolation='nearest', origin='lower', cmap=plt.cm.jet); #plt.colorbar()
plt.title('Anisotropic - $u$')
#plt.colorbar()
#plt.quiver(np.cos(angle.T),np.sin(angle.T))

plt.subplot(nb_li,nb_col,3)
plt.imshow(Y.T, interpolation='nearest', origin='lower', cmap=plt.cm.afmhot); #plt.colorbar()
plt.title('Anisotropic - $y$')

plt.subplot(nb_li,nb_col,4)
taux_ok  = (X_mpm == X).mean()
plt.imshow(X_mpm.T, interpolation='nearest', origin='lower', cmap=plt.cm.afmhot,vmin=0,vmax=1); #plt.colorbar()
plt.title('$\hat{x}_{\mathrm{MPM}}$ ; correct %.3f'%taux_ok)


plt.subplot(nb_li,nb_col,5)
taux_ok  = (U_mpm == U).mean()
plt.imshow(U_mpm.T, interpolation='nearest', origin='lower', cmap=plt.cm.jet); #plt.colorbar()
plt.title('$\hat{u}_{\mathrm{MPM}}$ ; correct %.3f'%taux_ok)

plt.tight_layout()


plt.savefig('current_TMC.eps', format='eps',dpi=100)