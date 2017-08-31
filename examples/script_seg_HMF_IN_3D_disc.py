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
sys.path.insert(0,'../Champs')
import MF_tools as mft

import matplotlib.pyplot as plt
import time
import scipy.stats as st

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
  
        
def calc_proba_gibbs_x(x, vals, vois, type_psi):
    
        psi_x = psi( x, vals,type_psi)
              
        energie = psi_x * (vois >=0)
        
        nb_elt = (vois >=0).sum()
        cst_norm = np.exp(-nb_elt)
        proba = np.exp(-(energie.sum()))/cst_norm 
        
        return proba
        
def calc_energie_gibbs_xy(x, vals,x_range,likelihood_yij, vois, type_psi):
    
#
#       nb_elt = (vois >=0).sum()


        psi_x = psi( x, vals,type_psi)
#
#        
#        #likelihood =  np.exp(-( 0.5 * sig**2 *np.linalg.norm(y-lyman_line)  ))
#       
        #likelihood = np.exp(- np.linalg.norm(y-mu)**2 /(2*sig**2) )#st.multivariate_normal.pdf(y, mean=mu,cov=sig)   #juste mais pose des pb numeriques.                  
        likelihood = likelihood_yij[np.where(x==x_range)]
#        energie = (-psi_x + np.log(likelihood) ) * (vois >=0)
        energie_x = -psi_x.sum() # somme sur les cliques
        energie_y = np.log(likelihood) #- lyman_line.size
        
        #proba = np.exp(energie_y  + energie_x)
        proba = energie_y  + energie_x
        
     
        
        return proba  
        
def gen_x_gibbs(X_init,S0,S1,nb_iter, x_range,type_psi='abs'):
    """ Generation d'un champs de Gibbs a partir d'une initialisation.
    Genere x selon p(X). Distribution de Gibbs / Markov.
    """
    X_courant = np.copy(X_init)
    proba_courant = np.zeros_like(X_init).astype(float)
    num_ecart = np.zeros(shape=(nb_iter))
    X_res = np.zeros(shape=(S0,S1,nb_iter))
    
    # Retrieving locals neighborhoods:
    Vois = np.zeros(shape=(S0,S1,8))

    for i in range(S0):
        for j in range(S1):
            Vois[i,j,:] = mft.get_num_voisins(np.array([i,j]),X_init)
            
    
    #energie_totale = np.sqrt((X_init**2).sum())
    
    # Generating all candidates:
    X_new_tout = np.random.choice(x_range, size=(S0,S1,nb_iter))
    
    for k in range(nb_iter):
        # Proposition de candidats
        X_new = X_new_tout[:,:,k]

        for i in range(S0):
            for j in range(S1):
                
                vals = mft.get_vals_voisins(np.array([i,j]),X_courant)
                vois = Vois[i,j,:]


                if k == 0:
                    proba_courant[i,j] = calc_proba_gibbs_x(X_courant[i,j], vals, vois, type_psi)

                proba_new = calc_proba_gibbs_x(X_new[i,j], vals, vois,type_psi)


                q = proba_new / proba_courant[i,j]
                if q > 1:
                    X_courant[i,j] = X_new[i,j]
                    proba_courant[i,j] = proba_new
                    num_ecart[k] +=1


        X_res[:,:,k] = X_courant        
                                  
                    
    return X_courant, X_res
    
 

def gen_x_gibbs_cond(X_init,S0,S1,nb_iter, x_range,Y,sig,mu,type_psi='abs',isotropic=True):
    """ Generation d'un champs de Gibbs a partir d'une initialisation.
    Genere x selon p(X | Y), Y etant l'observation 
    """
    X_courant = np.copy(X_init)
    energie_courant = np.zeros_like(X_init)
    num_ecart = np.zeros(shape=(nb_iter))
    X_res = np.zeros(shape=(S0,S1,nb_iter))
    
    # Y posterior distribution (likelihood) for each  class : 
    likelihood_y = np.zeros(shape=(S0,S1,x_range.size))
    
    likelihood_y[:,:,0] = st.multivariate_normal.pdf(Y, mean=np.zeros_like(mu),cov=sig)     
    likelihood_y[:,:,1] = st.multivariate_normal.pdf(Y, mean=mu,cov=sig) 
   
    #likelihood_y /= np.exp(-lyman_line.size/2)#cstnorm#st.multivariate_normal.pdf(np.zeros_like(lyman_line), mean=np.zeros_like(lyman_line),cov=sig)

    # Retrieving locals neighborhoods:
    Vois = np.zeros(shape=(S0,S1,8))

    for i in range(S0):
        for j in range(S1):
            Vois[i,j,:] = mft.get_num_voisins(np.array([i,j]),X_init)

    # Generating all candidates:
    X_new_tout = np.random.choice(x_range, size=(S0,S1,nb_iter))  
    
    for k in range(nb_iter):
        # Proposition de candidats
        X_new = X_new_tout[:,:,k]

        for i in range(S0):
            for j in range(S1):
                
                vals = mft.get_vals_voisins(np.array([i,j]),X_courant)
                vois = Vois[i,j,:]


                if k == 0:
                    energie_courant[i,j] = calc_energie_gibbs_xy(X_courant[i,j], vals,x_range, likelihood_y[i,j,:], vois, type_psi)  
                

                    
                energie_new = calc_energie_gibbs_xy(X_new[i,j], vals, x_range, likelihood_y[i,j,:], vois, type_psi)  #proba_cond_y#proba_gibbs_x + proba_cond_y#calc_proba_gibbs_xy(X_new[i,j], vals, Y[i,j,:],sig,lyman_line, vois, type_psi)

                # Calcul direct du ratio des probas ?
                #q = proba_new / proba_courant[i,j]
                if energie_new > energie_courant[i,j]: # q > 1
                    X_courant[i,j] = X_new[i,j]
                    #print proba_new
                    energie_courant[i,j] = energie_new
                    num_ecart[k] +=1


        X_res[:,:,k] = X_courant        
                                  
                    
    return X_courant, X_res,num_ecart 
    
def simu_gibbs(sig,mu,S0,S1,nb_sim_gibbs,nb_iter, x_range,Y):
    """ 
    Genere *nb_sim_gibbs* realisation selon p(X=x|Y).
    """
    X_cond = np.zeros(shape=(S0,S1,nb_sim_gibbs))
    # nb. les boucles sont independantes. Donc on peut parraleliser !
    for sim in range(nb_sim_gibbs):
        #print 'Iteration %.0f...'%sim
    
        X_init = np.random.choice(x_range,(S0,S1))
        
        X_cond[:,:,sim], X_cond_tous, num_ecart = gen_x_gibbs_cond(X_init,S0,S1,nb_iter, x_range,Y,sig,mu,type_psi='kro')
        
    return X_cond
    
def calc_mpm(X_cond,S0,S1,x_range):
    """
    Maximum Posterior Mode from a realisation sample (e.g Gibbs sample).
    """
    X_mpm = np.zeros(shape=(S0,S1))
    proba_post_x = np.zeros(shape=(S0,S1,x_range.size))
    
    for x_num in range(x_range.size):
        proba_post_x[:,:,x_num] = (X_cond==x_range[x_num]).mean(axis=2)
    
    pp_xmax = proba_post_x.max(axis=2)
    for x_num in range(x_range.size):
        X_mpm += x_range[x_num] * (proba_post_x[:,:,x_num] == pp_xmax)
    
    return X_mpm

def est_kmeans(Y,x_range):
    """ Simple routine for kmeans on HSI"""
    S0,S1,W = Y.shape
    centroid, X_init_flat = cvq.kmeans2(Y.reshape(S0*S1,W),x_range.size)
    X_km = X_init_flat.reshape((S0,S1))
    if (X_km==x_range[1]).mean() < 0.5:
        X_km = 1 - X_km
    
    return X_km
    
def est_param_complete(X,Y):

    #mean - vector
    mu = (Y*X[:,:,np.newaxis]).sum(axis=(0,1))/X.sum()

    # standard deviation - real
    Y_manip = np.copy(Y)
    for x_inst in x_range:
        
        Y_manip  = Y_manip - x_inst * mu[np.newaxis,np.newaxis,:] *  (X[:,:,np.newaxis]==x_inst)
        

    sig = np.std(Y_manip)    
    
    return mu,sig
    
def est_param_ice(Y,x_range,nb_iter_ice):

    S0,S1,W = Y.shape
    
    #nb_iter_ice = 5 #nb : 1st step is initialization
    nb_simu_ice = 5
    nb_iter_gibbs = 10
    
    sigma = np.zeros(shape=(nb_iter_ice))
    mu = np.zeros(shape=(nb_iter_ice,W))
    
    # First step : initiate parameters with rough estimates. K-means for instance.
    X_km = est_kmeans(Y)
    mu[0,:],sigma[0] = est_param_complete(X_km,Y)
    
    # Then, iterate :
    for iter_ice in range(1,nb_iter_ice):
        print 'Iter ICE %.0f'%iter_ice
        # generating multiple realisation along p(X = x|y)
        X_cond = simu_gibbs(sigma[iter_ice-1],mu[iter_ice-1,:],S0,S1,nb_simu_ice,nb_iter_gibbs, x_range,Y)
        
        # Estimating parameters on each realization:
        sigma_courant = np.zeros(shape=nb_simu_ice)
        mu_courant = np.zeros(shape=(nb_simu_ice,W))
        
        for simu_ice in range(nb_simu_ice):
            mu_courant[simu_ice,:], sigma_courant[simu_ice]= est_param_complete(X_cond[:,:,simu_ice],Y)
    
        # Uploading parameters for next iteration:
        sigma[iter_ice] = sigma_courant.mean()
        mu[iter_ice,:] = mu_courant.mean(axis=0)
            
    sigma_ice = sigma[-1]
    mu_ice = mu[-1,:]   
    return sigma_ice,mu_ice,sigma,mu
    
# parametres utiles:
S0 = 50
S1 = 50
W = 30


######################### Generation observations
#%%
#Initialisation


#%%
nb_iter_gen = 20


#%% Generation de X 
#start = time.time()
#x_range = np.array([0,1])
#init = np.random.choice(x_range, size=(S0,S1))
#
#X, X_tous = gen_x_gibbs(init, S0,S1,nb_iter_gen,x_range,type_psi='kro')
#temps = time.time()-start ; print 'Generation de la simu : %.2f s'%temps
#
#
###%% Creation de l'observation, hyperspectrale
#lyman_line = np.zeros(shape=W)
#lyman_line[9] = 0.3 ; lyman_line[10] = 1 ; lyman_line[11] = 0.5 
##
##%% Generation observation y
## Valable dans le cas ou le bruit est independant !
m = 0
sig = 0.25
Y = X[:,:,np.newaxis]*lyman_line[np.newaxis,np.newaxis,:] + st.norm.rvs(loc=m,scale=sig,size=(S0,S1,W))


################### ################### ################### ################### 
################### Parameter(s) estimation
################# Algo : ICE (See Pieczynski et al)
start = time.time()
nb_iter_ice = 20
sigma_ice,mu_ice,sigma,mu = est_param_ice(Y,x_range,nb_iter_ice)

temps = time.time()-start ; print 'Estimation ICE : %.0f iterations en %.2f s'%(nb_iter_ice-1,temps)


################### ################### ################### ###################
######## Gibbs :  plusieurs instance de simulation selon p(X | Y) sachant theta

start = time.time()
nb_iter_gibbs = 15
nb_sim_gibbs = 5 # Impair pour eviter les cas d'egalite dans le mpm !

X_cond = simu_gibbs(sigma_ice,mu_ice,S0,S1,nb_sim_gibbs,nb_iter_gibbs, x_range,Y)

temps = time.time()-start ; print '%.0f simulations selon p(X|Y=y) : %.2f s'%(nb_sim_gibbs,temps)

####### X selon le MPM :
X_mpm = calc_mpm(X_cond,S0,S1,x_range)
#%%
# Affichages
plt.close('all')
nb_li = 2;
nb_col = 3;


fig=plt.figure(figsize=(6*nb_col,6*nb_li))

plt.subplot(nb_li,nb_col,1)
plt.imshow(X.T, interpolation='nearest', origin='lower', cmap=plt.cm.afmhot,vmin=0,vmax=1); #plt.colorbar()
plt.title('Isotropic - $x$')

#plt.colorbar()
#plt.quiver(np.cos(angle.T),np.sin(angle.T))

plt.subplot(nb_li,nb_col,2)
plt.imshow(Y.mean(axis=2).T, interpolation='nearest', origin='lower', cmap=plt.cm.afmhot); #plt.colorbar()
plt.title('Isotropic - $y$')
#
plt.subplot(nb_li,nb_col,3)
plt.imshow(Y[:,:,10].T, interpolation='nearest', origin='lower', cmap=plt.cm.afmhot); #plt.colorbar()
plt.title('Isotropic - $y_{\\lambda=10}$')

plt.subplot(nb_li,nb_col,4)
taux_ok  = (X_mpm == X).mean()
print 'MPM correct a %.3f'%taux_ok
plt.imshow(X_mpm.T, interpolation='nearest', origin='lower', cmap=plt.cm.afmhot,vmin=0,vmax=1); #plt.colorbar()
plt.contour(np.abs(X_mpm-X).T,1,colors='r')
plt.title('$\hat{x}_{\mathrm{MPM}}$ ; correct %.3f'%taux_ok)


plt.subplot(nb_li,nb_col,6)

plt.plot(range(sigma.size),sigma,'.--',label='estimations ICE')
plt.plot(range(sigma.size),sig*np.ones_like(sigma),':k',label='Valeur reelle')
plt.legend(loc='lower right')

plt.ylabel('Iter. ICE')
plt.title('$\hat{\\sigma}_{\mathrm{ICE}}$')

plt.subplot(nb_li,nb_col,5)
eqm = np.linalg.norm(mu - lyman_line[np.newaxis,:],axis=1)
plt.plot(eqm)

#for i in range(mu.shape[0]):
#    plt.plot(i+mu[i,:],'--')
#    plt.plot(i+lyman_line,':k')

plt.ylabel('Iter. ICE')
plt.title('EQM sur $\hat{\\mu}_{\mathrm{ICE}}$')


plt.tight_layout()


plt.savefig('current_HMF.eps', format='eps',dpi=100)