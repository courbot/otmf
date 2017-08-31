# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 09:45:48 2015

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

import parameters
#import image_tools as it
import gibbs_sampler as gs
import fields_tools as ft
#%%
def plot_taux(im,ref,title):
    
    taux = (im!=ref).mean() * 100
    plt.imshow(im.T,  interpolation='nearest', origin='lower', cmap=plt.cm.bone,); 
    plt.title(title + ' - %.2f'%taux)
    plt.axis('off')

def plot_directions(angle, intensite):
    
    S0 = angle.shape[0]
    S1 = angle.shape[1]
    
    y,x = np.ogrid[0:S0,0:S1]
    

    deb_x = np.tile(x,(S0,1)) - 0.5*np.sin(angle) * intensite
    deb_y = np.tile(y,(1,S1)) - 0.5*np.cos(angle) * intensite
    
    fin_x = np.tile(x,(S0,1)) + 0.5*np.sin(angle) * intensite
    fin_y = np.tile(y,(1,S1)) + 0.5*np.cos(angle) * intensite
    
    
    for i in range(0,S0,4):
        for j in range(0,S1,4):
            #if angle[i,j] != 0:
            plt.plot((deb_x[i,j],fin_x[i,j]), (deb_y[i,j],fin_y[i,j]) ,'k')
    plt.xlim((-0.5,S1-0.5))
    plt.ylim((-0.5,S0-0.5))     
    
#%%
#def get_nom(par):
#    
#    if par.fuzzy==True:
#        deb = 'fuzzy-'
#    else:
#        deb = 'hard-'
#    
#    nom = deb + 'a' + str(par.alpha) +'-' + 'b' + str(par.beta) +'-' + 'd' + str(par.delta) +'-' + 'p' + str(par.phi_theta_0) 
#           
#    # remplacer les '.' par des '_'
#    nom = nom.translate('_', '.')
#    return nom    
#%%      
#%%    
def est_kmeans(Y,x_range):
    """ Simple routine for kmeans on HSI"""
    S0,S1,W = Y.shape
    centroid, X_init_flat = cvq.kmeans2(Y.reshape(S0*S1,W),x_range.size)
    X_km = X_init_flat.reshape((S0,S1))
    if (X_km==x_range[1]).mean() < 0.5:
        X_km = 1 - X_km
    
    return X_km
    
#
#def ICE(Y,x_range,nb_iter_ice):
#    
#    S0,S1,W = Y.shape
#    
#    #nb_iter_ice = 5 #nb : 1st step is initialization
#    nb_simu_ice = 5
#    nb_iter_ice = 10
#    
#    
#    sig_ice = np.zeros(shape=(nb_simu_ice))
#    mu_ice = np.zeros(shape=(nb_simu_ice,W))
#    
#    # First step : initiate parameters with rough estimates. K-means for instance.
#    X_courant = est_kmeans(Y,np.array([0,1]))
#    parchamp = est_param_noise(X_courant,Y,parchamp)
#    pargibbs = par
#    
#    # Then, iterate :
#    for iter_ice in range(1,nb_iter_ice):
#        print 'Iter ICE %.0f'%iter_ice
#        # generating multiple realisation along p(X = x|y)
#        pargibbs.sig = parchamp.sig
#        pargibbs.mu = parchamp.mu
#        
#        sig_simus = np.zeros(shape=(nb_simu_ice))
#        mu_simus = np.zeros(shape=(nb_simu_ice,W))   
#        for simu_ice in range(nb_simu_ice):
#            print 'simu ICE %.0f'%simu_ice
#            # estimating V
#            pargibbs.X = X_courant
#            Vc_xy_tout,energie = gs.gen_champs_v_cond_xy(pargibbs)
#            V_courant = Vc_xy_tout[:,:,-1]
#            
#            # estimating X
#            pargibbs.angle = V_courant    
#            Xc_yv_tout,energie = gs.gen_champs_x_cond_yv(par)
#            X_courant = Xc_yv_tout[:,:,-1]
#        
#            # estimating Theta (parameters)
#            parchamp = est_param_noise(X_courant,Y,parchamp)
#            
#            # retrieving
#            mu_simus[simu_ice,:] = parchamp.mu
#            sig_simus[simu_ice] = parchamp.sig
#        
#        # updating for the next iter
#        parchamp.sig = sig_simus.mean()
#        parchamp.mu=mu_simus.mean(axis=0)
#        
#        # retrieving
#        mu_ice[iter_ice,:] = parchamp.mu
#        sig_ice[iter_ice] = parchamp.sig
#    return sigma_ice,mu_ice,sigma,mu 
#    
#    
  
def SEM(parchamp,pargibbs,nb_iter_sem):
    
    Y = pargibbs.Y
    alpha_sem = np.zeros(shape=(nb_iter_sem))
    
    sig_sem = np.zeros(shape=(nb_iter_sem))
    mu_sem = np.zeros(shape=(nb_iter_sem,W))
    
    # First step : initiate parameters with rough estimates. K-means for instance.
    X_courant = est_kmeans(Y,np.array([0,1]))
    parchamp = est_param_noise(X_courant,Y,parchamp)
    #parchamp.alpha = est_param_champ_bin(X_courant)
    #nb_iter_grad = 30
    pargibbs.sig = parchamp.sig
    pargibbs.mu = parchamp.mu   
    parchamp.alpha = est_param_champ(X_courant)  
    
    #pargibbs = par
    
    for iter_sem in xrange(nb_iter_sem):
        #print 'iter SEM %.0f'%iter_sem
        
        pargibbs.sig = parchamp.sig
        pargibbs.mu = parchamp.mu   
        pargibbs.alpha = parchamp.alpha
        
        parvx = gs.gen_champs_fast(pargibbs, generate_v=True, generate_x=True, use_y=True)
        
        # En vrai il faudra un MPM ici... ??
        X_courant = parvx.X_res[:,:,-21:-1].mean(axis=2)>0.5


        # estimating Theta (parameters)
        parchamp = est_param_noise(X_courant,Y,parchamp)
        
        #nb_iter_grad = 20
        
        # nb. on initialise le gradient stochastique avec la valeur precedente de alpha.
        parchamp.alpha = est_param_champ(X_courant)  
        
        # retrieving
        mu_sem[iter_sem,:] = parchamp.mu
        sig_sem[iter_sem] = parchamp.sig
        alpha_sem[iter_sem] = parchamp.alpha
        print parchamp.alpha
        
    parchamp.mu_sem = mu_sem
    parchamp.sig_sem = sig_sem
    parchamp.alpha_sem=alpha_sem
     
    return parchamp
  
    

def est_param_noise(X,Y,parchamp):
    """ 
    Parameters estimation from complete data
    -> to be a method from parameters.ParamsChamps ??
    """
    
    #mean - vector
    mu = (Y*X[:,:,np.newaxis]).sum(axis=(0,1))/X.sum()

    # standard deviation - real
    Y_manip = np.copy(Y)
    for x_inst in (0,1):
        
        Y_manip  = Y_manip - x_inst * mu[np.newaxis,np.newaxis,:] *  (X[:,:,np.newaxis]==x_inst)
        
    sig = np.std(Y_manip)    

    # next : rho - correlations !

    
    parchamp.mu = mu
    parchamp.sig = sig
    
    # Critere d'arret : variation de l'energie inferierure a x% de l'energie
    
    return parchamp
##%%
def est_param_champ(X):
    
        vals_vois = it.get_vals_voisins_tout(X)    
                
        iseq = (X[:,:,np.newaxis]== vals_vois)
        iseq = iseq[1:-1,1:-1,:]
        
        iseq_5 = (iseq.sum(axis=2)==5).mean() # exp(-2 alpha) /C
        isneq_5 = (iseq.sum(axis=2)==3).mean() # exp(2 alpha) / C
        
        p_iseq = iseq_5/(iseq_5 + isneq_5)
        p_isneq = isneq_5/(iseq_5 + isneq_5)
        
        
        alpha = -0.25 * np.log(p_isneq/p_iseq)
        
        return alpha

#%%    
def serie_simu(pargibbs,nb_rea):
    
    X_rea = np.zeros(shape=(pargibbs.S0,pargibbs.S1,nb_rea))
    V_rea = np.zeros(shape=(pargibbs.S0,pargibbs.S1,nb_rea))
    E_rea = np.zeros(shape=(pargibbs.S0,pargibbs.S1,nb_rea))
    
      
    generate_v = True
    generate_x = True
    use_y = True
    
    pool = mp.Pool(processes=7)
    results = [pool.apply_async(gs.gen_champs_fast,args=(pargibbs,generate_v,generate_x,use_y)) for i in range(nb_rea)]
    output = [p.get() for p in results]
    
    for i in range(nb_rea):
        
        X_rea[:,:,i] = output[i].X_res[:,:,-1]
        V_rea[:,:,i] = output[i].V_res[:,:,-1]
        E_rea[:,:,i] = output[i].energie[:,:,-1]
        
        
    return X_rea,V_rea,E_rea


def MPM(pargibbs,nb_rea_mpm):
    """Parralel MPM algo"""
    #1) built numerous simumations
 
    
    X_mpm = pargibbs.X_res
    V_mpm = pargibbs.V_res
    #E_mpm = pargibbs.energie[:,:,100:]
    
    #X_mpm,V_mpm,E_mpm = serie_simu(pargibbs,nb_rea_mpm)    
    
    # 2)  Estimate frequencies
    freqs = np.zeros(shape=(pargibbs.S0,pargibbs.S1,2*pargibbs.v_range.size))
    for x in (0,1):
        for v in range(v_range.size):
            freqs[:,:,x*v_range.size+v] = ((X_mpm[:,:,50:]==x)*(V_mpm[:,:,50:]==v_range[v])).sum(axis=2)
    
    # 3) get the most frequent mode
    mode = np.argmax(freqs,axis=2)
    
    X_mpm_est = mode/v_range.size
    V_mpm_est = v_range[mode%v_range.size]
    
    return X_mpm_est,V_mpm_est,X_mpm,V_mpm

#%%
def MAP(pargibbs,nb_rea_map):
    

#    pargibbs.nb_iter = nb_rea_map
#    pargibbs.autoconv==False
#    
#    pargibbs = gs.gen_champs_fast(pargibbs,generate_v=True,generate_x=True,use_y=True)    
    
    X_rea = pargibbs.X_res
    V_rea = pargibbs.V_res
    E_rea = pargibbs.energie
    
    E_mean = E_rea.mean(axis=(0,1))
    ind_map = np.argmin(E_mean[10:])
    
    X_map_est = X_rea[:,:,ind_map+10]
    V_map_est = V_rea[:,:,ind_map+10]

    return X_map_est,V_map_est,X_rea,V_rea


def seg_hmc_mpm(pargibbs):
    
    nb_rea_mpm = 100
    start=time.time()
    pargibbs.nb_iter = nb_rea_mpm
    pargibbs.phi_theta_0 = 1 # c'est ca qui fait le H mc
    pargibbs.autoconv==False
    pargibbs = gs.gen_champs_fast(pargibbs,generate_v=True,generate_x=True,use_y=True)  
    temps =  (time.time()-start) 
    print 'Serie simu : %.0f iterations et %.2f s - %.3f s/iter.'%(nb_rea_mpm, temps,temps/nb_rea_mpm  )
    
    
    start=time.time()
    X_mpm_est,V_mpm_est,X_mpm,V_mpm = MPM(pargibbs,nb_rea_mpm)
    temps =  (time.time()-start)    
    
    return X_mpm_est
    
    
    
    
    
#%%        
# parametres utiles:
S0 = 80
S1 = 80

###################
#%%
#v_range = np.array([np.pi/4, np.pi/2, 3*np.pi/4,np.pi])
v_range = np.array([np.pi/4,3*np.pi/4])
#v_range= np.array([np.pi/3,2*np.pi/3])
V = np.zeros(shape=(S0,S1))
V[:S0/2,:] = v_range[0]
V[S0/2:,:] = v_range[1]

V = v_range[1]
dat = np.load('./data/square.npz')
X=dat['X']
V=dat['V']
#X = imread('./data/be.bmp')>0
#V = np.ones_like(X)*v_range[1]
#%%
###################

##%%
#
print('---------------------------------------')
pargibbs = parameters.ParamsGibbs(S0 = S0,
                             S1 = S1,
                             type_pot = 'potts',
                             phi_uni = 0.,
                             thr_conv=0.005,
                             nb_iter=100,
                             fuzzy=False,
                             anisotropic=True,   
                             angle=V,
                             beta = 1.,
                             phi_theta_0 = 1.,
                             alpha = 1.1,
                             alpha_v = 1.,
                             delta = 0.,
                             init_method = 'std',
                             nb_fuzzy = 256. ,
                             v_range = v_range
                             )# beta=1.25,
start=time.time()  
pargibbs.V=V
pargibbs.nb_iter = 50
pargibbs.X = X
parvx = gs.gen_champs_fast(pargibbs, generate_v=False, generate_x=True, use_y=False)
temps =  (time.time()-start)
print 'Champ de Gibbs rapide : %.0f iterations et %.2f s - %.3f s/iter.'%(parvx.nb_iter_conv, temps,temps/(parvx.nb_iter_conv)  )
      
X = pargibbs.X_res[:,:,-1]
#V = pargibbs.V_res[:,:,-1]
#    
                             
##%%                             
pargibbs.V = V
#pargibbs.X = X        
##  
#pargibbs.nb_iter = 50
#start=time.time()                
#pargibbs = gs.gen_champs_fast(pargibbs,generate_v=False,generate_x=True,use_y=False)
#X = pargibbs.X_res[:,:,-1]#80:].mean(axis=2)>0.5
##V = pargibbs.V_res[:,:,-1]
#temps =  (time.time()-start)
#print 'Simu Gibbs : %.0f iterations et %.2f s - %.3f s/iter.'%(pargibbs.nb_iter_conv, temps,temps/(pargibbs.nb_iter_conv)  )

#%%

#%% Generation observation

#Parametres pour Y
W = 10
    
###%% Creation de l'observation, hyperspectrale
lyman_line = np.zeros(shape=W)
lyman_line[4] = 0.3 ; lyman_line[5] = 1 ; lyman_line[6] = 0.5 
##
###%% Generation observation y
### Valable dans le cas ou le bruit est independant !
m = 0
sig = 0.75

Y = X[:,:,np.newaxis]*lyman_line[np.newaxis,np.newaxis,:] + st.norm.rvs(loc=m,scale=sig,size=(S0,S1,W))

pargibbs.Y = Y
#pargibbs.mu = lyman_line
#pargibbs.sig = sig

#----------------------------#


#%% noise params estimation

parchamp = parameters.ParamsChamps()


#%% SEM algo
#start=time.time()
#
v_help = True
pargibbs.nb_nn_v_help = 1 #nb. dependance assez lourde a ce parametre...
pargibbs.v_help==v_help
nb_iter_sem=10
pargibbs.nb_iter = 50
parsem = SEM(parchamp,pargibbs,nb_iter_sem)

pargibbs.sig=parsem.sig
pargibbs.mu = parsem.mu
pargibbs.alpha = parsem.alpha

temps =  (time.time()-start)
print 'SEM : %.0f iterations et %.2f s - %.3f s/iter.'%(nb_iter_sem, temps,temps/nb_iter_sem  )
  
##----------------------------#
## SEM Error
#  

sig_sem=parsem.sig_sem
mu_sem = parsem.mu_sem
alpha_sem = parsem.alpha_sem

residus_mu = mu_sem - lyman_line[np.newaxis,:]
eqm_mu = (residus_mu**2).sum(axis=1)

residus_sig = sig_sem - sig
eqm_sig = residus_sig**2
  
#%% HMC for init
#X_init = seg_hmc_mpm(pargibbs)
#pargibbs.X_init = X_init
#pargibbs.phi_theta_0 = 0. # attention à ca !!!
#
##%% Parralel MPM algo

nb_rea_mpm = 200
start=time.time()
pargibbs.nb_iter = nb_rea_mpm
pargibbs.autoconv==False
pargibbs = gs.gen_champs_fast(pargibbs,generate_v=True,generate_x=True,use_y=True)  
temps =  (time.time()-start) 
print 'Serie simu : %.0f iterations et %.2f s - %.3f s/iter.'%(nb_rea_mpm, temps,temps/nb_rea_mpm  )


start=time.time()
X_mpm_est,V_mpm_est,X_mpm,V_mpm = MPM(pargibbs,nb_rea_mpm)
temps =  (time.time()-start)
#print 'MPM : %.0f iterations et %.2f s - %.3f s/iter.'%(nb_rea_mpm, temps,temps/nb_rea_mpm  )

#
##%% MAP algo

nb_rea_map = 200
start=time.time()
X_map_est,V_map_est,X_map,V_map = MAP(pargibbs,nb_rea_map)
temps =  (time.time()-start)
#print 'MAP : %.0f iterations et %.2f s - %.3f s/iter.'%(nb_rea_map, temps,temps/nb_rea_map  )

#%% Error computation
erreur_mpm_x = np.mean(X_mpm_est!=X)
erreur_mpm_v = np.mean(V_mpm_est!=V)

erreur_map_x = np.mean(X_map_est!=X)
erreur_map_v = np.mean(V_map_est!=V)

#
##
##%% Convergence rate
cv_x_mpm = (X_mpm[:,:,1:] != X_mpm[:,:,:-1]).astype(float).sum(axis=(0,1))/(S0*S1)
cv_v_mpm = (V_mpm[:,:,1:] != V_mpm[:,:,:-1]).astype(float).sum(axis=(0,1))/(S0*S1)

cv_x_map = (X_map[:,:,1:] != X_map[:,:,:-1]).astype(float).sum(axis=(0,1))/(S0*S1)
cv_v_map = (V_map[:,:,1:] != V_map[:,:,:-1]).astype(float).sum(axis=(0,1))/(S0*S1)


#%%
##%%% MAP and MPM results
#plt.close('all')
nb_li = 3
nb_col = 4

plt.figure(figsize=(5*nb_col,5*nb_li))

plt.subplot(nb_li,nb_col,1)
plt.imshow(X.T, interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin=0,vmax=1); 
plt.axis('off')
plt.title('$X$')


plt.subplot(nb_li,nb_col,2*nb_col+1)
plt.imshow(Y.mean(axis=2).T, interpolation='nearest', origin='lower', cmap=plt.cm.bone,); 
plt.axis('off')
plt.title('$Y$ (white)')


plt.subplot(nb_li,nb_col,nb_col+1)
plt.imshow(V.T, interpolation='nearest', origin='lower', cmap=plt.cm.Spectral,vmin=0,vmax=np.pi)
plot_directions(V.T, np.ones_like(V.T))
plt.axis('off')
plt.title('$V$')

##
############### Valeurs moyennes
plt.subplot(nb_li,nb_col,2)
plt.imshow(X_mpm.mean(axis=2).T, interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin=0,vmax=1); 
plt.axis('off')
plt.title('Realisations MPM (moyenne) ')


plt.subplot(nb_li,nb_col,nb_col+2)
plt.imshow(V_mpm.mean(axis=2).T, interpolation='nearest', origin='lower', cmap=plt.cm.Spectral,vmin=0,vmax=np.pi)
plot_directions(V_mpm.mean(axis=2).T, np.ones_like(V.T))
plt.axis('off')
plt.title('Realisations MPM (moyenne) ')


############## Valeurs MPM
plt.subplot(nb_li,nb_col,3)
plt.imshow(X_mpm_est.T, interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin=0,vmax=1); 
plt.axis('off')
plt.title('$\\hat{x}_{MPM}$ - %.2f '%(erreur_mpm_x*100))


plt.subplot(nb_li,nb_col,nb_col+3)
plt.imshow(V_mpm_est.T, interpolation='nearest', origin='lower', cmap=plt.cm.Spectral,vmin=0,vmax=np.pi)
plot_directions(V_mpm_est.T, np.ones_like(V.T))
plt.axis('off')
plt.title('$\\hat{v}_{MPM}$ - %.2f '%(erreur_mpm_v*100))

############## Valeurs MAP
plt.subplot(nb_li,nb_col,4)


plt.imshow(X_map_est.T, interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin=0,vmax=1); 
plot_taux(X_map_est,X,'$\\hat{x}_{MAP}$')
#plt.axis('off')
#plt.title('$\\hat{x}_{MAP}$ - %.2f '%(erreur_map_x*100))


plt.subplot(nb_li,nb_col,nb_col+4)
#plot_taux(V_map_est,V,'$\\hat{v}_{MAP}$')
plt.imshow(V_map_est.T, interpolation='nearest', origin='lower', cmap=plt.cm.Spectral,vmin=0,vmax=np.pi)
plot_directions(V_map_est.T, np.ones_like(V.T))
plt.axis('off')
plt.title('$\\hat{v}_{MAP}$ - %.2f '%(erreur_map_v*100))


plt.subplot(nb_li,nb_col,2*nb_col+2)
plot_taux(X_init,X,'X init (HMC)')

#
#plt.imshow(X_init.T, interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin=0,vmax=1); 
#plt.axis('off')
#plt.title('$X$')

#
#
plt.subplot(nb_li,nb_col,2*nb_col+3)
plt.plot(cv_x_mpm)
plt.plot(cv_v_mpm)

plt.xlabel('$k$ (iter. Gibbs)')
plt.title('Convergences MPM/MAP')
plt.ylim((0,.1))
#
##plt.subplot(nb_li,nb_col,2*nb_col+3)
##plt.plot(cv_x_map)
##plt.plot(cv_v_map)
##plt.ylim((0,.1))
##
##plt.xlabel('$k$ (iter. Gibbs)')
##plt.title('Convergences MAP')
#
#
#plt.tight_layout()
#%%
#plt.savefig('./figures/seg_results.eps', format='eps',dpi=100)


##%% Essais get_val_voisins
#import image_tools as it
#a=it.get_vals_voisins_tout(V_mpm_est)
#plt.figure(figsize=(10,10))
#for i in range(4):
#    for j in range(2):
#        plt.subplot(3,3,j*4+i+1)
#        plt.imshow(a[:,:,j*4+i].T,interpolation='nearest',vmin=0)
#plt.tight_layout()

#%% Affichages EstParams

nb_li = 2
nb_col = 3
plt.figure(figsize=(5*nb_col,5*nb_li))
#sig_sem=parchamp.sig_sem
#mu_sem = parchamps.mu

plt.subplot(nb_li,nb_col,1)
plt.plot(sig_sem)
plt.plot(sig*np.ones(np.size(sig_sem)),':k')
plt.xlabel('$k$ (iter. SEM)')
plt.title('$\\hat{\\sigma}^k$')
plt.ylim((0.9*sig,None))

plt.subplot(nb_li,nb_col,2)
plt.plot(lyman_line-1,':k')
for i in range(nb_iter_sem):
    plt.plot(lyman_line+i,':k')
    plt.plot(mu_sem[i,:]+i)

plt.ylim((-1.1,None))
plt.ylabel('$k$ (iter. SEM)')
plt.xlabel('$\\lambda$')
plt.title('$\\hat{\\mu}^k$')


residus = mu_sem - lyman_line[np.newaxis,:]
eqm = (residus**2).sum(axis=1)

plt.subplot(nb_li,nb_col,3)
plt.plot(eqm)
plt.ylim((0,None))
plt.xlabel('$k$ (iter. SEM)')
plt.title('EQM sur $\\hat{\\mu}^k$')

residus_rel = mu_sem[1:] - mu_sem[:-1]
eqm_rel = (residus_rel**2).sum(axis=1)
plt.subplot(nb_li,nb_col,4)
plt.plot(eqm_rel)
plt.ylim((0,None))
plt.xlabel('$k$ (iter. SEM)')
plt.title('EQM entre $\\hat{\\mu}^k$ et $\\hat{\\mu}^{k+1}$')

plt.subplot(nb_li,nb_col,5)
plt.plot(alpha_sem)
plt.plot(1*np.ones(np.size(alpha_sem)),':k')
plt.xlabel('$k$ (iter. SEM)')
plt.title('$\\hat{\\alpha}^k$')
plt.ylim((0,1.25))

plt.tight_layout()
##
#plt.savefig('./figures/sem_results.eps', format='eps',dpi=100)
#%% Affichages Gibbs
#
#plt.close('all')
#nb_li = 3
#nb_col = 3
#
#plt.figure(figsize=(5*nb_col,5*nb_li))
#
#plt.subplot(nb_li,nb_col,1)
#plt.imshow(X.T, interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin=0,vmax=1); 
#plt.axis('off')
#plt.title('$X$')
#
##plt.savefig('fuzzy_ray_astro.eps', format='eps',dpi=100)
#
#
#plt.subplot(nb_li,nb_col,2)
#plt.imshow(Y.mean(axis=2).T, interpolation='nearest', origin='lower', cmap=plt.cm.bone,); 
#plt.axis('off')
#plt.title('$Y$ (white)')
#
#
#plt.subplot(nb_li,nb_col,3)
#plt.imshow(V.T, interpolation='nearest', origin='lower', cmap=plt.cm.Spectral,vmin=0)
#plot_directions(V.T, np.ones_like(V.T))
#plt.title('$V$')
#
#
############## Initialisations 
#plt.subplot(nb_li,nb_col,4)
#plt.imshow(Xc_y_tout[:,:,0].T, interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin=0,vmax=1); 
#plt.axis('off')
#plt.title('$x \\sim p(X|Y=y)$ (init) ')
#
##
#plt.subplot(nb_li,nb_col,5)
#plt.imshow(Vc_xy_tout[:,:,0].T, interpolation='nearest', origin='lower', cmap=plt.cm.Spectral,vmin=0)
#plot_directions(Vc_xy_tout[:,:,0].T, np.ones_like(V.T))
#plt.title('$v \\sim p(V|Y=y,X=x)$ (init) ')
#
#
#plt.subplot(nb_li,nb_col,6)
#plt.imshow(Xc_yv_tout[:,:,0].T, interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin=0,vmax=1); 
#plt.axis('off')
#plt.title('$x \\sim p(X|Y=y,V=v)$ (init)')
#
#
#
#
#plt.subplot(nb_li,nb_col,7)
#plt.imshow(Xc_y_tout[:,:,-1].T, interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin=0,vmax=1); 
#plt.axis('off')
#taux = (Xc_y_tout[:,:,-1]!=X).mean()
#plt.title('$x \\sim p(X|Y=y)$, erreur %.2f '%(taux*100))
#plt.contour((Xc_y_tout[:,:,-1]!=X).T,1, colors='r')
#
##
#
#plt.subplot(nb_li,nb_col,8)
#taux = (Vc_xy_tout[:,:,-1]!=V).mean()
#plt.imshow(Vc_xy_tout[:,:,-1].T, interpolation='nearest', origin='lower', cmap=plt.cm.Spectral,vmin=0)
#plot_directions(Vc_xy_tout[:,:,-1].T, np.ones_like(V.T))
#plt.title('$v \\sim p(V|Y=y,X=x)$, erreur %.2f '%(taux*100))
#
#
#
#plt.subplot(nb_li,nb_col,9)
#plt.imshow(Xc_yv_tout[:,:,-1].T, interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin=0,vmax=1); 
#plt.axis('off')
#taux = (Xc_yv_tout[:,:,-1]!=X).mean()
#plt.title('$x \\sim p(X|Y=y,V=v)$, erreur %.2f '%(taux*100))
#plt.contour((Xc_yv_tout[:,:,-1]!=X).T,1, colors='r')
#
#
#plt.tight_layout()
#plt.savefig('./figures/gibbs_samplers.eps', format='eps',dpi=100)
#%%
#dx,dy = np.gradient(X_tout[:,:,-1])
#an = np.arctan2(dy,dx)
##an = (an + np.pi/2)%np.pi
##an = an%np.pi
#ux, uy = np.ogrid[0:S0,0:S1]
#ux = np.tile(ux,(1,S1))
#uy = np.tile(uy,(S0,1))
#
## removing incorrect/unknown values
#an[an==0] +=np.nan
##msk=np.zeros_like(an)
##for v in par.v_range:
##    msk += (an==v)
##an[msk==0] += np.nan
#
#an_flat = an.flatten()
#an_interp = np.copy(an)
#
#for i in range(an.shape[0]):
#    for j in range(an.shape[1]):
#        if np.isnan(an[i,j]):
#            dist = np.sqrt((i-ux)**2 + (j-uy)**2)
#            dist_flat = dist.flatten()
#            
#            dist_flat[np.isnan(an_flat)] = 10000
#            ind_min = np.argmin(dist_flat)
#            an_interp[i,j] = an_flat[ind_min]
#an_interp = (an_interp+np.pi/2)%np.pi           
#for i in range(an.shape[0]):
#    for j in range(an.shape[1]):
#        ecart = np.abs(an_interp[i,j] - par.v_range)
#        ind_min = np.argmin(ecart)
#
#        an_interp[i,j] =v_range[ind_min]        
#
##%%
#nb_li = 2
#nb_col = 2
#
#plt.figure(figsize=(5*nb_col,5*nb_li))
#
#plt.subplot(nb_li,nb_col,1)
#plt.imshow(dx.T, interpolation='nearest', origin='lower', cmap=plt.cm.bone,); 
#
#plt.subplot(nb_li,nb_col,2)
#plt.imshow(dy.T, interpolation='nearest', origin='lower', cmap=plt.cm.bone,); 
#
#plt.subplot(nb_li,nb_col,3)
#%%
#plt.figure(figsize=(5,5))
#an = get_dir(Xc_yv_tout[:,:,-1],par)
#
#
#plt.imshow((an.T), interpolation='nearest', origin='lower', cmap=plt.cm.Spectral,vmin=0,)
#plot_directions(an.T, np.ones_like(an.T))
#
#plt.subplot(nb_li,nb_col,4)
#plt.imshow((an_interp.T), interpolation='nearest', origin='lower', cmap=plt.cm.Spectral,)#vmax=np.pi); 
#plot_directions(an_interp.T, np.ones_like(an_interp.T))
######################################
# MPM algo "simple"

#for iter_mpm in xrange(nb_rea_mpm):
#    print 'iter MPM %.0f'%iter_mpm
#    
#    parvx = gs.gen_champs_fast(pargibbs, generate_v=True, generate_x=True, use_y=True)
#    
#    
#    X_mpm[:,:,iter_mpm] = parvx.X_res[:,:,-1]
#    V_mpm[:,:,iter_mpm] = parvx.V_res[:,:,-1]
#
#X_mpm_est = st.mode(X_mpm,axis=2)[0][:,:,0]
#V_mpm_est = st.mode(V_mpm,axis=2)[0][:,:,0]
#
#temps =  (time.time()-start)
#print 'MPM : %.0f iterations et %.2f s - %.3f s/iter.'%(nb_rea_mpm, temps,temps/nb_rea_mpm  )

#%% let us generate instead a dumb case
#
#dx,dy = np.ogrid[0:int(S0/2),0:int(S1/2)]    
#dx = np.tile(dx,(1,int(S1/2)))
#dy = np.tile(dy,(int(S0/2),1))    
#    
#largeur = 3
#X = np.zeros(shape=(S0,S1))
#
#diag1 = (dx > dy-largeur)*(dx < dy+largeur) 
#diag2 = diag1[::-1,:]
#
#X[:int(S0/2),:int(S1/2)] = diag1
#X[int(S0/2):,int(S1/2):] = diag2
#
#X[int(S0/2):,:int(S1/2)] = diag2
#X[:int(S0/2),int(S1/2):] = diag2
#plt.imshow(X)
#V=np.zeros(shape=(S0,S1))
#V[:int(S0/2),:] = v_range[0]
#V[int(S0/2):,:] = v_range[1]
#
#
#dx,dy = np.ogrid[0:int(S0/2),0:int(S1/2)]    
#dx = np.tile(dx,(1,int(S1/2)))
#dy = np.tile(dy,(int(S0/2),1))    
#    
#largeur = 6
#X = np.zeros(shape=(S0,S1))
#
#diag1 = (dx > dy-largeur)*(dx < dy+largeur) 
#diag2 = diag1[::-1,:]
#
#X[:int(S0/2),:int(S1/2)] = diag2
#X[int(S0/2):,int(S1/2):] = diag2
#V[:int(S0/2),:int(S1/2)] = v_range[1]
#V[int(S0/2):,int(S1/2):] = v_range[1]
#
#
#X[int(S0/2):,:int(S1/2)] = diag1
#X[:int(S0/2),int(S1/2):] = diag1
#V[int(S0/2):,:int(S1/2)] = v_range[0]
#V[:int(S0/2),int(S1/2):] = v_range[0]

# Tentative de gradient stochastique.Inutile ?
#%%
#eta = 20./(S0*S1) # plus tard : eta decroisssant
#nb_iter_grad = 30
#alpha_0 =2.
#X_init = X
#
#pargibbs.mu=lyman_line
#pargibbs.sig=0.25
#
#
#alpha_seq = np.zeros(shape=nb_iter_grad)
#E_seq = np.zeros(shape=(S0,S1,nb_iter_grad))
#alpha_seq[0] = alpha_0
#
#Beta = np.ones_like(pargibbs.Vois)    
#for i in xrange(S0):
#    for j in xrange(S1):
#        Beta[i,j,:] =  ft.gen_beta(pargibbs.Vois[i,j],V[i,j],pargibbs.phi_theta_0)
#        
#vals_vois = it.get_vals_voisins_tout(X)  
#iseq0 = (X[:,:,np.newaxis]== vals_vois)
#
#grad_0 = 0.5 * np.sum( (1-2*iseq0) * Beta )    
#
#
##%%
#
#for i in range(1,nb_iter_grad):
#    
#
#    pargibbs.alpha=alpha_seq[i-1]    
#
#
#    pargibbs.nb_iter = 50
#    pargibbs.V = V
#    parvx = gs.gen_champs_fast(pargibbs, generate_v=False, generate_x=True, use_y=True)
#    
#    X_courant=parvx.X_res[:,:,-1]
#    
#    
#    
#    vals_vois = it.get_vals_voisins_tout(X_courant)    
#
#    
#    iseq = (X_courant[:,:,np.newaxis]== vals_vois)
#
#    grad =  0.5 *np.sum( (1.-2. *iseq) * Beta  )    
#
#    
#    alpha_seq[i] = alpha_seq[i-1] + eta * (grad - grad_0)
#    #alpha_seq[i] = np.abs(alpha_seq[i])
#    
#    print  i,alpha_seq[i]
#    
#
#
#    
#plt.figure(figsize=(8,4))
#plt.plot(alpha_seq)
#plt.plot(np.ones_like(alpha_seq),'r')
#plt.ylim((0,None))

#%%
#%% Affichage MPM
#import fields_tools as ft
#vals_vois = it.get_vals_voisins_tout(X)    
#somme_vois = vals_vois.sum(axis=2)
#probas = np.zeros(shape=(2,8+1) )
#x_util = np.copy(X)
#v_util = np.copy(V)
#Vois = np.zeros(shape=(S0,S1,8))
#Beta = np.ones_like(Vois)
#
#for i in xrange(S0):
#    for j in xrange(S1):
#        Vois[i,j,:] = it.get_num_voisins(i,j,np.zeros(shape=(S0,S1)))
#        
#        if np.isnan(v_util[i,j])==0:
#             Beta[i,j,:] =  ft.gen_beta(Vois[i,j],v_util[i,j],0) 

#%%
#
## Estimation du parametre alpha par la methode de Derin et al. :
#vals_vois = it.get_vals_voisins_tout(X)    
#somme_vois = vals_vois.sum(axis=2)
#probas = np.zeros(shape=(2,8+1) )
#x_util = np.copy(X)
#
## changer la boucle pour faire sur (0,1)
#inds_v = np.arange(8+1)
#for v in range(8+1):
#    probas[0,v] = ( (somme_vois == v) * (x_util==0) ).mean()
#    probas[1,v] = ( (somme_vois == v) * (x_util==1) ).mean()
#
#
#numer = np.zeros(shape=(S0,S1))
## changer la boucle pour faire sur (0,1)
#for i in range(S0):
#    for j in range(S1):
#        # prevenir la division par 0 ?
#        #            denom_den = probas[1,np.where(somme_vois[i,j]==inds_v)]
#        #            if denom_den!=0:
#        #                numer_den = probas[0,np.where(somme_vois[i,j]==inds_v)]
#        #                denom[i,j] = numer_den/denom_den
#        #            else:
#        #                denom[i,j] = np.inf
#        #            
#        numer[i,j] = np.log( probas[0,np.where(somme_vois[i,j]==inds_v)]   /  probas[1,np.where(somme_vois[i,j]==inds_v)] )
#
#
#is_0 =  (  (x_util[:,:,np.newaxis]== 0)*(x_util[:,:,np.newaxis]== vals_vois))#.sum(axis=2)
#is_1 =  (  (x_util[:,:,np.newaxis]== 1)*(x_util[:,:,np.newaxis]== vals_vois))#.sum(axis=2)
#
#denom = 2*(is_0-is_1).sum(axis=2)
#
#msk = (np.isinf(numer) + (denom==0)) >0
#msk_new = np.ones_like(msk)
#msk_new[1:-1,1:-1] = msk_new[1:-1,1:-1]
##msk=msk_new
#
#alpha_tous = numer/denom
#alpha_est = alpha_tous[msk==0].mean()  
#
##
#print alpha_est
#%%
#%%% MAP and MPM results
#plt.close('all')
#nb_li = 3
#nb_col = 4
#
#plt.figure(figsize=(5*nb_col,5*nb_li))
#
#plt.subplot(nb_li,nb_col,1)
#plt.imshow(X.T, interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin=0,vmax=1); 
#plt.axis('off')
#plt.title('$X$')
#
#
#plt.subplot(nb_li,nb_col,2*nb_col+1)
#plt.imshow(Y.mean(axis=2).T, interpolation='nearest', origin='lower', cmap=plt.cm.bone,); 
#plt.axis('off')
#plt.title('$Y$ (white)')
#
#
#plt.subplot(nb_li,nb_col,nb_col+1)
#plt.imshow(V.T, interpolation='nearest', origin='lower', cmap=plt.cm.Spectral,vmin=0,vmax=np.pi)
#plot_directions(V.T, np.ones_like(V.T))
#plt.axis('off')
#plt.title('$V$')
#
#
#plt.subplot(nb_li,nb_col,2)
#plt.imshow(Xg[:,:,200:].mean(axis=2).T, interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin=0,vmax=1); 
#plt.axis('off')
#plt.title('$X$ gibbs')
#
#
#
#
#plt.subplot(nb_li,nb_col,nb_col+2)
#plt.imshow(Vg[:,:,200:].mean(axis=2).T, interpolation='nearest', origin='lower', cmap=plt.cm.Spectral,vmin=0,vmax=np.pi)
#plot_directions(Vg[:,:,200:].mean(axis=2).T, np.ones_like(V.T))
#plt.axis('off')
#plt.title('$V$')
#
#
#plt.tight_layout()

#    
#    # changer la boucle pour faire sur (0,1)
#    inds_v = np.arange(8+1)
#    for v in range(8+1):
#        probas[0,v] = ( (somme_vois == v) * (x_util==0) ).mean()
#        probas[1,v] = ( (somme_vois == v) * (x_util==1) ).mean()
#    
#    
#    numer = np.zeros(shape=(S0,S1))
#    # changer la boucle pour faire sur (0,1)
#    for i in range(S0):
#        for j in range(S1):
#            # prevenir la division par 0 ?
#            #            denom_den = probas[1,np.where(somme_vois[i,j]==inds_v)]
#            #            if denom_den!=0:
#            #                numer_den = probas[0,np.where(somme_vois[i,j]==inds_v)]
#            #                denom[i,j] = numer_den/denom_den
#            #            else:
#            #                denom[i,j] = np.inf
#            #            
#            numer[i,j] = np.log( probas[1,np.where(somme_vois[i,j]==inds_v)]   /  probas[0,np.where(somme_vois[i,j]==inds_v)] )
#    
#    
#    is_0 =  (  (x_util[:,:,np.newaxis]== 0)*(x_util[:,:,np.newaxis]== vals_vois))#.sum(axis=2)
#    is_1 =  (  (x_util[:,:,np.newaxis]== 1)*(x_util[:,:,np.newaxis]== vals_vois))#.sum(axis=2)
#        
#    
#    #E_seq[:,:,i] = parvx.energie[:,:,-1]
#    E_seq[:,:,i] = parvx.energie[:,:,50:].mean(axis=2)
#    #delta_alpha = alpha_seq[i] - alpha_seq[i-1]  
#    
#        
#    if i > 1:
#        delta_alpha = alpha_seq[i-1] - alpha_seq[i-2]  
#        
#        delta_E = (E_seq[:,:,i]- E_seq[:,:,i-1]).sum()
#    else:
#        delta_alpha=1.
#        delta_E =  (E_seq[:,:,i-1]).sum()
#    
#    grad_im = delta_E/delta_alpha     
#    #grad = grad_im.sum(axis=(0,1))#/ delta_alpha  
#    #    
#    #grad_0 = (E_seq[:,:,i]- E_seq[:,:,0]).sum()/(alpha_seq[i-1]-alpha_0) #E_0.sum()/delta_alpha

#%% Essai gradient stochastique
#parchamp = parameters.ParamsChamps()
#parchamp.sig=sig
#parchamp.mu=lyman_line
#pargibbs.mu = lyman_line
#pargibbs.sig=sig
#pargibbs.nb_iter = 50
#alpha_test = gradient_stoch(X,Y,pargibbs,parchamp,30,alpha_0=1.5)

#%% Essai estimation methode de Vincent

## Estimation du parametre alpha par la methode de Derin et al. :
#vals_vois = it.get_vals_voisins_tout(X)    
#somme_vois = vals_vois.sum(axis=2)
#probas = np.zeros(shape=(2,8+1) )
#x_util = np.copy(X)
#
#somme_1 = somme_vois * (x_util==1)
#somme_0 = (8-somme_vois) * (x_util==0)
#
#somme_pareil = (somme_1+somme_0).sum()
#
#somme_1b = somme_vois * (x_util==0)
#somme_0b = (8-somme_vois) * (x_util==1)
#
#somme_paspareil = (somme_1b+somme_0b).sum()
#
#print np.log( np.sqrt(somme_pareil/somme_paspareil)  )
#%%
#
## changer la boucle pour faire sur (0,1)
#inds_v = np.arange(8+1)
#for v in range(8+1):
#    probas[0,v] = ( (somme_vois == v) * (x_util==0) ).mean()
#    probas[1,v] = ( (somme_vois == v) * (x_util==1) ).mean()
#
#
#numer = np.zeros(shape=(S0,S1))
## changer la boucle pour faire sur (0,1)
#for i in range(S0):
#    for j in range(S1):
#        # prevenir la division par 0 ?
#        #            denom_den = probas[1,np.where(somme_vois[i,j]==inds_v)]
#        #            if denom_den!=0:
#        #                numer_den = probas[0,np.where(somme_vois[i,j]==inds_v)]
#        #                denom[i,j] = numer_den/denom_den
#        #            else:
#        #                denom[i,j] = np.inf
#        #            
#        numer[i,j] = np.log( probas[1,np.where(somme_vois[i,j]==inds_v)]   /  probas[0,np.where(somme_vois[i,j]==inds_v)] )
#
#
#is_0 =  (  (x_util[:,:,np.newaxis]== 0)*(x_util[:,:,np.newaxis]== vals_vois))#.sum(axis=2)
#is_1 =  (  (x_util[:,:,np.newaxis]== 1)*(x_util[:,:,np.newaxis]== vals_vois))#.sum(axis=2)
#
#denom = 2*(is_1-is_0).sum(axis=2)
#
#msk = (np.isinf(numer) + (denom==0)) >0
#msk_new = np.ones_like(msk)
#msk_new[1:-1,1:-1] = msk_new[1:-1,1:-1]
##msk=msk_new
#
#alpha_tous = numer/denom
#alpha_est = alpha_tous[msk==0].mean()  
#%%
# Essai estimation directe
#
#vals_vois = it.get_vals_voisins_tout(X)    
#        
#iseq = (X[:,:,np.newaxis]== vals_vois)
#iseq = iseq[1:-1,1:-1,:]
#
#iseq_5 = (iseq.sum(axis=2)==5).mean() # exp(-2 alpha) /C
#isneq_5 = (iseq.sum(axis=2)==3).mean() # exp(2 alpha) / C
#
#p_iseq = iseq_5/(iseq_5 + isneq_5)
#p_isneq = isneq_5/(iseq_5 + isneq_5)
#
#
#alpha_1 = -0.25 * np.log(p_isneq/p_iseq)
#
#
#
#iseq_5 = (iseq.sum(axis=2)==6).mean() # exp(-2 alpha) /C
#isneq_5 = (iseq.sum(axis=2)==2).mean() # exp(2 alpha) / C
#
#p_iseq = iseq_5/(iseq_5 + isneq_5)
#p_isneq = isneq_5/(iseq_5 + isneq_5)
#
#
#alpha_2 = -0.125 * np.log(p_isneq/p_iseq)
#
#print alpha_1,alpha_2

#
#grad =  0.5*np.sum( (1.-2. *iseq) * phi_theta)

 
##%%
#def calc_grad(X,phi_theta):
#        vals_vois = it.get_vals_voisins_tout(X)    
#        
#        iseq = (X[:,:,np.newaxis]== vals_vois)
#        iseq = iseq[1:-1,1:-1,:]
#
#        grad =  0.5*np.sum( (1.-2. *iseq) * phi_theta)
#   
#        return grad
#           
#   
#def gradient_stoch(X,Y,pargibbs,parchamps,nb_iter_grad,alpha_0,anisotropic=True):
#    
#    S0 = pargibbs.S0
#    S1 = pargibbs.S1
#    #eta = 1.5/((S0-2)*(S1-2)) # plus tard : eta decroisssant
#    eta = 1./((S0)*(S1))
#    #eta = 30./((S0-2)*(S1-2))  
#    
#    alpha_seq = np.zeros(shape=nb_iter_grad)
#    alpha_seq[0] = alpha_0
#    
#    phi_theta = np.ones_like(pargibbs.Vois)    
#    if anisotropic==True:
#        for i in xrange(S0):    
#            for j in xrange(S1):
#                phi_theta[i,j,:] =  ft.gen_beta(pargibbs.Vois[i,j],V[i,j],pargibbs.phi_theta_0)
#                
#
#    # si c'est x estime :
#    phi_theta = phi_theta[1:-1,1:-1,:]      
#    
#
#    grad_0 = calc_grad(X,phi_theta)
#
#    for i in xrange(1,nb_iter_grad):
#        
#        # Simulation selon p_theta(X)
#        pargibbs.alpha=alpha_seq[i-1] 
#        
#        #grad = 0
#        #for sim in range(10):
#                    
#        # simulation de X sans Y   
#        parvx = gs.gen_champs_fast(pargibbs, generate_v=False, generate_x=True, use_y=False)
#        X_1=parvx.X_res[:,:,-1]
#
#        # Calcul du gradient :
#        grad_1 = calc_grad(X_1,phi_theta)
#
#
#        # simulation de X conditionellement a Y:
#
##        parvx = gs.gen_champs_fast(pargibbs, generate_v=False, generate_x=True, use_y=True)
##        X_2=parvx.X_res[:,:,-1]
##
##        # Calcul du gradient :
##        grad_2 = calc_grad(X_2,phi_theta)
#
#
##        # Mise à jour de alpha
##        if i <=10:
##            alpha_seq[i] = alpha_seq[i-1] + eta * (grad_1 - grad_2)
##            
##        else :
##            alpha_seq[i] = alpha_seq[i-1] + eta/(i-10) * (grad_1 - grad_2)#/grad_0
#                
#        alpha_seq[i] = alpha_seq[i-1] + eta * (grad_1 - grad_0)#/grad_0
#        #alpha_seq[i] = np.abs(alpha_seq[i])
#        print alpha_seq[i]
#        
#
#    fin_iter = int(0.75*nb_iter_grad)
#    alpha_est = alpha_seq[fin_iter:].mean()
#     
#    return alpha_seq
    
#%%    
    #def est_param_champ_bin(X):
#
#    # Estimation du parametre alpha par la methode de Derin et al. :
#    vals_vois = it.get_vals_voisins_tout(X)    
#    somme_vois = vals_vois.sum(axis=2)
#    probas = np.zeros(shape=(2,8+1) )
#    x_util = np.copy(X)
#    
#    # changer la boucle pour faire sur (0,1)
#    inds_v = np.arange(8+1)
#    for v in range(8+1):
#        probas[0,v] = ( (somme_vois == v) * (x_util==0) ).mean()
#        probas[1,v] = ( (somme_vois == v) * (x_util==1) ).mean()
#    
#    
#    numer = np.zeros(shape=(S0,S1))
#    # changer la boucle pour faire sur (0,1)
#    for i in range(S0):
#        for j in range(S1):
#            # prevenir la division par 0 ?
#            #            denom_den = probas[1,np.where(somme_vois[i,j]==inds_v)]
#            #            if denom_den!=0:
#            #                numer_den = probas[0,np.where(somme_vois[i,j]==inds_v)]
#            #                denom[i,j] = numer_den/denom_den
#            #            else:
#            #                denom[i,j] = np.inf
#            #            
#            numer[i,j] = np.log( probas[1,np.where(somme_vois[i,j]==inds_v)]   /  probas[0,np.where(somme_vois[i,j]==inds_v)] )
#    
#    
#    is_0 =  (  (x_util[:,:,np.newaxis]== 0)*(x_util[:,:,np.newaxis]== vals_vois))#.sum(axis=2)
#    is_1 =  (  (x_util[:,:,np.newaxis]== 1)*(x_util[:,:,np.newaxis]== vals_vois))#.sum(axis=2)
#    
#    denom = 2*(is_1-is_0).sum(axis=2)
#    
#    msk = (np.isinf(numer) + (denom==0)) >0
#    msk_new = np.ones_like(msk)
#    msk_new[1:-1,1:-1] = msk_new[1:-1,1:-1]
#    #msk=msk_new
#    
#    alpha_tous = numer/denom
#    alpha_est = alpha_tous[msk==0].mean()  
#
#
#    
#    return alpha_est
