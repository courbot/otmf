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
import seg_OTMF as sot
#%%
def plot_taux(im,ref,title):
    
    taux = (im!=ref).mean() * 100
    plt.imshow(im.T,  interpolation='nearest', origin='lower', cmap=plt.cm.bone,); 
    plt.title(title + ' - %.2f'%taux)
    plt.axis('off')
    
def plotref(dat,ref,titre):

    plt.plot(dat,'b')
    plt.plot(ref*np.ones(np.size(sig_sg)),':r')
    plt.xlabel('$k$ (iter. SG)')
    plt.ylim((0.95*min(ref,min(dat)),1.05*max(ref,max(dat))))
    plt.title(titre)
    
#%%
def plot_directions(angle, intensite,pas):
    
    S0 = angle.shape[0]
    S1 = angle.shape[1]
    
    y,x = np.ogrid[0:S0,0:S1]
    

    deb_x = np.tile(x,(S0,1)) - 0.5*np.sin(angle) * intensite
    deb_y = np.tile(y,(1,S1)) - 0.5*np.cos(angle) * intensite
    
    fin_x = np.tile(x,(S0,1)) + 0.5*np.sin(angle) * intensite
    fin_y = np.tile(y,(1,S1)) + 0.5*np.cos(angle) * intensite
    
    
    for i in range(int(pas/2.),S0,pas):
        for j in range(int(pas/2.),S1,pas):
            if angle[i,j] != 0:
                plt.plot((deb_x[i,j],fin_x[i,j]), (deb_y[i,j],fin_y[i,j]) ,'k')
    plt.xlim((-0.5,S1-0.5))
    plt.ylim((-0.5,S0-0.5))     
#%%     
    
#%%
    
#def est_param_de(X,V,pargibbs,cond):
#
#    S0 = pargibbs.S0
#    S1 = pargibbs.S1   
#    V = pargibbs.V
#    
#    if cond==True:
#        phi_theta = np.ones_like(pargibbs.Vois)    
#        for i in xrange(S0):    
#            for j in xrange(S1):
#                phi_theta[i,j,:] =  ft.gen_beta(pargibbs.Vois[i,j,:],V[i,j],pargibbs.phi_theta_0)   
#                
#        phi_theta = phi_theta[1:-1,1:-1,:]
#        
#    vals_vois = it.get_vals_voisins_tout(X)   
#    iseq = (X[:,:,np.newaxis]== vals_vois)
#    iseq = iseq[1:-1,1:-1,:]
#    
#    
#    #alpha_tous = np.zeros(9)+np.nan
#    facteur = np.zeros(9)
#    en = np.zeros(9)
#            
#    pchaps = np.zeros(9)     
#    for a in range(9) :       
#        pchaps[a] = (iseq.sum(axis=2)==a).mean()
#        
#        if (iseq.sum(axis=2)==a).sum() < 10:
#            pchaps[a] = 0
#        
#        
#        energies_sans_alpha = phi_theta * (1 - 2*iseq)
#        msk_a = (iseq.sum(axis=2)==a)
#        en[a] = (energies_sans_alpha * msk_a[:,:,np.newaxis]).sum() / msk_a.sum()
#        
#    ratios = pchaps[np.newaxis,:] / pchaps[:,np.newaxis]   
#    ran = np.arange(9) 
#    facteur = ran[np.newaxis,:] - ran[:,np.newaxis] 
#    correc_en = en[np.newaxis,:] / en[:,np.newaxis]  
#    
#    logratios = np.log(ratios)
#    
#    a = logratios/(facteur) * correc_en
#    alpha_tous = a[(np.isnan(a)==0)*(np.isinf(a)==0)*(a!=0)]
#    
#    alpha = alpha_tous.mean()
#    return alpha
#%%   
    
#
#def calc_phi(val_loc,vals_vois):
#    
#    Phi_tot = np.zeros(2+4)
#    
#    #    Phi_tot[0] = (val_loc==0)
#    #    Phi_tot[1] = (val_loc==1)
#    
#    Phi_tot[2] = (1-2*(vals_vois[0]==val_loc)) + (1-2*(vals_vois[4]==val_loc))
#    Phi_tot[3] = (1-2*(vals_vois[1]==val_loc)) + (1-2*(vals_vois[5]==val_loc))
#    Phi_tot[4] = (1-2*(vals_vois[2]==val_loc)) + (1-2*(vals_vois[6]==val_loc))
#    Phi_tot[5] = (1-2*(vals_vois[3]==val_loc)) + (1-2*(vals_vois[7]==val_loc))
#    
#    return Phi_tot[2:].mean()
#
#Phi = np.zeros(shape=(2,9))    
#for classe in (0,1):
#    for nb_pareil in range(9):
#        vois = np.zeros(9)# * (1-classe)
#        vois[:nb_pareil] = 1
#        #        print vois
#        Phi[classe,nb_pareil] = calc_phi(classe,vois)
#
##print Phi
## a deux classes on a une seule combinaison de i,j
#
#vals_vois = it.get_vals_voisins_tout(X)   
#iseq = (X[:,:,np.newaxis]== vals_vois)
#iseq = iseq[1:-1,1:-1]
#Xtemp = X[1:-1,1:-1]
#pchaps =np.zeros(shape=(2,9))    
#for classe in (0,1):
#    for nb_pareil in range(9):       
#        if ((Xtemp==classe)*(iseq.sum(axis=2)==nb_pareil) ).sum() > 10:
#            tot = float((iseq.sum(axis=2)==nb_pareil).sum())
#            pchaps[classe,nb_pareil] = ((Xtemp==classe)*(iseq.sum(axis=2)==nb_pareil) ).sum()/tot
#            
#    #pchaps[classe,:] /= pchaps[classe,:].sum()
##
##pchaps /= pchaps.sum(axis=0)[np.newaxis,:]
##print pchaps
#p0 = pchaps[0,:]
#p1 = pchaps[1,:]
#pchap_ratios = p0[:,np.newaxis] / p1[np.newaxis,:]
#
#phi0= Phi[0,:]
#phi1= Phi[1,:]
#
#phi_diff = - phi0[:,np.newaxis] + phi1[np.newaxis,:]
#
#a = np.log(pchap_ratios)/phi_diff#(np.log(pchaps[0,:]/pchaps[1,:]) ) / (Phi[1,:] - Phi[0,:])
##
##print alpha_tous
#alpha_tous = a[(np.isnan(a)==0)*(np.isinf(a)==0)*(a!=0)]
#    
#alpha = alpha_tous.mean()
#print alpha
#%%  

experiment = '0'
      
# parametres utiles:
if experiment=='3':
    S0 = 90
    S1 = 53
    
else:
    S0 = 128
    S1 = 128

###################
#%%
#v_range = np.array([np.pi/4, np.pi/2, 3*np.pi/4,np.pi])
v_range = np.array([np.pi/4,3*np.pi/4])

###################

##%%
#
alpha = 2.5
alpha_v = 5.
print('---------------------------------------')
pargibbs = parameters.ParamsGibbs(S0 = S0,
                             S1 = S1,
                             type_pot = 'potts',
                             phi_uni = 0.,
                             thr_conv=0.005,
                             nb_iter=100,
                             fuzzy=False,
                             anisotropic=True,   
                             angle=np.zeros(shape=(S0,S1)),
                             beta = 1.,
                             phi_theta_0 = 0.,
                             alpha =alpha,
                             alpha_v = alpha_v,
                             delta = 0.,
                             init_method = 'std',
                             nb_fuzzy = 256. ,
                             v_range = v_range
                             )# beta=1.25,



#------- Generate V
#print('------- Generate V')

if experiment == '1':
    ################ EXPERIMENT 1 : X,V,Y is a TMF

    print('------- Generate X, V')
    pargibbs.nb_iter = 450
    parv = gs.gen_champs_fast(pargibbs, generate_v=True, generate_x=False, use_y=False,normal=False)
    V = parv.V_res[:,:,-1]
#    X_mpm_est,V_mpm_est,X_mpm,V_mpm = sot.MPM(parvx,lim=400)
#    X = X_mpm_est
#    V = V_mpm_est
    pargibbs.nb_iter = 200
    pargibbs.V = V
    parx = gs.gen_champs_fast(pargibbs, generate_v=False, generate_x=True, use_y=False,normal=False)
    X = parx.X_res[:,:,-20:-1].mean(axis=2)>0.5
#
#    np.savez('./data/tmf1.npz', X=X, V=V)
#    dat = np.load('./data/tmf1.npz')
#    X=dat['X']
#    V=dat['V']
    
elif experiment=='2':
    ################ EXPERIMENT 2 :V fixed, X, Y simulated
    print('------- Generate X | V')
    V = v_range[1] * np.ones(shape=(S0,S1))
    V[:S0/2,:S1/2] = v_range[0]
    V[S0/2:,S1/2:] = v_range[0]
    
    pargibbs.nb_iter = 500
    pargibbs.V = V
    parx = gs.gen_champs_fast(pargibbs, generate_v=False, generate_x=True, use_y=False,normal=False)
    X = parx.X_res[:,:,-20:-1].mean(axis=2)>0.5
    X = parx.X_res[:,:,-1]
#    np.savez('./data/tmf_vf1.npz', X=X, V=V)
#    dat = np.load('./data/tmf_vf1.npz')
#    X=dat['X']
#    V=dat['V']

    
elif experiment=='3':
    dat = np.load('./data/lyabig.npz')
    Y = dat['Y']
    Y = Y[:,8:,:]
    Y = Y[:S0,:S1,120:170]
#    
dat = np.load('./data/tmf_good.npz')
X=dat['X']
V=dat['V']
Y = dat['Y']

#    
#    
#def maj_parchamp(parchamp, mu,sig,rho1,rho2,alpha,alpha_v):
#    
#    parchamp.mu = mu
#    parchamp.sig = sig
#    parchamp.rho_1 = rho1
#    parchamp.rho_2 = rho2
#    parchamp.alpha = alpha
#    parchamp.alpha_v = alpha_v
#    
#    return parchamp
#    
#    
#def mesure_ecart(A_tout,A, mu_tout,mu,alpha_tout,alpha,alpha_v_tout,alpha_v,taille_fen):
#
#
#    ecart_a = np.linalg.norm(A_sg[i,:,:]-A_sg[i-taille_fen:i,:,:].mean(axis=0),axis=(0,1))/np.linalg.norm(A_sg[i-taille_fen:i,:,:].mean(axis=0))    
#    
#    ecart_mu = np.linalg.norm(mu_sg[i,:]-mu_sg[i-taille_fen:i,:].mean(axis=0))/np.linalg.norm(mu_sg[i-taille_fen:i,:].mean(axis=0))
#    
#    ecart_alpha = np.abs(alpha_sg[i] - alpha_sg[i-taille_fen:i].mean())/alpha_sg[i-taille_fen:i].mean()
#    ecart_alpha_v = np.abs(alpha_v_sg[i] - alpha_v_sg[i-taille_fen:i].mean())/alpha_v_sg[i-taille_fen:i].mean()
#    
#    ecarts = np.array([ecart_a,ecart_mu,ecart_alpha,ecart_alpha_v])
#    
#    return ecarts
#    
#%%
#alpha = est_param_de(X,V,pargibbs)
#
#print alpha
 
#%%
#%%

#%%
#
#cond = True
#S0 = pargibbs.S0
#S1 = pargibbs.S1   
#V = pargibbs.V
#
#if cond==True:
#    phi_theta = np.ones_like(pargibbs.Vois)    
#    for i in xrange(S0):    
#        for j in xrange(S1):
#            phi_theta[i,j,:] =  ft.gen_beta(pargibbs.Vois[i,j,:],V[i,j],pargibbs.phi_theta_0)   
#            
#    phi_theta = phi_theta[1:-1,1:-1,:]
#    
#vals_vois = it.get_vals_voisins_tout(X)   
#iseq = (X[:,:,np.newaxis]== vals_vois)
#iseq = iseq[1:-1,1:-1,:]
#
#
##alpha_tous = np.zeros(9)+np.nan
#facteur = np.zeros(9)
#en = np.zeros(9)
#        
#pchaps = np.zeros(9)     
#for a in range(9) :       
#    pchaps[a] = (iseq.sum(axis=2)==a).mean()
#    
#    if (iseq.sum(axis=2)==a).sum() < 40:
#        pchaps[a] = 0
#    
#    
#    energies_sans_alpha = phi_theta * (1 - 2*iseq)
#    msk_a = (iseq.sum(axis=2)==a)
#    en[a] = (energies_sans_alpha * msk_a[:,:,np.newaxis]).sum() / msk_a.sum()
#    
#ratios = pchaps[np.newaxis,:] / pchaps[:,np.newaxis]   
#ran = np.arange(9) 
#facteur = ran[np.newaxis,:] - ran[:,np.newaxis] 
#correc_en = en[np.newaxis,:] / en[:,np.newaxis]  
#
#logratios = np.log(ratios)
#
#a = logratios/(facteur) * correc_en
#alpha_tous = a[(np.isnan(a)==0)*(np.isinf(a)==0)*(a!=0)]
#
#alpha = alpha_tous.mean()
#


#%%
#------- Generate Y | X
if experiment=='3':
    W = Y.shape[2]
else:
    W = 20
pargibbs.W = W
m = 0

if experiment=='3':

    pargibbs.S0 = S0
    pargibbs.S1 = S1
#sig = 0.75
#%% Options
SNR = -5
pargibbs.phi_theta_0 = 0. # 1 c'est un champ caché !

#%%

#dat = np.load('./data/estparam.npz')
#X = dat['X'] ; Y = dat['Y']; V = dat['V']

mu = sot.gen_mu(pargibbs.W)
sig = np.linalg.norm(mu)/np.sqrt(pargibbs.W)* 10**(-SNR/20.)
#sig = 0.25
rho_1 = 0.5 * sig**2
rho_2 = 0.25 * sig**2
##
#print('------- Generation donnees : sig2=%.4f (SNR = %.2f dB), rho1= %.4f, rho2 = %.4f'%(sig**2,SNR,rho_1,rho_2))

#if experiment !='3':
#    pargibbs, Y = sot.gen_obs(pargibbs,X,pargibbs.W,mu,sig,rho_1,rho_2,corrnoise=True)
#    
#else:
#    X = np.zeros(shape=(S0,S1))
#    V = np.zeros(shape=(S0,S1))
#
#
#
#
pargibbs.Y = Y

#%%



############### Segmentation


 # 1) SEM
#parchamp = parameters.ParamsChamps()
#v_help = True
#pargibbs.nb_nn_v_help = 1 #nb. dependance assez lourde a ce parametre...
#pargibbs.v_help==v_help
#
#
#nb_iter_sem = 10
#print 'SEM ...'
#start=time.time()
#parsem = sot.SEM(parchamp,pargibbs,nb_iter_sem)
#temps =  (time.time()-start)
#print '     %.0f iterations et %.2f s - %.3f s/iter.'%(nb_iter_sem, temps,temps/nb_iter_sem  )

# 2) Gradient stochastique
#%%
#np.savez('./data/estparam.npz', X=X, V=V,Y=Y)    
# repeat for a number of trials
nb_t = 1

mu_t = np.zeros(shape=(nb_t, W))
A_t = np.zeros(shape=(nb_t,W,W))
alpha_t = np.zeros(shape=nb_t)
alpha_v_t = np.zeros(shape=nb_t)
rho1_t = np.zeros(shape=nb_t)
rho2_t = np.zeros(shape=nb_t)
sig_t = np.zeros(shape=nb_t)
    
#%%
seuil_conv = 0.02
taille_fen_sg = 10
pargibbs.nb_iter = 200
nb_iter_sg = 40    
nb_iter_sem = 40
taille_fen_sem = 7
    

for t in range(nb_t):
    print('----- %.4f, %.4f, %.4f, %.4f, %.4f'%(sig**2,rho_1,rho_2,alpha, alpha_v))
    print('essai %.0f'%(t+1))

    
    
    parchamp = parameters.ParamsChamps()
    start = time.time()
    parchamp = sot.SEM(parchamp,pargibbs,nb_iter_sem, seuil_conv, taille_fen_sem)
    mu_sem,sig_sem,rho1_sem,rho2_sem,alpha_sem,alpha_v_sem,A_sem = parchamp.mu_sem, parchamp.sig_sem,parchamp.rho1_sem, parchamp.rho2_sem, parchamp.alpha_sem,parchamp.alpha_v_sem,parchamp.A_sem 
    #mu_sg, A_sg, alpha_sg,alpha_v_sg, sig_sg, rho1_sg, rho2_sg = sot.stogra(pargibbs,nb_iter_sg,seuil_conv, taille_fen_sg)
            
    mu_t[t,:] = mu_sem[-taille_fen_sg:-1,:].mean(axis=0)
    A_t[t,:,:] = A_sem[-taille_fen_sg:-1,:,:].mean(axis=(0,1))
    
    alpha_t[t] = alpha_sem[-taille_fen_sg:-1].mean()
    alpha_v_t[t] = alpha_v_sem[-taille_fen_sg:-1].mean()
    
    rho1_t[t] = rho1_sem[-taille_fen_sg:-1].mean()
    rho2_t[t] = rho2_sem[-taille_fen_sg:-1].mean()
    sig_t[t] = sig_sem[-taille_fen_sg:-1].mean()
        
    
    
    
    end = time.time()
    print 'Temps : %.2f s'%(end-start)

#%% Saving results
import os

if os.path.exists('./results/trials_sem.npz') == False:
    # first time saving in this dir
    np.savez('./results/trials_sem.npz', mu_t=mu_t,
                                        A_t=A_t,
                                        alpha_t=alpha_t,
                                        alpha_v_t=alpha_v_t,
                                        rho1_t=rho1_t, 
                                        rho2_t=rho2_t,
                                        sig_t=sig_t,                                    
                                        )
else:
    dat = np.load('./results/trials_sem.npz')
    np.savez('./results/trials_sem.npz', mu_t=np.append(mu_t,dat['mu_t'],axis=0),
                                        A_t=np.append(A_t,dat['A_t'],axis=0),
                                        alpha_t=np.append(alpha_t,dat['alpha_t']),
                                        alpha_v_t=np.append(alpha_v_t,dat['alpha_v_t']),
                                        rho1_t=np.append(rho1_t,dat['rho1_t']), 
                                        rho2_t=np.append(rho2_t,dat['rho2_t']),
                                        sig_t=np.append(sig_t,dat['sig_t']),  
                                        SNR = SNR,
                                        X=X,
                                        V=V
                                        )
                                        
#%% Analyse resultats
dat = np.load('./results/trials_sem.npz')


mu_t = dat['mu_t']
mu_bias = (mu_t.mean(axis=0)-mu).mean()                              
mu_std = np.std(mu_t,axis=0).mean()

sig_t = dat['sig_t']
sig_bias = (sig_t.mean()-sig)                              
sig_std = np.std(sig_t)

rho1_t = dat['rho1_t']
rho1_bias = (rho1_t.mean()-rho_1)                              
rho1_std = np.std(rho1_t)

rho2_t = dat['rho2_t']
rho2_bias = (rho2_t.mean()-rho_2)                              
rho2_std = np.std(rho2_t)

alpha_t = dat['alpha_t']
alpha_bias = (alpha_t.mean()-alpha)                              
alpha_std = np.std(alpha_t)

alpha_v_t = dat['alpha_v_t']
alpha_v_bias = (alpha_v_t.mean()-alpha_v)                              
alpha_v_std = np.std(alpha_v_t)


print ('------------------------------ \n'
      +'Gradient stochastique - %.0f essais \n'
      +'mu (valeurs moyennes)      biais = %.6f  std = %.6f \n'
      +'sigma   : reel = %.6f  biais = %.6f  std = %.6f \n'
      +'rho_1   : reel = %.6f  biais = %.6f  std = %.6f \n' 
      +'rho_2   : reel = %.6f  biais = %.6f  std = %.6f \n'
      +'alpha   : reel = %.6f  biais = %.6f  std = %.6f \n'
      +'alpha_v : reel = %.6f  biais = %.6f  std = %.6f \n')%(dat['sig_t'].size,
                                                        mu_bias,mu_std,
                                                        sig,sig_bias,sig_std,
                                                        rho_1, rho1_bias,rho1_std,
                                                        rho_2, rho2_bias,rho2_std,
                                                        alpha, alpha_bias,alpha_std,
                                                        alpha_v, alpha_v_bias,alpha_v_std)                         
#%%
#
#var_mu = np.linalg.norm(mu_sg[1:,:]-mu_sg[:-1,:],axis=1)
#var_alpha = np.abs(alpha_sg[1:]-alpha_sg[:-1])
#var_alpha_v = np.abs(alpha_v_sg[1:]-alpha_v_sg[:-1])
#var_a = np.linalg.norm(A_sg[1:,:,:]-A_sg[:-1,:,:],axis=(1,2))
#
#

#%%
#
#def mesure_ecart(A_tout,A, mu_tout,mu,alpha_tout,alpha,alpha_v_tout,alpha_v,taille_fen):
#
#
#    ecart_a = np.linalg.norm(A_sg[i,:,:]-A_sg[i-taille_fen:i,:,:].mean(axis=0),axis=(0,1))/np.linalg.norm(A_sg[i-taille_fen:i,:,:].mean(axis=0))    
#    
#    ecart_mu = np.linalg.norm(mu_sg[i,:]-mu_sg[i-taille_fen:i,:].mean(axis=0))/np.linalg.norm(mu_sg[i-taille_fen:i,:].mean(axis=0))
#    
#    ecart_alpha = np.abs(alpha_sg[i] - alpha_sg[i-taille_fen:i].mean())/alpha_sg[i-taille_fen:i].mean()
#    ecart_alpha_v = np.abs(alpha_v_sg[i] - alpha_v_sg[i-taille_fen:i].mean())/alpha_v_sg[i-taille_fen:i].mean()
#    
#    ecarts = np.array([ecart_a,ecart_mu,ecart_alpha,ecart_alpha_v])
#    
#    return ecarts
#
#taille_fen = 7
#ecart_alpha = np.zeros(shape=nb_iter_sg-taille_fen)
#ecart_alpha_v = np.zeros(shape=nb_iter_sg-taille_fen)
#ecart_mu = np.zeros(shape=nb_iter_sg-taille_fen)
#ecart_a = np.zeros(shape=nb_iter_sg-taille_fen)
#ecart_tous = ecart_a = np.zeros(shape=(nb_iter_sg-taille_fen,4))
#
#for i in range(taille_fen,nb_iter_sg):
#    
#    ecart_tous[i-taille_fen] = mesure_ecart(A_sg[:i,:,:],A_sg[i,:,:], mu_sg[:i,:],mu_sg[i,:],alpha_sg[:i],alpha_sg[i],alpha_v_sg[:i],alpha_v_sg[i],taille_fen)
##    ecart_a[i-taille_fen] = np.linalg.norm(A_sg[i,:,:]-A_sg[i-taille_fen:i,:,:].mean(axis=0),axis=(0,1))/np.linalg.norm(A_sg[i-taille_fen:i,:,:].mean(axis=0))    
##    
##    ecart_mu[i-taille_fen] = np.linalg.norm(mu_sg[i,:]-mu_sg[i-taille_fen:i,:].mean(axis=0))/np.linalg.norm(mu_sg[i-taille_fen:i,:].mean(axis=0))
##    
##    ecart_alpha[i-taille_fen] = np.abs(alpha_sg[i] - alpha_sg[i-taille_fen:i].mean())/alpha_sg[i-taille_fen:i].mean()
##    ecart_alpha_v[i-taille_fen] = np.abs(alpha_v_sg[i] - alpha_v_sg[i-taille_fen:i].mean())/alpha_v_sg[i-taille_fen:i].mean()
#
##%%  
#plt.figure(figsize=(10,10))
#
#plt.plot(ecart_alpha, label='$\\alpha^x$')
#plt.plot(ecart_alpha_v, label='$\\alpha^{x|v}$')
#plt.plot(ecart_mu,label='$\\mu$')
#plt.plot(ecart_a,label='$A$')
#plt.legend()

##%%  
#plt.figure(figsize=(10,10))
#
#plt.plot(var_alpha, label='$\\alpha^x$')
#plt.plot(var_alpha_v, label='$\\alpha^{x|v}$')
#plt.plot(var_mu/W,label='$\\mu$')
#plt.plot(var_a/(3*W),label='$A$')
#
#plt.legend()
#%%
#plt.close('all')
plt.figure(figsize=(20,10))
nb_li = 2
nb_col = 4

plt.subplot(nb_li, nb_col,1)
plt.plot(mu-1,':k')
for i in range(alpha_sg.size):
    plt.plot(mu+i,':k')
    plt.plot(mu_sg[i,:]+i)

plt.ylim((-1.1,None))
plt.ylabel('$k$ (iter. SEM)')
plt.xlabel('$\\lambda$')
plt.title('$\\hat{\\mu}^k$')


residus = mu_sg - mu[np.newaxis,:]
eqm = (residus**2).mean(axis=1)

plt.subplot(nb_li, nb_col,2)
plotref(eqm,0,'EQM sur $\\hat{\\mu}^k$' )
#plt.plot(eqm)
#plt.ylim((0,None))
#plt.xlabel('$k$ (iter. SG)')
#plt.title()

plt.subplot(nb_li,nb_col,3)
plotref(sig_sg,sig,'$\\hat{\\sigma}^k$')

plt.subplot(nb_li, nb_col,4)
plotref(rho1_sg,rho_1,'$\\hat{\\rho}_1^k$')

plt.subplot(nb_li, nb_col,5)
plotref(rho2_sg,rho_2,'$\\hat{\\rho}_2^k$')

plt.subplot(nb_li,nb_col,6)
plotref(alpha_sg,alpha,'$\\hat{\\alpha^{x|v}}^k$')

plt.subplot(nb_li,nb_col,7)
plotref(alpha_v_sg,alpha_v,'$\\hat{\\alpha^v}^k$')

plt.tight_layout()

##%%
#plt.figure(figsize=(10,10))
#plt.subplot(231)
#plt.imshow(X)
#plt.subplot(234)
#plt.imshow(V)
#plt.subplot(232)
#plt.imshow(X_courant1)
#plt.subplot(235)
#plt.imshow(V_courant1)
#plt.subplot(233)
#plt.imshow(X_courant2)
#plt.subplot(236)
#plt.imshow(V_courant2)
#%%
#
#
#cond = False
#
#if cond==True:
#    phi_theta = np.ones_like(pargibbs.Vois)    
#    for i in xrange(S0):    
#        for j in xrange(S1):
#            phi_theta[i,j,:] =  ft.gen_beta(pargibbs.Vois[i,j,:],V[i,j],pargibbs.phi_theta_0)   
#            
#    phi_theta = phi_theta[1:-1,1:-1,:]
#    
#vals_vois = it.get_vals_voisins_tout(X)   
#iseq = (X[:,:,np.newaxis]== vals_vois)
#iseq = iseq[1:-1,1:-1,:]
#
#
#alpha_tous = np.zeros(9)+np.nan
#facteur = np.zeros(9)
#
#
## essayons 3,5
#a = 5
#phi = 2
#
#iseq_a = (iseq.sum(axis=2)==a).mean()
#isneq_a = (iseq.sum(axis=2)==(8-a)).mean()
#
#pchap_a = iseq_a/(iseq_a + isneq_a)
#pchap_acomp = isneq_a/(iseq_a + isneq_a)
#
##%%
#
#for a in range(8):
#    print a
#    iseq_a = (iseq.sum(axis=2)==a).mean() # exp(-2 alpha) /C
#    isneq_a = (iseq.sum(axis=2)==(8-a)).mean() # exp(2 alpha) / C
#    
#    msk_a = (iseq.sum(axis=2)==a)
#    msk_acomp = (iseq.sum(axis=2)==(8-a))  
#    
#    #if (iseq_a !=0) and (isneq_a !=0) and ((8-a)!=a): #and (msk_acomp.sum()>5) and (msk_a.sum()>5):
#    pchap_a = iseq_a/(iseq_a + isneq_a)
#    pchap_acomp = isneq_a/(iseq_a + isneq_a)
#    
#    facteur[a] = (a - (8.-a))
#    
#    if cond==False:
#        alpha_tous[a] = 1./(2*facteur[a]) * np.log(pchap_a/pchap_acomp)
#    
#    else:
#                    
#        energies_sans_alpha = phi_theta * (1 - 2*iseq)
#        #print energies_sans_alpha.mean()
#        #msk_a = (iseq.sum(axis=2)==a)
#        #msk_acomp = (iseq.sum(axis=2)==(8-a))  
#        
#        en_a = (energies_sans_alpha * msk_a[:,:,np.newaxis]).sum() / msk_a.sum()
#        en_acomp = (energies_sans_alpha * msk_acomp[:,:,np.newaxis]).sum() / msk_acomp.sum()
#        
#        if a > (8-a):
#            ratio = (en_a/en_acomp)
#        else:
#            ratio = (en_acomp/en_a)
#        alpha_tous[a] = 1./(2*facteur[a]) * np.log( pchap_acomp/pchap_a) *ratio
#        
#            #print ratio
##print alpha_tous                        
#        
#
##print alpha_tous
#alpha_est = alpha_tous[~np.isnan(alpha_tous)].mean()

#%% Affichages EstParams
#
#nb_li = 2
#nb_col = 4
#plt.figure(figsize=(5*nb_col,5*nb_li))
#sig_sem= parsem.sig_sem
#rho_1sem= parsem.rho_1sem
#rho_2sem= parsem.rho_2sem
#mu_sem = parsem.mu_sem
#alpha_sem = parsem.alpha_sem
#alpha_v_sem = parsem.alpha_v_sem
#
#plt.subplot(nb_li,nb_col,1)
#plt.plot(sig_sem)
#plt.plot(sig*np.ones(np.size(sig_sem)),':k')
#plt.xlabel('$k$ (iter. SEM)')
#plt.title('$\\hat{\\sigma}^k$')
#plt.ylim((0.9*sig,1.5*sig))
#
#plt.subplot(nb_li,nb_col,2)
#plt.plot(mu-1,':k')
#for i in range(nb_iter_sem):
#    plt.plot(mu+i,':k')
#    plt.plot(mu_sem[i,:]+i)
#
#plt.ylim((-1.1,None))
#plt.ylabel('$k$ (iter. SEM)')
#plt.xlabel('$\\lambda$')
#plt.title('$\\hat{\\mu}^k$')
#
#
#residus = mu_sem - mu[np.newaxis,:]
#eqm = (residus**2).sum(axis=1)
#
#plt.subplot(nb_li,nb_col,3)
#plt.plot(eqm)
#plt.ylim((0,None))
#plt.xlabel('$k$ (iter. SEM)')
#plt.title('EQM sur $\\hat{\\mu}^k$')
#
#
#plt.subplot(nb_li,nb_col,5)
#plt.plot(rho_1sem,'b')
#plt.plot(rho_1*np.ones(np.size(sig_sem)),':b')
#plt.plot(rho_2sem,'r')
#plt.plot(rho_2*np.ones(np.size(sig_sem)),':r')
#plt.xlabel('$k$ (iter. SEM)')
#plt.title('$\\hat{\\rho}_1^k$ and $\\hat{\\rho}_1^k$')
#
#
#plt.subplot(nb_li,nb_col,6)
#plt.plot(alpha_sem)
#plt.plot(alpha*np.ones(np.size(alpha_sem)),':k')
#plt.xlabel('$k$ (iter. SEM)')
#plt.title('$\\hat{\\alpha}^k$')
#plt.ylim((0,2))
#
#
#plt.subplot(nb_li,nb_col,7)
#plt.plot(alpha_v_sem)
#plt.plot(alpha_v*np.ones(np.size(alpha_v_sem)),':k')
#plt.xlabel('$k$ (iter. SEM)')
#plt.title('$\\hat{\\alpha^v}^k$')
#plt.ylim((0,2))
#
#plt.subplot(nb_li,nb_col,4)
#plt.imshow(X.T, interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin=0,vmax=1); 
#plt.axis('off')
#plt.title('$X$')
##
#
#
#plt.tight_layout()
###
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
