# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:40:02 2016

Tous les algos et methodes relatifs au gradient stochastique.

@author: courbot
"""

import numpy as np 

import image_tools as it


import parameters
#import image_tools as it
import gibbs_sampler as gs
import fields_tools as ft
import est_param_gen as epg
import seg_OTMF as sot


import numpy.ma as ma



def stogra(pargibbs, nb_iter_sg, seuil_conv, taille_fen):
        
    print 'GS init...'
    cond_x = True
    cond_v = False
    mu_true = gen_mu(pargibbs.W)
    Y_obs = np.copy(pargibbs.Y)

    W = pargibbs.W
    S0 = pargibbs.S0
    S1 = pargibbs.S1

    parchamp = parameters.ParamsChamps()
    #
    
    parchamp,X_courant,V_courant = init_params(pargibbs,parchamp,cond_x,cond_v)
    
    Sigma_courant = gen_cov(W,parchamp.sig,parchamp.rho_1,parchamp.rho_2)
    A_courant = np.linalg.inv(Sigma_courant)
    sig_courant,rho1_courant,rho2_courant = get_parcov(Sigma_courant)
    alpha_courant = 3.
    alpha_v_courant = 3.
    
    mu_courant = np.copy(parchamp.mu)



    mu_sg = np.zeros(shape=(nb_iter_sg,W))
    A_sg = np.zeros(shape=(nb_iter_sg,W,W))
    
    
    sig_sg = np.zeros(shape=(nb_iter_sg))
    rho1_sg = np.zeros(shape=(nb_iter_sg))
    rho2_sg = np.zeros(shape=(nb_iter_sg))
    alpha_sg = np.zeros(shape=(nb_iter_sg))
    alpha_v_sg = np.zeros(shape=(nb_iter_sg))

    ecart_tous =  np.zeros(shape=(nb_iter_sg-taille_fen,4))

    
    mse = np.zeros(nb_iter_sg)
    X_courant1 = np.copy(X_courant)
    X_courant2 = np.copy(X_courant)
    
    V_courant1 = np.copy(V_courant)
    V_courant2 = np.copy(V_courant)
    
    i = 0
    mu_sg[i,:] = mu_courant
    sig_sg[i] = sig_courant
    rho1_sg[i] = rho1_courant
    rho2_sg[i] = rho2_courant
     
    alpha_sg[i] = alpha_courant
    alpha_v_sg[i] = alpha_v_courant
    
    
    
    
    print 'GS iter...'
    print '  mse    sigma   rho1    rho2   alpha  alpha_v'
    mse[i] = np.mean((mu_courant-mu_true)**2)
    print '%.4f, %.4f, %.4f, %.4f, %.4f, %.4f'%(mse[i], sig_sg[i]**2, rho1_sg[i], rho2_sg[i],alpha_sg[i], alpha_v_sg[i])
    for i in range(1,nb_iter_sg):
         
         pargibbs.mu = mu_courant
         pargibbs.sig = sig_courant
         pargibbs.rho_1 = rho1_courant
         pargibbs.rho_2 = rho2_courant
         
         pargibbs.alpha = alpha_courant
         pargibbs.alpha_v = alpha_v_courant
    
    
         eta = 10./((i+1)*W*S0*S1) 

         
         # (a) generation triplet y,x,v
         #     pargibbs.X_init = X_courant1
         #     pargibbs.V_init = V_courant1
         parvx = gs.gen_champs_fast(pargibbs, generate_v=True, generate_x=True, use_y=False)
         X_courant1 = parvx.X_res[:,:,-1]
         V_courant1 = parvx.V_res[:,:,-1]
         
         par, Y_courant = gen_obs(pargibbs,X_courant1,pargibbs.W,mu_courant,sig_courant,rho1_courant,rho2_courant,corrnoise=True)
         
         # (a-2) calcul du gradient correspondant
         y_moins_mu = Y_courant - mu_courant[np.newaxis,np.newaxis,:]*X_courant1[:,:,np.newaxis]
         y_moins_mu_vec = np.reshape(y_moins_mu, (S0*S1,W))
         
         grad_mu = calc_grad_mu(y_moins_mu_vec,A_courant)
         grad_a = calc_grad_a(y_moins_mu_vec)
         grad_alpha = calc_grad_alphax(pargibbs,X_courant1,V_courant1)
         grad_alpha_v = calc_grad_alphav(V_courant1)
         
         
         # (b) generation triplet x,v|y
         pargibbs.Y = Y_obs
         #     pargibbs.X_init = X_courant2
         #     pargibbs.V_init = V_courant2
         parvx = gs.gen_champs_fast(pargibbs, generate_v=True, generate_x=True, use_y=True)
         X_courant2 = parvx.X_res[:,:,-1]
         V_courant2 = parvx.V_res[:,:,-1]
         
         # (b-2) calcul du gradient correspondant
         y_moins_mu = Y_obs - mu_courant[np.newaxis,np.newaxis,:]*X_courant2[:,:,np.newaxis]
         y_moins_mu_vec = np.reshape(y_moins_mu, (S0*S1,W))
         
         grad_cond_mu = calc_grad_mu(y_moins_mu_vec,A_courant)
         grad_cond_a = calc_grad_a(y_moins_mu_vec)
         grad_cond_alpha = calc_grad_alphax(pargibbs,X_courant2,V_courant2)
         grad_cond_alpha_v = calc_grad_alphav(V_courant2)
         
         
         
         delta_mu = eta* (grad_mu-grad_cond_mu)
         delta_a = eta * (grad_a-grad_cond_a)
         delta_alpha = eta * W* (S0*S1)/((S0-2)*(S1-2))*(grad_alpha-grad_cond_alpha)
         delta_alpha_v = eta *W*  (S0*S1)/((S0-2)*(S1-2))*(grad_alpha_v-grad_cond_alpha_v)
         
         
         mu_courant = mu_courant + delta_mu
         A_courant = A_courant + delta_a
         alpha_courant = alpha_courant + delta_alpha
         alpha_v_courant = alpha_v_courant + delta_alpha_v
         
         
              
         Sigma_courant = np.linalg.inv(A_courant)
         sig_courant,rho1_courant,rho2_courant = get_parcov(Sigma_courant)
         
         
         mu_sg[i,:] = mu_courant
         A_sg[i,:,:] = np.linalg.inv(gen_cov(W,sig_courant,rho1_courant,rho2_courant))
         sig_sg[i] = sig_courant
         rho1_sg[i] = rho1_courant
         rho2_sg[i] = rho2_courant
         
         alpha_sg[i] = alpha_courant
         alpha_v_sg[i] = alpha_v_courant
         
         mse[i] = np.mean((mu_courant-mu_true)**2)
         
         #parchamp = maj_parchamp(parchamp, mu_courant,sig_courant,rho1_courant,rho2_courant,alpha_courant,alpha_v_courant)
         print '%.4f, %.4f, %.4f, %.4f, %.4f, %.4f'%(mse[i], sig_sg[i]**2, rho1_sg[i], rho2_sg[i],alpha_sg[i], alpha_v_sg[i])
         
         
         if i > taille_fen:
              ecart_tous[i-taille_fen] = mesure_ecart(A_sg[:i,:,:],A_sg[i,:,:], mu_sg[:i,:],mu_sg[i,:],alpha_sg[:i],alpha_sg[i],alpha_v_sg[:i],alpha_v_sg[i],taille_fen)
              
              if (ecart_tous[i-taille_fen,:]<seuil_conv).all() == True:
                  print 'stop iter %.0f'%i
                  # lets truncate parameters arrays
                  mu_sg = mu_sg[:i+1,:]
                  A_sg = A_sg[:i+1,:,:]
                  alpha_sg = alpha_sg[:i+1]
                  alpha_v_sg = alpha_v_sg[:i+1]
                  sig_sg = sig_sg[:i+1]
                  rho1_sg = rho1_sg[:i+1]
                  rho2_sg = rho2_sg[:i+1]
                  break
              
    return mu_sg, A_sg, alpha_sg,alpha_v_sg, sig_sg, rho1_sg, rho2_sg








def calc_grad_mu(y_moins_mu_vec,A):

    grad = -2*np.dot(y_moins_mu_vec,A).sum(axis=0)
    
    return grad   
    
def calc_grad_a(y_moins_mu_vec):
    return np.dot(y_moins_mu_vec.T,y_moins_mu_vec)    


def calc_grad_alphax(pargibbs,X,V):
    phi_vois = np.ones_like(pargibbs.Vois)    
    for i in xrange(pargibbs.S0):
        for j in xrange(pargibbs.S1):
            phi_vois[i,j,:] =  ft.gen_beta(pargibbs.Vois[i,j],V[i,j],pargibbs.phi_theta_0)
    #        
    vals_vois = it.get_vals_voisins_tout(X)  
    iseq0 = (X[:,:,np.newaxis]== vals_vois)
    #
    iseq0 = iseq0[1:-1,1:-1,:]
    phi_vois = phi_vois[1:-1,1:-1,:]     
    
    grad = -np.sum( (1-2*iseq0) * phi_vois )  # c'est a la fois une somme sur les cliques et sur le voisinage.
    return grad

def calc_grad_alphav(V):

    #        
    vals_vois = it.get_vals_voisins_tout(V)  
    iseq0 = (V[:,:,np.newaxis]== vals_vois)
    #
    iseq0 = iseq0[1:-1,1:-1,:]
    
    grad = -np.sum( (1-2*iseq0))   # c'est a la fois une somme sur les cliques et sur le voisinage.
    return grad
