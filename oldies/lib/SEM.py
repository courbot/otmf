# -*- coding: utf-8 -*-
"""
This module contains all the functions related to parameter estimation.

:author: Jean-Baptiste Courbot - jb.courbot@unistra.fr
:date: Sep 01, 2017 (created Apr 20, 2016)
"""
import numpy as np 
from . import fields_tools as ft
from . import est_param_gen as epg
from . import seg_OTMF as sot


def SEM(parseg,parchamp,pargibbs,disp=False):
    """ Stochastic Expectation-Maximization algorithm.
    
    This iterative method allows, at each iteration, a parameter re-estimation 
    directly from complete data thanks to realization of the missing data, 
    simulated using parameters estimated at the previous step.
    
    See the original works from Celeux and Diebolt.
    

    :param parameters parseg: parameters ruling the segmentation method
    :param parameters parchamp: parameters of the model (priors, noise parameters)
    :param parameters pargibbs: parameters of the Gibbs sampling.
    :param bool disp: set the verbose mode [True]

    :returns: **parchamp** *(parameter)* - set of estimated parameters
    """
    Y = pargibbs.Y
    W = pargibbs.W
    if parseg.multi:
        nb_classe = pargibbs.x_range.size
    else:
        nb_classe = 1 #useless?
        
    # recuperation parametres de segmentation
    nb_iter_sem, seuil_conv, taille_fen = parseg.nb_iter_sem, parseg.seuil_conv,parseg.taille_fen
  
    # mono versus multi-band
    if W == 1:
        mono = True
    else:
        mono=False
    x_range=pargibbs.x_range

    #==============================================================================
    #   Values initialization
    #==============================================================================
    sig_sem = np.zeros(shape=(nb_iter_sem,nb_classe))
    alpha_sem = np.zeros(shape=(nb_iter_sem))
    alpha_v_sem = np.zeros(shape=(nb_iter_sem))

    mu_sem = np.zeros(shape=(nb_iter_sem,W,nb_classe)) ;    
    rho1_sem, rho2_sem = np.zeros(shape=(nb_iter_sem)), np.zeros(shape=(nb_iter_sem))
    pi_sem =  np.zeros(shape=(nb_iter_sem,2,9)) # deux champs, 9 types de config
    
    if mono==False:
        A_sem = np.zeros(shape=(nb_iter_sem,W,W)) # ce sont les matrices de covariance
    else:
        A_sem = np.zeros(shape=(nb_iter_sem)) # ce sont des ecart-type
        
    # mesure des écarts entre paramètres
    ecart_tous =  np.zeros(shape=(nb_iter_sem-taille_fen,3))
    
    if disp:
        print '  SEM init...'
    if hasattr(pargibbs, 'X_init') :
        parchamp = est_param_noise(pargibbs.X_init,Y,parchamp,x_range) # modif pour multiclasse fait

        parchamp.pi = np.zeros(shape=(2,9))
        parchamp.pi[0,:] = np.exp(np.arange(9.)) / np.exp(np.arange(9.)).sum()
        parchamp.pi[1,:] = np.exp(np.arange(9.)) / np.exp(np.arange(9.)).sum()
        
        parchamp.alpha = 1.#est_param_de_x(X_courant)
        parchamp.alpha_v = 1.

    else:
        
        parchamp,X_courant,V_courant = epg.init_params(pargibbs,parchamp)

    if disp:
        print '  SEM iter...'
    for iter_sem in xrange(nb_iter_sem):

        i = iter_sem

        pargibbs.parchamp = parchamp

        if parseg.tmf==True:
            
            pargibbs.V = np.zeros_like(X_courant)
            
            nb_iter_to_keep = np.copy(pargibbs.nb_iter)
            pargibbs.nb_iter = 100
            pargibbs=sot.serie_gibbs(pargibbs,nb_rea=int(parseg.nb_iter_serie_sem),generate_v=True,generate_x=True,use_y=True,use_pi=True,tmf=True)#(pargibbs,nb_rea=int(parseg.nb_iter_serie_sem))
            pargibbs.nb_iter = np.copy(nb_iter_to_keep)        
            
            X_courant, V_courant, d, d = sot.MPM_uncert(pargibbs, tmf=False)
            
            if (X_courant.sum() == 0) or ((1.-X_courant).sum()==0):
                X_courant = np.random.random(size=X_courant.shape) > 0.5
        else:
            
            # a) Simulation given the previous parameter
           
            pargibbs.V = np.zeros_like(X_courant)
            
            nb_iter_to_keep = np.copy(pargibbs.nb_iter)
            pargibbs.nb_iter = 100
            pargibbs=sot.serie_gibbs(pargibbs,nb_rea=int(parseg.nb_iter_serie_sem),generate_v=False,generate_x=True,use_y=True,use_pi=True,tmf=False)#(pargibbs,nb_rea=int(parseg.nb_iter_serie_sem))
            pargibbs.nb_iter = np.copy(nb_iter_to_keep)        
            
            X_courant, d, d, d = sot.MPM_uncert(pargibbs, tmf=False)
            
            #        parvx = gen_champs_fast(pargibbs)            
            #        X_courant = parvx.X_res[:,:,-1]
            
            if (X_courant.sum() == 0) or ((1.-X_courant).sum()==0):
                X_courant = np.random.random(size=X_courant.shape) > 0.5

            V_courant = np.zeros_like(X_courant)

        #==============================================================================
        #         # Estimation des parametres a partir de donnees completes
        #==============================================================================
        parchamp = est_param_noise(X_courant,Y,parchamp,x_range) # modif pour multiclasse fait

        
        if parseg.use_pi :
            parchamp.pi = est_pi(X_courant,V_courant, pargibbs)#est_pi(X_courant, pargibbs)
        else:
            parchamp.pi = np.ones_like(parchamp.pi)
            
        if parseg.use_alpha :
            a  = est_param_de_x(X_courant)#est_param_de_x(X_courant,pargibbs)
            if np.isnan(a):
                parchamp.alpha = 1.0
            else:
                parchamp.alpha = a#est_param_de_x(X_courant,pargibbs)
        else:
            parchamp.alpha = 1.0   
        
        parchamp.alpha_v = 1.
        
        #==============================================================================
        #         on stocke ces donnees
        #==============================================================================

        # attention la forme du mu pourrait poser probleme en mono classe
        #faire une option ici
        
                #ici   
        mu_sem[iter_sem,:,:], sig_sem[iter_sem,:], pi_sem[iter_sem,:,:], alpha_sem[iter_sem], alpha_v_sem[iter_sem]= parchamp.mu.T, parchamp.sig, parchamp.pi,parchamp.alpha,parchamp.alpha_v
        
        if mono==False:
            rho1_sem[iter_sem], rho2_sem[iter_sem] = parchamp.rho_1, parchamp.rho_2
            A_sem[i,:,:] = np.linalg.inv(epg.gen_cov(W,parchamp.sig,parchamp.rho_1,parchamp.rho_2))

        #==============================================================================
        #       Decision to stop or continue
        #==============================================================================
        if i > taille_fen:
              if mono==False:
                  ecart_tous[i-taille_fen,:] = epg.mesure_ecart(A_sem[:i,:,:],A_sem[i,:,:], mu_sem[:i,:],mu_sem[i,:],pi_sem[:i,:,:],pi_sem[i,:,:],taille_fen,W)
              else:
                          #ici   
#                  print ecart_tous[i-taille_fen,:]#shape
                  ecart_tous[i-taille_fen,:] = epg.mesure_ecart(sig_sem[:i,:],sig_sem[i,:], mu_sem[:i,:],mu_sem[i,:],pi_sem[:i,:,:],pi_sem[i,:,:],taille_fen,W)
              
              if (ecart_tous[i-taille_fen,:]<seuil_conv).all() == True:
                  if disp:
                      print 'stop iter %.0f'%i
                  # Trucation of parameter arrays
                  mu_sem, pi_sem, sig_sem = mu_sem[:i+1,:], pi_sem[:i+1,:,:], sig_sem[:i+1]

                  if mono==False:
                      A_sem, rho1_sem,rho2_sem  = A_sem[:i+1,:,:],rho1_sem[:i+1],rho2_sem[:i+1]
                  else:
                      sig_sem = sig_sem[:i+1]
                  break

    #==============================================================================
    #   Stockage des parametres
    #==============================================================================
    parchamp.mu_sem, parchamp.sig_sem, parchamp.pi_sem = mu_sem, sig_sem, pi_sem
    
    if mono == False:
        parchamp.rho1_sem, parchamp.rho2_sem, parchamp.A_sem = rho1_sem, rho2_sem, A_sem
       
    #==============================================================================
    #   moyenne des derniers parametres
    #==============================================================================
    
    parchamp.mu = mu_sem[-taille_fen:-1,:,:].mean(axis=0)
    parchamp.sig = sig_sem[-taille_fen:-1,:].mean(axis=0)
    parchamp.alpha = alpha_sem[-taille_fen:-1].mean()
    parchamp.alpha_v = alpha_v_sem[-taille_fen:-1].mean()
    parchamp.pi = pi_sem[-taille_fen:-1,:,:].mean(axis=0)
    
    if mono ==False:
        parchamp.rho_1 = rho1_sem[-taille_fen:-1].mean()
        parchamp.rho_2 = rho2_sem[-taille_fen:-1].mean()
    
        parchamp.A = A_sem[-taille_fen:-1,:,:].mean(axis=0)

     
    return parchamp
    
def est_param_noise(X,Y,parchamp,x_range):
    """ 
    Noise parameter estimation from complete data.
    
    The mean and variance estimators are based on standard pseudo 
    maximum-likelihood estimators.
    

    :param ndarray X: "hidden" classification
    :param ndarray Y: Hyperspectral observation, arranged in x, y, lambda.
    :param misc parchamp: parameters of the model (priors, noise parameters)
    :param ndarray x_range: possibles values for x

    :returns: **parchamp** *(parameter)* - set of estimated parameters
    """
    W = Y.shape[2]
    weights = parchamp.weights
    weights_1d =weights.reshape(Y.shape[0]* Y.shape[1])
    # mono versus multi-band
    if W == 1:
        mono = True
    else:
        mono=False

    if mono==False:
#        liste_vec = np.reshape(Y,(Y.shape[0]*Y.shape[1],W))
        nanmap = np.isnan(Y).any(axis=2)
        msk = (nanmap).reshape(Y.shape[0]*Y.shape[1])  
    else:
        msk = np.isnan(Y).flatten()


#==============================================================================
#     Le cas particulier de l'astro
#==============================================================================
    
    if parchamp.multi==False and  mono==False: # 2 classes hyperspectral
            # OK
    
            # moyenne spectrale
            mu = np.zeros(shape=(1,W))
            mu[0,:] = (X[:,:,np.newaxis]*weights[:,:,np.newaxis]*Y).sum(axis=(0,1))/(X*weights).sum()
            
            # maintenant on calcule sigma
            mut = mu[0,:]
            Y_manip = Y - X[:,:,np.newaxis] * mut[np.newaxis,np.newaxis,:] 
            liste_vec = np.reshape(Y_manip,(Y.shape[0]*Y.shape[1],W))
            Sigma = np.cov(liste_vec[(msk==0),:],rowvar=False, aweights=weights_1d)
            
            sig = np.sqrt(Sigma[np.eye(W)==1].mean())
            rho_1= Sigma[ (np.eye(W,k=-1)+np.eye(W,k=1))==1].mean()
            rho_2= Sigma[ (np.eye(W,k=-2)+np.eye(W,k=2))==1].mean()
 
        
    elif parchamp.multi==True and mono==False: # multiple classes hyperspectral
        # NEEDS UPDATE FOR STD
        nb_classe = x_range.size
        
        # compute the mean mu
        mu = np.zeros(shape=(W,nb_classe)) # attention forme differente ici que plus bas
        for id_x in range(nb_classe):
             mask = (X == x_range[id_x])
             mu[:,id_x] = (mask[:,:,np.newaxis]*Y*weights[:,:,np.newaxis]).sum(axis=(0,1))/(mask*weights).sum()
    
    elif parchamp.multi==True and mono==True:   # mono bande multi classe   
    
        nb_classe = x_range.size
         
        # compute the mean mu
        mu = np.zeros(shape=(nb_classe,1))
        for id_x in range(nb_classe):
             mask = (X == x_range[id_x])
             Y_manip = Y[:,:,0]
             mu[id_x] = Y_manip[mask].mean()
             # ajouter la possibilite d'avoir zero instance
        
        # compute the STD sig
        sig = np.zeros(shape=(nb_classe))
        
        nb_classe = x_range.size
        for id_x in range(nb_classe):
             mask = (X == x_range[id_x])
        
             Y_manip = Y[:,:,0] -  mu[id_x] * mask
        
             sig[id_x] = np.std(Y_manip[mask].flatten())
                 
    # Retrieving

    parchamp.mu = mu
    parchamp.sig= sig     
    
    if mono==False:
        parchamp.rho_1 = rho_1
        parchamp.rho_2 = rho_2
    

    return parchamp   
    
def est_pi(X,V,pargibbs):
    """ 
    Estimation of the prior parameter pi.
    
    :param ndarray X: "hidden" classification
    :param ndarray V: "hidden" orientations
    :param misc pargibbs: parameters of the Gibbs sampling.

    :returns: **pi_est** *(ndarray)* - Estimated values of pi.
    """
    S0 = pargibbs.S0
    S1 = pargibbs.S1  
    
    # Pour V déjà :
    vals_vois = ft.get_vals_voisins_tout(V)   
    iseq = (V[:,:,np.newaxis]== vals_vois)
    iseq = iseq[1:-1,1:-1,:]
    
    pi_est=np.zeros(shape=(2,9)) # 2 champs, 9 types de config.
   
    energies =1-2*iseq
    energies_sum = energies.sum(axis=2)
    for a in range(9) :       
        proba_empirique = (iseq.sum(axis=2)==a).mean()
        msk_a = (iseq.sum(axis=2)==a)
        if msk_a.sum()!=0:
#            energie_config = energies_sum[msk_a]#/msk_a.sum()
#            denom = np.exp(energies_sum[msk_a]).sum()/msk_a.sum() # proba moyenne de v_s|v_ns
            secondterme  = np.exp(-energies_sum[msk_a]).mean()
#            denom = np.exp(-energies_sum).mean()
            pi_est[1,a] = proba_empirique * secondterme#/denom#np.exp(energie_config)
            
        else:
            pi_est[1,a] = 0
    
   
    # Pour X maintenant :
    if pargibbs.multi:
       X_nn = np.copy(X)  # il faudra bien réfléchir à ce qu'on fait dans le cas d'un mélange...
    else:
       X_nn = (X>0) # carte de X non nul - toute les fractions de 1 sont vues comme 1

    # A changer pour le multi-classe !!!
       
    vals_vois = ft.get_vals_voisins_tout(X_nn)   
    iseq = (X_nn[:,:,np.newaxis]== vals_vois)
    iseq = iseq[1:-1,1:-1,:]

    # Ponderation des voisinages
    phi_theta = np.ones_like(pargibbs.Vois)    
    for i in xrange(S0):    
        for j in xrange(S1):
            phi_theta[i,j,:] =  ft.gen_beta(pargibbs.Vois[i,j,:],V[i,j])   
            
    phi_theta = phi_theta[1:-1,1:-1,:]
    
    energies_avec_v = phi_theta * (1 - 2*iseq)
    energies_tous_sum = ( energies_avec_v).sum(axis=2)
    
    
    for a in range(9) :       
        proba_empirique = (iseq.sum(axis=2)==a).mean()
        
        msk_a = (iseq.sum(axis=2)==a)
        if msk_a.sum()!=0:
#            energie_config = energies_tous_sum[msk_a].sum()/msk_a.sum()
#            denom = np.exp(energies_tous_sum[msk_a]).sum()/msk_a.sum()
            secondterme  = np.exp(-energies_tous_sum[msk_a]).mean()
            pi_est[0,a] = proba_empirique*secondterme#/denom#np.exp(energie_config)
            
        else:
            pi_est[0,a] = 0
        
    pi_est[0,:]/= pi_est[0,~np.isnan(pi_est[0,:])].sum()
    pi_est[1,:]/= pi_est[1,~np.isnan(pi_est[1,:])].sum()
    
    # Etape de smoothing
    pi_est[np.isnan(pi_est)+(pi_est==0)] = 0.01/(S0*S1)
    
    pi_est[0,:]/= pi_est[0,~np.isnan(pi_est[0,:])].sum()
    pi_est[1,:]/= pi_est[1,~np.isnan(pi_est[1,:])].sum() # juste mais negligeable
    
    return pi_est
  


def est_param_de_x(X):
    """
    Estimation of the :math:`\\alpha` parameter from a realization :math:`X=x`.
    
    :param ndarray X: Values taken by :math:`x`


    :returns: **alpha** *(float)* - estimation of the :math:`\\alpha` parameter
    """

    vals_vois = ft.get_vals_voisins_tout(X)   
    iseq = (X[:,:,np.newaxis]== vals_vois)
    iseq = iseq[1:-1,1:-1,:]
    
    
    #alpha_tous = np.zeros(9)+np.nan
    facteur = np.zeros(9)
    en = np.zeros(9)
            
    pchaps = np.zeros(9)     
    for a in range(9) :       
        pchaps[a] = (iseq.sum(axis=2)==a).mean()
        
        if (iseq.sum(axis=2)==a).sum() < 20:
            pchaps[a] = 0
        
        
        energies_sans_alpha = (1 - 2*iseq)
        msk_a = (iseq.sum(axis=2)==a)
        en[a] = (energies_sans_alpha * msk_a[:,:,np.newaxis]).sum() / msk_a.sum()
        
    ratios = pchaps[np.newaxis,:] / pchaps[:,np.newaxis]   
    ran = np.arange(9) 
    facteur = ran[np.newaxis,:] - ran[:,np.newaxis] 
    correc_en = en[np.newaxis,:] / en[:,np.newaxis]  
    
    logratios = np.log(ratios)
    
    a = logratios/(facteur) * correc_en
    alpha_tous = a[(np.isnan(a)==0)*(np.isinf(a)==0)*(a!=0)]
    
    alpha = alpha_tous.mean()
    
#    alpha = max(alpha,0.5)
    return alpha  