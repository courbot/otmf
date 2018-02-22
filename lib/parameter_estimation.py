# -*- coding: utf-8 -*-
"""
This module contains all the functions related to parameter estimation.

:author: Jean-Baptiste Courbot - jb.courbot@unistra.fr
:date: Sep 01, 2017 (created Apr 20, 2016)
"""
import numpy as np 
import scipy.cluster.vq as cvq
from scipy.ndimage.filters import gaussian_filter 
import scipy.ndimage.morphology as morph

from otmf.fields_tools import get_vals_voisins_tout,gen_beta
from otmf.gibbs_sampler import get_dir

from otmf import mpm as mpm

def SEM(parseg,parchamp,pargibbs,disp=False):
    """ Stochastic Expectation-Maximization algorithm.
    
    This iterative method allows, at each iteration, a parameter re-estimation 
    directly from complete data thanks to realization of the missing data, 
    simulated using parameters estimated at the previous step.
    
    See the original works from Celeux and Diebolt.
    

    :param parameters parseg: parameters ruling the segmentation method
    :param parameters parchamp: parameters of the model (priors, noise parameters)
    :param parameters pargibbs: parameters of the Gibbs sampling.
    :param bool disp: set the verbose mode [False/True]

    :returns: **parchamp** *(parameter)* - set of estimated parameters
    """
    Y = pargibbs.Y
    W = pargibbs.W

    nb_classe = pargibbs.x_range.size
  
    # recuperation parametres de segmentation
    nb_iter_sem, seuil_conv, taille_fen = parseg.nb_iter_sem, parseg.seuil_conv,parseg.taille_fen

    x_range=pargibbs.x_range

    #==============================================================================
    #   Values initialization
    #==============================================================================
    sig_sem = np.zeros(shape=(nb_iter_sem,nb_classe))
    alpha_sem = np.zeros(shape=(nb_iter_sem))
    alpha_v_sem = np.zeros(shape=(nb_iter_sem))

    mu_sem = np.zeros(shape=(nb_iter_sem,W,nb_classe)) ;    
  
    pi_sem =  np.zeros(shape=(nb_iter_sem,2,9)) # deux champs, 9 types de config

        
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
        
        parchamp,X_courant,V_courant = init_params(pargibbs,parchamp)

    if disp:
        print '  SEM iter...'
    for iter_sem in xrange(nb_iter_sem):

        i = iter_sem

        pargibbs.parchamp = parchamp

        if parseg.tmf==True:
            
            pargibbs.V = np.zeros_like(X_courant)
            
            nb_iter_to_keep = np.copy(pargibbs.nb_iter)
            pargibbs.nb_iter = 100
            pargibbs=mpm.serie_gibbs(pargibbs,nb_rea=int(parseg.nb_iter_serie_sem),generate_v=True,generate_x=True,use_y=True,use_pi=True,tmf=True)#(pargibbs,nb_rea=int(parseg.nb_iter_serie_sem))
            pargibbs.nb_iter = np.copy(nb_iter_to_keep)        
            
            X_courant, V_courant, d, d = mpm.MPM_uncert(pargibbs, tmf=False)
            
            if (X_courant.sum() == 0) or ((1.-X_courant).sum()==0):
                X_courant = np.random.random(size=X_courant.shape) > 0.5
        else:
            
            # a) Simulation given the previous parameter
           
            pargibbs.V = np.zeros_like(X_courant)
            
            nb_iter_to_keep = np.copy(pargibbs.nb_iter)
            pargibbs.nb_iter = 100
            pargibbs=mpm.serie_gibbs(pargibbs,nb_rea=int(parseg.nb_iter_serie_sem),generate_v=False,generate_x=True,use_y=True,use_pi=True,tmf=False)#(pargibbs,nb_rea=int(parseg.nb_iter_serie_sem))
            pargibbs.nb_iter = np.copy(nb_iter_to_keep)        
            
            X_courant, d, d, d = mpm.MPM_uncert(pargibbs, tmf=False)
            
   
            
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
        

        mu_sem[iter_sem,:,:], sig_sem[iter_sem,:], pi_sem[iter_sem,:,:], alpha_sem[iter_sem], alpha_v_sem[iter_sem]= parchamp.mu.T, parchamp.sig, parchamp.pi,parchamp.alpha,parchamp.alpha_v
        

        #==============================================================================
        #       Decision to stop or continue
        #==============================================================================
        if i > taille_fen:
            ecart_tous[i-taille_fen,:] = mesure_ecart(sig_sem[:i,:],sig_sem[i,:], mu_sem[:i,:],mu_sem[i,:],pi_sem[:i,:,:],pi_sem[i,:,:],taille_fen,W)
              
        if (ecart_tous[i-taille_fen,:]<seuil_conv).all() == True:
              if disp:
                  print 'stop iter %.0f'%i
              # Trucation of parameter arrays
              mu_sem, pi_sem, sig_sem = mu_sem[:i+1,:], pi_sem[:i+1,:,:], sig_sem[:i+1]
        
              sig_sem = sig_sem[:i+1]
              break

    #==============================================================================
    #   Stockage des parametres
    #==============================================================================
    parchamp.mu_sem, parchamp.sig_sem, parchamp.pi_sem = mu_sem, sig_sem, pi_sem

    #==============================================================================
    #   moyenne des derniers parametres
    #==============================================================================
    
    parchamp.mu = mu_sem[-taille_fen:-1,:,:].mean(axis=0)
    parchamp.sig = sig_sem[-taille_fen:-1,:].mean(axis=0)
    parchamp.alpha = alpha_sem[-taille_fen:-1].mean()
    parchamp.alpha_v = alpha_v_sem[-taille_fen:-1].mean()
    parchamp.pi = pi_sem[-taille_fen:-1,:,:].mean(axis=0)

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


    nb_classe = x_range.size
     
    # compute the means mu
    mu = np.zeros(shape=(nb_classe,1))
    for id_x in range(nb_classe):
         mask = (X == x_range[id_x])
         Y_manip = Y[:,:,0]
         mu[id_x] = Y_manip[mask].mean()

    
    # compute the standard deviations sig
    sig = np.zeros(shape=(nb_classe))
    
    nb_classe = x_range.size
    for id_x in range(nb_classe):
         mask = (X == x_range[id_x])
    
         Y_manip = Y[:,:,0] -  mu[id_x] * mask
    
         sig[id_x] = np.std(Y_manip[mask].flatten())

    parchamp.mu = mu
    parchamp.sig= sig     

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
    vals_vois = get_vals_voisins_tout(V)   
    iseq = (V[:,:,np.newaxis]== vals_vois)
    iseq = iseq[1:-1,1:-1,:]
    
    pi_est=np.zeros(shape=(2,9)) # 2 fields, 9 possible configurations
   
    # First let us evaluate the terms related to V :
    energies =1-2*iseq
    energies_sum = energies.sum(axis=2)
    for a in range(9) :       
        proba_empirique = (iseq.sum(axis=2)==a).mean()
        msk_a = (iseq.sum(axis=2)==a)
        if msk_a.sum()!=0:

            secondterme  = np.exp(-energies_sum[msk_a]).mean()

            pi_est[1,a] = proba_empirique * secondterme
            
        else:
            pi_est[1,a] = 0
    
   
    # Now let us evaluate the terms related to X :

    vals_vois = get_vals_voisins_tout(X)   
    iseq = (X[:,:,np.newaxis]== vals_vois)
    iseq = iseq[1:-1,1:-1,:]

    # Ponderation des voisinages
    phi_theta = np.ones_like(pargibbs.Vois)    
    for i in xrange(S0):    
        for j in xrange(S1):
            phi_theta[i,j,:] =  gen_beta(pargibbs.Vois[i,j,:],V[i,j])   
            
    phi_theta = phi_theta[1:-1,1:-1,:]
    
    energies_avec_v = phi_theta * (1 - 2*iseq)
    energies_tous_sum = ( energies_avec_v).sum(axis=2)
    
    
    for a in range(9) :       
        proba_empirique = (iseq.sum(axis=2)==a).mean()
        
        msk_a = (iseq.sum(axis=2)==a)
        if msk_a.sum()!=0:
            secondterme  = np.exp(-energies_tous_sum[msk_a]).mean()
            pi_est[0,a] = proba_empirique*secondterme#/denom#np.exp(energie_config)
            
        else:
            pi_est[0,a] = 0
        
    # Finally we normalize so that NaN are ignored :    
    
    pi_est[0,:]/= pi_est[0,~np.isnan(pi_est[0,:])].sum()
    pi_est[1,:]/= pi_est[1,~np.isnan(pi_est[1,:])].sum()
    
    # Etape de smoothing
    pi_est[np.isnan(pi_est)+(pi_est==0)] = 0.01/(S0*S1)
    
    pi_est[0,:]/= pi_est[0,~np.isnan(pi_est[0,:])].sum()
    pi_est[1,:]/= pi_est[1,~np.isnan(pi_est[1,:])].sum()
    # The two latter steps are accurate but of negligible effect on the result.
    
    return pi_est
  


def est_param_de_x(X):
    """
    Estimation of the :math:`\\alpha` parameter from a realization :math:`X=x`.
    
    The methods is adapted from Derin and Elliot's paper.
    
    :param ndarray X: Values taken by :math:`x`


    :returns: **alpha** *(float)* - estimation of the :math:`\\alpha` parameter
    """

    vals_vois = get_vals_voisins_tout(X)   
    iseq = (X[:,:,np.newaxis]== vals_vois)
    iseq = iseq[1:-1,1:-1,:]
    
    

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

    return alpha  



def get_parcov(Sigma):
    """
    Estimate the three parameter ruling a penta-diagonal covariance matrix, 
    with identical values along the diagonals.
    
    
    Parameters
    ----------
    Sigma:ndarray
        Covariance matrix
    
    Returns
    -------
    sigma:float
        Estimated value of the scalar covariance.
        
    rho1:float
        Estimated value of the 1-offset term.
        
    rho2:float
        Estimated value of the 2-offset term.
    
    """
    
    W = Sigma.shape[0]
    sig = np.sqrt(Sigma[np.eye(W)==1].mean())
    rho1= Sigma[ (np.eye(W,k=-1)+np.eye(W,k=1))==1].mean()
    rho2= Sigma[ (np.eye(W,k=-2)+np.eye(W,k=2))==1].mean()
    
    return sig, rho1,rho2

    
def maj_parchamp(parchamp, mu,sig,rho1,rho2,alpha,alpha_v):
    """Set all values as fields of the parchamp class. """
    parchamp.mu = mu
    parchamp.sig = sig
    parchamp.rho_1 = rho1
    parchamp.rho_2 = rho2
    parchamp.alpha = alpha
    parchamp.alpha_v = alpha_v
    
    return parchamp
    
    
def mesure_ecart(A_tout,A, mu_tout,mu,pi_tout, pi,taille_fen,W):
    """
    Measures the gap between parameters and an average of a parameter serie.
    
    Since parameters are inhomogenous, gaps are normalized and computed 
    parameter-wise.

    Parameters
    ----------
    A_tous:ndarray
        Sequence of matrices.
    A:ndarray
        Single matrix.
    mu_tous:ndarray
        Sequence of vectors (mean parameter).
    mu_tous:ndarray
        Single vector.
    pi_tous:ndarray
        Sequence of vectors.
    pi:ndarray
        Single vector.
    taille_fen:int
        Window lenght in which the averaging is performed.
    W:ind
        DEPRECATED.
        
    Returns
    -------
    ecarts:ndarray
        1D array containing relative gaps.
    
    """
    
    ecart_mu = np.linalg.norm(mu-mu_tout[-taille_fen:-1,:].mean(axis=0))/np.linalg.norm(mu_tout[-taille_fen:-1,:].mean(axis=0))
    
    ecart_pi = np.linalg.norm(pi-pi_tout[-taille_fen:-1,:,:].mean(axis=0))/np.linalg.norm(pi_tout[-taille_fen:-1,:,:].mean(axis=0)) 
    
    if W>1:
        ecart_a = np.linalg.norm(A-A_tout[-taille_fen:-1,:,:].mean(axis=0),axis=(0,1))/np.linalg.norm(A_tout[-taille_fen:-1,:,:].mean(axis=0))        
    
        ecarts = np.array([ecart_a,ecart_mu, ecart_pi])
    else:
        ecart_sig = np.linalg.norm(A - A_tout[-taille_fen:-1,:].mean(axis=0))/np.linalg.norm(A_tout[-taille_fen:-1,:].mean(axis=0))
        ecarts = np.array([ecart_sig, ecart_mu, ecart_pi])

    return ecarts
       
def est_kmeans(Y,x_range,multi=False):
    """ 
    Simple routine for running K-means clustering on an hyperspectral image.
    
    Parameters
    ----------
    
    Y:ndarray
        Hyperspectral image aranged as (spatial,spatial,spectral)
    x_range:ndarray
        Classes (clusters) to recover.
    multi:bool
        Set if we search for only 2 classes [False] or not [True].
       
    Returns
    -------
    X_km:ndarray
        Classification image.
        
    """
    S0,S1,W = Y.shape
    nanmap = np.isnan(Y).any(axis=2)
    msk = (nanmap).reshape(Y.shape[0]*Y.shape[1])  
    liste_vec = Y.reshape(S0*S1,W)
    liste_sans_nan = liste_vec[msk==0,:]
    
    centroid, X_init_flat = cvq.kmeans2(liste_sans_nan,x_range.size)

    X_km_flat = np.zeros(shape=(S0*S1))
    X_km_flat[msk==0] = X_init_flat
    X_km_flat[msk==1] += np.nan
    X_km = X_km_flat.reshape((S0,S1))
    if multi==False:
        if (Y*(X_km[:,:,np.newaxis]==x_range[0])).mean() > (Y*(X_km[:,:,np.newaxis]==x_range[1])).mean():
            X_km = 1 - X_km
    X_km /= (x_range.size-1.)
    return X_km

def init_params(pargibbs,parchamp):
    """ Initialize a parameter set

    :param misc pargibbs: parameters of the Gibbs sampling.
    :param misc parchamp: parameters of the model (priors, noise parameters)


    :returns: **parchamp** *(parameter)* - set of estimated parameters
    :returns: **X_courant** *(ndarray)* - instance of X which the parameter are estimated
    """    

    Y = pargibbs.Y
    


    if pargibbs.multi == True: # multiclasse
        X_courant = est_kmeans(Y,pargibbs.x_range,pargibbs.multi)
    else:
        
        X_courant = est_kmeans(Y,np.array([0,1]))
        
        if pargibbs.x_range.size>2:
            # creation d'un X initial continu par morceau
            
            if pargibbs.multi==False:
                x_range = pargibbs.x_range
                X_courant = morph.binary_closing(X_courant,iterations = 1).astype(float)
                pas = x_range[1]-x_range[0]
                X_new = np.zeros_like(X_courant)
                for id_x in range(x_range.size):
                    if id_x ==0:
                        xmax = x_range[id_x]+pas/2.
                        X_new[X_courant < xmax] = x_range[id_x]
                        
                    elif id_x == x_range.size - 1:
                        xmin = x_range[id_x]-pas/2.
                        X_new[X_courant > xmin] = x_range[id_x]
                    else:
                        xmax = x_range[id_x]+pas/2.
                        xmin = x_range[id_x]-pas/2.
                        X_new[(X_courant > xmin)*(X_courant<xmax)] = x_range[id_x]
        
                X_courant = X_new
                
        else:
            X_courant = gaussian_filter(X_courant.astype(float), sigma=(1.0,1.0)) > 0     
 
    V_courant = get_dir(X_courant,pargibbs)

    parchamp = est_param_noise(X_courant,Y,parchamp,pargibbs.x_range)

    parchamp.pi = est_pi(X_courant,V_courant,pargibbs)
    
    parchamp.pi[1,:] = np.exp(np.arange(9.)) / np.exp(np.arange(9.)).sum()

    parchamp.alpha = 1.
    parchamp.alpha_v = 1.

    return parchamp,X_courant,V_courant
 
    
def gen_cov(W,sig,rho_1,rho_2):
    """ Generate a 5-band covariance matrix given the standard deviation + 
    correlation parameters.
    
    :param int W: spectral width of the covariance matrix to create.    
    :param float sig: standard deviation
    :param float rho_1: correlation between two adjacent spectral bands.
    :param float rho_2: correlation between one band and the off-by-two neighbor band.

    :returns: **Sigma** *(ndarray)* - Covariance matrix
    """
    Sigma = np.eye(W)*sig**2 + (np.eye(W,k=1)+np.eye(W,k=-1)) * rho_1 + (np.eye(W,k=2)+np.eye(W,k=-2))*rho_2
    return Sigma
