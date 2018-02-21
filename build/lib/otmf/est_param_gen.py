# -*- coding: utf-8 -*-


import numpy as np 
import scipy.cluster.vq as cvq
from scipy.ndimage.filters import gaussian_filter 
import scipy.ndimage.morphology as morph


from otmf import gibbs_sampler as gs


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
    """ Simple routine for kmeans on HSI"""
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
    """ bla """
#    W = pargibbs.Y.shape[2]
    Y = pargibbs.Y
    


    if pargibbs.multi == True: # multiclasse
        X_courant = est_kmeans(Y,pargibbs.x_range,pargibbs.multi)
    else:
        
        X_courant = est_kmeans(Y,np.array([0,1]))
    #    Y = Y[:,:,0]
        
        if pargibbs.x_range.size>2:
            # creation d'un X initial continu par morceau
            
            if pargibbs.multi==False:
                x_range = pargibbs.x_range
#                X_courant = gaussian_filter(X_courant.astype(float), sigma=(1,1))
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
#             X_courant = morph.binary_closing(X_courant,iterations = 1).astype(float)
        
#    plt.imshow(X_courant)    
    V_courant = gs.get_dir(X_courant,pargibbs)
#    plt.imshow(V_courant)    
#    print pargibbs.x_range
#    pargibbs.V = V_courant
    parchamp = sem.est_param_noise(X_courant,Y,parchamp,pargibbs.x_range)
#    print parchamp.mu
    parchamp.pi = sem.est_pi(X_courant,V_courant,pargibbs)
#    parchamp.pi[0,:] = np.exp(np.arange(9.)) / np.exp(np.arange(9.)).sum()
    parchamp.pi[1,:] = np.exp(np.arange(9.)) / np.exp(np.arange(9.)).sum()

##    print parchamp.mu

    parchamp.alpha = 1.
    parchamp.alpha_v = 1.

    return parchamp,X_courant,V_courant
 
    
def gen_cov(W,sig,rho_1,rho_2):
    """ bla """
    Sigma = np.eye(W)*sig**2 + (np.eye(W,k=1)+np.eye(W,k=-1)) * rho_1 + (np.eye(W,k=2)+np.eye(W,k=-2))*rho_2
    return Sigma
#    
#def gen_obs(pargibbs,X,W, mu,sig,rho_1,rho_2,corrnoise=False):
#    
#    
#    if corrnoise==True:
#        Sigma = gen_cov(W,sig,rho_1,rho_2)   
#        
#    else:
#        Sigma = np.eye(W) * sig**2
#    
#    Y = X[:,:,np.newaxis]*mu[np.newaxis,np.newaxis,:] +np.random.multivariate_normal(mean=np.zeros_like(mu),cov=Sigma,size=(pargibbs.S0,pargibbs.S1))
#    
#
#    
#    pargibbs.Y = Y
#    
#    return pargibbs,Y