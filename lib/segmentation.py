# -*- coding: utf-8 -*-
"""

This module contains the image segmentation functions.

:author: Jean-Baptiste Courbot - jb.courbot@unistra.fr
:date: Sep 01, 2017 (created Nov 03, 2015)
"""

import numpy as np 
import time


from otmf.parameters import ParamsChamps
from otmf.gibbs_sampler import gen_champs_fast
from otmf.parameter_estimation import SEM
import otmf.mpm as mpm


 
#%%


def seg_otmf(parseg,pargibbs,superv=False,disp=False):
    """
    Segmentation of the image/ hyperspectral image.
    
    
    :param misc parseg: parameters ruling the segmentation method
    :param misc pargibbs: parameters of the Gibbs sampling.
    :param bool superv: set if the segmentation is supervized or not.
    :param bool disp: trigger the verbose mode.
    
    
    :returns: **X_est** *(ndarray)* X segmentation 
    :returns: **V_est** *(ndarray)* V segmentation 
    :returns: **X_mpm** *(ndarray)* Sequence of X Gibbs samples used for the MPM segmentation.
    :returns: **V_mpm** *(ndarray)* Sequence of X Gibbs samples used for the MPM segmentation.
    :returns: **parsem** *(parameter)* Parameters estimated with the SEM method.
    """ 

    nb_iter_mpm = parseg.nb_iter_mpm
    nb_rea = parseg.nb_rea
#    incert=parseg.incert

    parchamp = ParamsChamps()
    v_help = True
    pargibbs.nb_nn_v_help = 1 # param a supprimer
    pargibbs.v_help=v_help # useless? no !
    parchamp.multi = parseg.multi
    
    parchamp.weights = parseg.weights
    pargibbs.multi = parchamp.multi
    
    
    # A. Parameter retrieving    
    
    if superv==False:
        if disp :
            print 'SEM ...'
            start=time.time()

        # Unsupervized parameter estimation with SEM
        parsem = SEM(parseg,parchamp,pargibbs) #!!!

        nb_iter_effectif = parsem.sig_sem.shape[0]-1
        if disp:
            temps =  (time.time()-start)
            print '     %.0f x %.0f iterations et %.2f s - %.3f s/iter.'%(nb_iter_effectif,pargibbs.nb_iter, temps,temps/nb_iter_effectif  )
        pargibbs.parchamp = parsem
        pargibbs.parchamp.mu = pargibbs.parchamp.mu.T

    else:
        parsem = parseg.real_par
        pargibbs.parchamp = parsem
        pargibbs.parchamp.mu = pargibbs.parchamp.mu.T


    # B. Segmentation 
    if parseg.mpm:
            # B 1)  if the estimator is the MPM
            
            pargibbs.nb_iter = nb_iter_mpm
        
            if disp:
                print 'Serie Gibbs...'
            start=time.time()
            if parseg.tmf ==True:
                pargibbs=mpm.serie_gibbs(pargibbs,nb_rea,generate_v=True,generate_x=True,use_y=True,use_pi = True,tmf=True)
            else:
                pargibbs.phi_theta_0 =0.
                pargibbs.V = np.zeros(shape=(pargibbs.S0,pargibbs.S1))#(pargibbs.X_res[:,:,-1])
                pargibbs=mpm.serie_gibbs(pargibbs,nb_rea,generate_v=False,generate_x=True,use_y=True,use_pi = True,tmf=False)
                pargibbs.V_res = np.zeros_like(pargibbs.X_res)
        
            
            temps =  (time.time()-start) 
            if disp:
                print 'Serie simu : %.0f iterations et %.2f s - %.3f s/iter.'%(nb_iter_mpm, temps,temps/nb_iter_mpm  )    

            X_mpm_est,V_mpm_est,X_mpm,V_mpm = mpm.MPM_uncert(pargibbs,parseg.tmf)
                
            X_est, V_est = X_mpm_est, V_mpm_est
                
    else:
        # B 2) if the estimator is the MAP, we approach it by ICM

        if parseg.tmf==True:
            pargibbs = gen_champs_fast(pargibbs,generate_v=True,generate_x=True,use_y=True,normal=False,use_pi=True,icm=True)
        else:
            pargibbs.phi_theta_0 =0.
            pargibbs.V = np.zeros(shape=(pargibbs.S0,pargibbs.S1))#(pargibbs.X_res[:,:,-1])
            pargibbs= gen_champs_fast(pargibbs,generate_v=False,generate_x=True,use_y=True,normal=False,use_pi=True,icm=True)
            pargibbs.V_res = np.zeros_like(pargibbs.X_res)
    
        X_est = pargibbs.X_res[:,:,-1]
        V_est = pargibbs.V_res[:,:,-1]
        
        X_mpm = np.zeros_like(X_est)
        V_mpm = np.zeros_like(V_est)

  
    return X_est, V_est, X_mpm,V_mpm, parsem

 
