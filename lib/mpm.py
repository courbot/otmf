#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 16:15:53 2018

@author: courbot
"""
import numpy as np
from otmf.gibbs_sampler import gen_champs_fast
import gc
import multiprocessing as mp
def serie_gibbs(pargibbs,nb_rea,generate_v,generate_x,use_y,use_pi,tmf=True):
    """ Generate a serie of Gibbs sampling using the same parameters.
    
        This functions uses multiprocessing.
        
    :param misc pargibbs: parameters of the Gibbs sampling.    
    :param int nb_rea: set how many independant sampling there will be.       
    :returns: **pargibbs** *(parameter)* parameters containing the Gibbs samples.
    
    """ 

    V_tous = np.zeros(shape=(pargibbs.S0,pargibbs.S1,nb_rea))  
    X_tous = np.zeros(shape=(pargibbs.S0,pargibbs.S1,nb_rea))  
    

    normal = False
#
    
    nb_proc = np.minimum(mp.cpu_count(),31)
    pool = mp.Pool(processes=nb_proc-1,maxtasksperchild=1)
    results = {}
    results = [pool.apply_async(gen_champs_fast,args=(pargibbs,generate_v,generate_x,use_y,normal,use_pi)) for i in range(nb_rea)]
    output = [p.get() for p in results]
    pool.close()
    pool.join()
    
    del results
    gc.collect()
    
    for i in range(nb_rea):
        outi = output[i]
        X_tous[:,:,i] = outi.X_res[:,:,-1]
        if tmf==True:
                V_tous[:,:,i] = outi.V_res[:,:,-1]
    del output
    
    gc.collect()

    # non-parralel version, to use for debugging
    #    for i in range(nb_rea):    
    #        parvx = gs.gen_champs_fast(pargibbs,generate_v,generate_x,use_y,normal,use_pi)    
    #        V_tous[:,:,i] = parvx.V_res[:,:,-1]
    #        X_tous[:,:,i] = parvx.X_res[:,:,-1]
    #    
    #    
    #    
    pargibbs.X_res = X_tous
    if tmf==True:
        pargibbs.V_res = V_tous

    return pargibbs





def MPM_uncert(pargibbs,tmf):
    """MPM segmentation method. Used in parrallel processing here.
    
    :param parameter pargibbs: parameters of the Gibbs sampling.    
    :param bool tmf: set if we are in the Triplet Markov Field [True] of Hidden
                    Markov Field [False].

    :returns: **X_mpm_est** *(ndarray)* MPM estimation of X
    :returns: **V_mpm_est** *(ndarray)* MPM estimation of V
    :returns: **Ux_map** *(ndarray)*  Uncertainty map, supplementing the X segmentation.
    :returns: **Uv_map** *(ndarray)*  Uncertainty map, supplementing the V segmentation.
    
    """
    #1) built numerous simulations
    X_mpm = pargibbs.X_res
    x_range=pargibbs.x_range
    
    if tmf==True:
        V_mpm = pargibbs.V_res
        v_range=pargibbs.v_range

    
    
    # 2)  Estimate frequencies
    if tmf == True:
        freqs = np.zeros(shape=(pargibbs.S0,pargibbs.S1,x_range.size*v_range.size))
        freqs_sep = np.zeros(shape=(pargibbs.S0,pargibbs.S1,pargibbs.x_range.size,pargibbs.v_range.size))
        for id_x in range(x_range.size) :
            for v in range(v_range.size):
                freqs[:,:,id_x*v_range.size+v] = ((X_mpm==x_range[id_x])*(V_mpm==v_range[v])).astype(float).mean(axis=2)
                freqs_sep[:,:,id_x,v] = freqs[:,:,id_x*v_range.size+v]  
        
        # 3) get the most frequent mode    
        mode_x = np.argmax(freqs_sep.sum(axis=3),axis=2)
        X_mpm_est = x_range[mode_x]
        
        mode_v = np.argmax(freqs_sep.sum(axis=2),axis=2)
        V_mpm_est = v_range[mode_v]
        
    else:
        freqs = np.zeros(shape=(pargibbs.S0,pargibbs.S1,x_range.size))
        for id_x in range(x_range.size) :
                
                freqs[:,:,id_x] = (X_mpm==x_range[id_x]).astype(float).mean(axis=2)

                
        # 3) get the most frequent mode    
                
        mode_x = np.argmax(freqs,axis=2)
        X_mpm_est = x_range[mode_x]
        V_mpm_est = np.zeros_like(X_mpm_est)
        
 
    
    Ux_map = np.zeros_like(X_mpm_est)
    Uv_map = np.zeros_like(X_mpm_est)

    Xi = pargibbs.Xi

    # recast dans des dimensions pratiques

    if tmf == True:
        freqs_marg_x = freqs_sep.sum(axis=3)   
    else:
        freqs_marg_x = np.copy(freqs)
        
    freqs_mpm_x =  np.amax(freqs_marg_x,axis=2)
    ratios_x = freqs_mpm_x[:,:,np.newaxis]/freqs_marg_x
    
    ineq_somme_x = (ratios_x < 1+Xi).sum(axis=2)
    incert_map_x = ineq_somme_x > 1 # car il y a une valeur qui vaut 1 dans la somme
    X_mpm_est[incert_map_x] +=np.nan
    
    freqs_diff = freqs_mpm_x[:,:,np.newaxis]-freqs_marg_x

    freqs_diff[freqs_diff==0] = 10000 #nombre plus grand que le min !
    #freqs_diff[freqs_diff>10000] = 10001 # pour les inf
    
    if x_range.size==2:
        Ux_map = np.amin(freqs_diff,axis=2)  
        Ux_map[Ux_map ==10000] = 0
    else:
        Ux_map = np.amin(freqs_diff,axis=2)  
        

    if tmf == True:
        freqs_marg_v = freqs_sep.sum(axis=2)
        freqs_mpm_v =  np.amax(freqs_marg_v,axis=2)
        ratios_v = freqs_mpm_v[:,:,np.newaxis]/freqs_marg_v
        ineq_somme_v= (ratios_v < 1+Xi).sum(axis=2)
        incert_map_v = ineq_somme_v > 1 
        
   
        V_mpm_est[incert_map_v] +=np.nan
        Uv_map = np.amax(freqs_mpm_v[:,:,np.newaxis]-freqs_marg_v ,axis=2)
        
    else:
        Uv_map = np.zeros_like(Ux_map)

    return X_mpm_est,V_mpm_est,Ux_map,Uv_map
    