# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 09:45:48 2015

@author: courbot
"""



import numpy as np 
import time

import multiprocessing as mp

import parameters
#import image_tools as it
import gibbs_sampler as gs

import SEM as sem


import gc

import matplotlib.pyplot as plt




def serie_gibbs(pargibbs,nb_rea,generate_v,generate_x,use_y,use_pi,tmf=True):
    
#    nb_iter = pargibbs.nb_iter  
    

#    print tmf
    V_tous = np.zeros(shape=(pargibbs.S0,pargibbs.S1,nb_rea))  
    X_tous = np.zeros(shape=(pargibbs.S0,pargibbs.S1,nb_rea))  
    

    normal = False
#
    
    nb_proc = np.minimum(mp.cpu_count(),31)
    pool = mp.Pool(processes=nb_proc-1,maxtasksperchild=1)
    results = {}
    results = [pool.apply_async(gs.gen_champs_fast,args=(pargibbs,generate_v,generate_x,use_y,normal,use_pi)) for i in range(nb_rea)]
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

# version non parrellèle pour debug
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
 
#%%


def seg_otmf(parseg,pargibbs,superv=False,disp=False):
   

    nb_iter_mpm = parseg.nb_iter_mpm
    nb_rea = parseg.nb_rea
    incert=parseg.incert
#    tmf=parseg.tmf
    
    
#    pargibbs.thr_conf = 1./(pargibbs.S1*pargibbs.S0)
#    spec_snr=parseg.spec_snr#False # pour l'instant !


    parchamp = parameters.ParamsChamps()
    v_help = True
    pargibbs.nb_nn_v_help = 1 # param a supprimer
    pargibbs.v_help=v_help # useless? no !
    parchamp.multi = parseg.multi
    parchamp.spec_snr = parseg.spec_snr
    if parchamp.spec_snr:
        parchamp.facteur=parseg.facteur
    parchamp.weights = parseg.weights
    pargibbs.multi = parchamp.multi
    
    if superv==False:
        if disp :
            print 'SEM ...'
        start=time.time()
        parsem = sem.SEM(parseg,parchamp,pargibbs) #!!!
        temps =  (time.time()-start)
        nb_iter_effectif = parsem.sig_sem.shape[0]-1
        if disp:
            print '     %.0f x %.0f iterations et %.2f s - %.3f s/iter.'%(nb_iter_effectif,pargibbs.nb_iter, temps,temps/nb_iter_effectif  )
        pargibbs.parchamp = parsem
        pargibbs.parchamp.mu = pargibbs.parchamp.mu.T

    else:
        parsem = parseg.real_par
        pargibbs.parchamp = parsem
        pargibbs.parchamp.mu = pargibbs.parchamp.mu.T
    # 1b) SEM retrieving
    

    
    
    #%%
    if parseg.mpm:
            # 2) MPM
        #    print pargibbs.mu.shape
            nb_rea_mpm = nb_iter_mpm
            pargibbs.nb_iter = nb_rea_mpm
        
            if disp:
                print 'Serie Gibbs...'
            start=time.time()
            if parseg.tmf ==True:
                pargibbs=serie_gibbs(pargibbs,nb_rea,generate_v=True,generate_x=True,use_y=True,use_pi = True,tmf=True)
            else:
                pargibbs.phi_theta_0 =0.
                pargibbs.V = np.zeros(shape=(pargibbs.S0,pargibbs.S1))#(pargibbs.X_res[:,:,-1])
                pargibbs=serie_gibbs(pargibbs,nb_rea,generate_v=False,generate_x=True,use_y=True,use_pi = True,tmf=False)
                pargibbs.V_res = np.zeros_like(pargibbs.X_res)
        
            
            temps =  (time.time()-start) 
            if disp:
                print 'Serie simu : %.0f iterations et %.2f s - %.3f s/iter.'%(nb_rea_mpm, temps,temps/nb_rea_mpm  )    
            if incert==False:  
                if parseg.tmf==True:
                    X_mpm_est,V_mpm_est,X_mpm,V_mpm = MPM(pargibbs)
                else:
#                    print 'attention ! souci de code dans SEG_OTMF ligne 180 env.'
                    X_mpm_est = (pargibbs.X_res.mean(axis=2)>0.25).astype(float) # c'est quoi ce 0.25 ??
                    V_mpm_est = np.zeros_like(X_mpm_est)
                    
                    X_mpm, V_mpm = pargibbs.X_res,pargibbs.V_res
        
            else:
                X_mpm_est,V_mpm_est,X_mpm,V_mpm = MPM_uncert(pargibbs,parseg.tmf)
                
            X_est, V_est = X_mpm_est, V_mpm_est
                
    else:
        # MAP approché par ICM
#        pargibbs.autoconv = False
#        pargibbs.nb_iter = 500
#        pargibbs.thr_conf = 1./(pargibbs.S1*pargibbs.S0)
        if parseg.tmf==True:
            pargibbs = gs.gen_champs_fast(pargibbs,generate_v=True,generate_x=True,use_y=True,normal=False,use_pi=True,icm=True)
        else:
            pargibbs.phi_theta_0 =0.
            pargibbs.V = np.zeros(shape=(pargibbs.S0,pargibbs.S1))#(pargibbs.X_res[:,:,-1])
            pargibbs= gs.gen_champs_fast(pargibbs,generate_v=False,generate_x=True,use_y=True,normal=False,use_pi=True,icm=True)
            pargibbs.V_res = np.zeros_like(pargibbs.X_res)
    
        X_est = pargibbs.X_res[:,:,-1]
        V_est = pargibbs.V_res[:,:,-1]
        
        X_mpm = np.zeros_like(X_est)
        V_mpm = np.zeros_like(V_est)

  
    return X_est, V_est, X_mpm,V_mpm, parsem




    
def MPM(pargibbs):
    """Parralel MPM algo"""
    print 'attention: fonction ancienne (MPM) a eviter...'
    #1) built numerous simumations
   
    X_mpm, V_mpm = pargibbs.X_res, pargibbs.V_res
    
    v_range, x_range = pargibbs.v_range, pargibbs.x_range
    
    # 2)  Estimate frequencies
    freqs = np.zeros(shape=(pargibbs.S0,pargibbs.S1,x_range.size*v_range.size))
    freqs_sep = np.zeros(shape=(pargibbs.S0,pargibbs.S1,pargibbs.x_range.size,pargibbs.v_range.size))
    for id_x in range(x_range.size) :
        for v in range(v_range.size):
            freqs[:,:,id_x*v_range.size+v] = ((X_mpm==x_range[id_x])*(V_mpm==v_range[v])).sum(axis=2)
            freqs_sep[:,:,id_x,v] = freqs[:,:,id_x*v_range.size+v]  
            
#    for id_x in range(x_range.size) :
#        for v in range(v_range.size):
#            freqs_sep[:,:,id_x,v] = freqs[:,:,id_x*v_range.size+v]     
    
    
    # 3) get the most frequent mode
    mode_x = np.argmax(freqs_sep.sum(axis=3),axis=2)
    X_mpm_est = x_range[mode_x]
    
    mode_v = np.argmax(freqs_sep.sum(axis=2),axis=2)
    V_mpm_est = v_range[mode_v]


    
    return X_mpm_est,V_mpm_est,X_mpm,V_mpm    
    

    
def MPM_uncert(pargibbs,tmf):
    """Parralel MPM algo"""
    #1) built numerous simumations
 
    
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
        
        #Ux_map_noz = freqs_mpm_x-freqs_marg_x[:,:,0]
        #Ux_map[X_mpm_est!=0] = Ux_map_noz[X_mpm_est!=0]
#    Ux_map *=2 # c'est comme si on avait 2 classes !
    

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
        
    # now constructing map of differences between probabilities
#    if x_range.size==2:    
#        Ux_map = np.amax(freqs_mpm_x[:,:,np.newaxis]-freqs_marg_x,axis=2)
#    elif x_range.size>2:# il faudra preciser ceci pour le vrai multi classe    
#        Ux_map = np.amax(freqs_mpm_x[:,:,np.newaxis]-freqs_marg_x,axis=2)
#        Ux_map_noz = freqs_mpm_x-freqs_marg_x[:,:,0]
#        Ux_map[X_mpm_est!=0] = Ux_map_noz[X_mpm_est!=0]
#        
#        

    
    

    
    return X_mpm_est,V_mpm_est,Ux_map,Uv_map
    
    
#    
#    
#    
#
#def seg_otmf_atten(pargibbs,nb_iter_sem,nb_iter_mpm,nb_rea,seuil_conv, taille_fen,snr_att,incert=False,longueur_mpm=1):
#
#    
#    
#    
#    parchamp = parameters.ParamsChamps()
#
#    
#    print 'SEM ...'
#    start=time.time()
#    parsem = sem.SEM(parchamp,pargibbs,nb_iter_sem,seuil_conv, taille_fen,est_alpha=True)
#    temps =  (time.time()-start)
#    nb_iter_effectif = parsem.rho1_sem.size-1
#    print '     %.0f x %.0f iterations et %.2f s - %.3f s/iter.'%(nb_iter_effectif,pargibbs.nb_iter, temps,temps/nb_iter_effectif  )
#    
#    # 1b) SEM retrieving
#    #%%    
#    pargibbs.sig=parsem.sig
#    pargibbs.rho_1=parsem.rho_1
#    pargibbs.rho_2=parsem.rho_2  
#    pargibbs.pi = parsem.pi
#    pargibbs.mu = parsem.mu
#    
#    #%%
#    # 2) MPM
#    levels_att = np.sqrt(pargibbs.W)*pargibbs.sig/np.linalg.norm(pargibbs.mu) * 10**(snr_att/20.)
#
#
#    X_mpm_att=np.zeros(shape=(pargibbs.S0,pargibbs.S1,levels_att.size))
#    V_mpm_att=np.zeros(shape=(pargibbs.S0,pargibbs.S1,levels_att.size))
#    for f in range(levels_att.size): 
#            pargibbs.mu = parsem.mu * levels_att[f]
#            
#            
#            nb_rea_mpm = nb_iter_mpm
#            pargibbs.nb_iter = nb_rea_mpm
#            pargibbs.autoconv==True
#        #    pargibbs.v_help =True
#        #    
#            print 'Serie Gibbs...'
#            start=time.time()
#            
#            pargibbs=serie_gibbs(pargibbs,nb_rea,generate_v=True,generate_x=True,use_y=True,use_pi = True,fin=(nb_rea_mpm-longueur_mpm))
#        #    pargibbs = gs.gen_champs_fast(pargibbs,generate_v=True,generate_x=True,use_y=True,use_pi=True) 
#        #    pargibbs.
#            temps =  (time.time()-start) 
#            print 'Serie simu : %.0f iterations et %.2f s - %.3f s/iter.'%(nb_rea_mpm, temps,temps/nb_rea_mpm  )    
#                
#            X_mpm_est,V_mpm_est,X_mpm,V_mpm = MPM(pargibbs,lim=0,incert=incert)
#            
#            X_mpm_att[:,:,f] = X_mpm_est
#            V_mpm_att[:,:,f] = V_mpm_est
##  
##    
#    return X_mpm_att,V_mpm_att, parsem
#
