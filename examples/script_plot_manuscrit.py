# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 08:43:43 2017

@author: courbot
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma


def gen_exp(experiment,x_range,S0,S1, W, m,sig,rho_1,rho_2):

    if experiment=='2':
        ################ EXPERIMENT 2 :V fixed, X, Y simulated
 
    ##    
#        print('------- (V fixe, X CM)')
        dat = np.load('./data/exp_pamiA.npz')
        X=dat['X'] > 0
#        V=dat['V']
        V = np.pi/4 * np.ones(shape=(128,128))
        V[64:,:] = 3*np.pi/4
        V[:,64:] = (V[:,64:] + np.pi/2)%np.pi
        
       
        
    elif experiment=='3':

    
        dat = np.load('./data/exp_pami3v2.npz')
        X=(dat['X']>0).astype(float)
        V=dat['V']
#        print('------- (X fixe, V inconnu)')

    return X,V











nom_fol = './results/manuscrit/exp'

S0=S1=128


pas = np.pi/2
v_range = np.arange(pas/2., np.pi, pas)
#    else:
#        pas = np.pi/6
#        v_range = np.arange(pas/2., np.pi, pas)


###################
nb_level_x =1
x_range = np.arange(0, 1+1./nb_level_x, 1./nb_level_x)

snr_range = np.arange(-20,7,2.)
#print 'sauvegarde sous ' + nom_can
#%%

### essai, mpm/map, suver/non superv, snr, exp, hmf/otmfx/otmfv


from matplotlib2tikz import save as tikz_save
import time
import gc
#%%
#i=0
##while True :
#    
err_tous = np.zeros(shape=(40,2,2,14,2,3))

for numexp in range(40):
    print numexp
    
    for mpm in (False,True):
        if mpm: n0='' ; id_mpm = 1
        else: n0='map' ; id_mpm = 0
    
   
        for superv in (True,False):
            if superv: n3 = 'known'; id_s = 0
            else: n3='unknown' ; id_s = 1
        
            
            
#            for sigma in (0.5,1.):
            for id_snr in range(snr_range.size):
                snr = snr_range[id_snr]
                n2 = str(int(snr))
#                if sigma==0.5: n2 = '05'; 
#                else: n2='10'
                
                
    
                
                for experiment in ('2','3'):#('2','3'):
                    if experiment=='3': n1 = 'b'; id_e = 0;
                    else: n1='a' ; id_e = 1
    
              
                    nom_can = nom_fol+n0+n1+'_snr'+n2+'_'+n3
#                    print 'sauvegarde sous ' + nom_can
                
                
                
    
                
#                    print "################# Exp. %.0f"%numexp
                    
                    try:

                
                
                        dat = np.load(nom_can+str(numexp)+'.npz')
                        X,V = gen_exp(experiment,x_range,S0,S1,W=1,m=0,sig=1,rho_1=1,rho_2=1)
                        
                        
                        Y = dat.f.Y
                        X_tmf = dat.f.X_mpm_est
                        V_tmf = dat.f.V_mpm_est
                        Ux_map = dat.f.Ux_map
                        Uv_map = dat.f.Uv_map
    #                        parsem = dat.f.parsem,
                        
                        X_hmf = dat.f.X_mpm_hmf
    #                         Ux_hmf = Ux_hmf,
    #                         parsem_hmf=parsem_hmf )
                        
                        err_hmf = (X_hmf !=X).astype(float).mean()
                        err_tous[numexp,id_mpm,id_s,id_snr,id_e, 0] = err_hmf
                        
                        err_otmfx = (X_tmf !=X).astype(float).mean()
                        err_tous[numexp,id_mpm,id_s,id_snr,id_e, 1] = err_otmfx
                        
                        
                        err_otmfv = (V_tmf !=V).astype(float).mean()
                        err_tous[numexp,id_mpm,id_s,id_snr,id_e, 2] = err_otmfv        
                    
#                        print "#################"
                    except:
#                        print "Erreur exp %.0f"%numexp
#                        print "#################"
                        pass
                    



#%%
err_tous_decote = np.copy(err_tous)
import scipy.stats.mstats as ms

plt.close('all')      
       
#try:

err_tous_nomask = np.minimum(err_tous_decote, 1.-err_tous_decote)  
msk = (err_tous_decote==0)

err_tous_mask = ma.masked_array(err_tous_nomask,msk==1)      
err_tous = ma.mean(err_tous_mask,axis=0)




for id_e in (0,1):
    plt.figure(figsize=(10,10))
    for id_s in (0,1):
        for id_mpm in (0,1):
            
            err_q1 = ms.mquantiles(err_tous_mask[:,id_mpm,id_s,:,id_e,0],prob=0.25, axis=0)[0,:]
            err_q2 = ms.mquantiles(err_tous_mask[:,id_mpm,id_s,:,id_e,0],prob=0.75, axis=0)[0,:]

            err_q1b = ms.mquantiles(err_tous_mask[:,id_mpm,id_s,:,id_e,1],prob=0.25, axis=0)[0,:]
            err_q2b = ms.mquantiles(err_tous_mask[:,id_mpm,id_s,:,id_e,1],prob=0.75, axis=0)[0,:]

            err_q1c = ms.mquantiles(err_tous_mask[:,id_mpm,id_s,:,id_e,2],prob=0.25, axis=0)[0,:]
            err_q2c = ms.mquantiles(err_tous_mask[:,id_mpm,id_s,:,id_e,2],prob=0.75, axis=0)[0,:]                
            
    
            if id_mpm==0 : n0 = 'MPM' 
            else: n0 = 'MAP'
            if id_e == 0 : n1 = 'B' 
            else: n1 = 'A'
            if id_s == 0 : n2 = 'supervise'
            else : n2 = 'non supervise'
            
            titre = 'Experience '+n1+' : '+n0 + ' '+n2
                  
                  #    plt.figure()
            plt.subplot(2,2,2*id_s+id_mpm+1)            
            plt.fill_between(snr_range,err_q1,err_q2, alpha=0.25,facecolor='blue')                            
            plt.plot(snr_range,err_tous[id_mpm,id_s,:,id_e,0],label='x hmf')
            
            plt.fill_between(snr_range,err_q1b,err_q2b, alpha=0.25,facecolor='orange')             
            plt.plot(snr_range,err_tous[id_mpm,id_s,:,id_e,1],label='x otmf')
            
            if id_e == 1:
                plt.fill_between(snr_range,err_q1c,err_q2c, alpha=0.25,facecolor='green')  
                plt.plot(snr_range,err_tous[id_mpm,id_s,:,id_e,2],label='v otmf')
            
            plt.xlim(snr_range.min(),snr_range.max())
            plt.ylim(0,0.5)
            plt.title(titre)
            plt.legend()
            plt.tight_layout()
        
            tikz_save('./figures/exp'+n1+'.tex', figureheight='7cm',figurewidth='10cm')
            tikz_save('/home/miv/courbot/Dropbox/figcour/otmf/exp'+n1+'.tex', figureheight='7cm',figurewidth='10cm')
#            plt.close('all') 
#except:
#    pass
 
# 
#i+=1  
#print i   
#gc.collect()
#time.sleep(300)

#%%
#
#err_tous = np.zeros(shape=(20,2,2,14,2,3))
## exp A : numero 2
## exp B : numéro 13
##for numexp in (13,):
#    #numexp=5
#
#numexp=13
#mpm = True
#superv = False
#id_snr = 7
#experiment = '3'
#
##    for mpm in (False,True):
#if mpm: n0='' ; id_mpm = 1
#else: n0='map' ; id_mpm = 0
#
#   
##        for superv in (True,False):
#if superv: n3 = 'known'; id_s = 0
#else: n3='unknown' ; id_s = 1
#
#
#
##            for sigma in (0.5,1.):
##            for id_snr in range(snr_range.size):
#snr = snr_range[id_snr]
#n2 = str(int(snr))
##                if sigma==0.5: n2 = '05'; 
##                else: n2='10'
#            
#            
#
#            
##                for experiment in ('2','3'):#('2','3'):
#if experiment=='3': n1 = 'b'; id_e = 0;
#else: n1='a' ; id_e = 1
#
#  
#nom_can = nom_fol+n0+n1+'_snr'+n2+'_'+n3
##                    print 'sauvegarde sous ' + nom_can
#            
#            
#            
#
#            
##                    print "################# Exp. %.0f"%numexp
#
##try:
#dat = np.load(nom_can+str(numexp)+'.npz')
#X,V = gen_exp(experiment,x_range,S0,S1,W=1,m=0,sig=1,rho_1=1,rho_2=1)
#
#
#Y = dat.f.Y
#X_tmf = dat.f.X_mpm_est
#V_tmf = dat.f.V_mpm_est
#Ux_map = dat.f.Ux_map
#Uv_map = dat.f.Uv_map
##                        parsem = dat.f.parsem,
#
#X_hmf = dat.f.X_mpm_hmf
##                         Ux_hmf = Ux_hmf,
##                         parsem_hmf=parsem_hmf )
#
#
#err_hmf = (X_hmf !=X).astype(float).mean()
#err_hmf = min(err_hmf,1.-err_hmf)
#
##err_tous[numexp,id_mpm,id_s,id_snr,id_e, 0] = err_hmf
#
#err_otmfx = (X_tmf !=X).astype(float).mean()
#err_otmfx = min(err_otmfx,1.-err_otmfx)
##err_tous[numexp,id_mpm,id_s,id_snr,id_e, 1] = err_otmfx
#
#
#err_otmfv = (V_tmf !=V).astype(float).mean()
##err_tous[numexp,id_mpm,id_s,id_snr,id_e, 2] = err_otmfv    
#
#print numexp,err_hmf*100, err_otmfx*100
#
#plt.figure();
#plt.subplot(121) ;         plt.imshow(Uv_map)
#plt.subplot(122) ;         plt.imshow(Ux_map)
#    except:
#        pass