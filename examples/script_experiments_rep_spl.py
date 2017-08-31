# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 09:45:48 2015

@author: courbot
"""



import numpy as np 
import sys 
import matplotlib.pyplot as plt
import time
#import scipy.stats as st
#import scipy.cluster.vq as cvq
#import multiprocessing as mp
#import image_tools as it
#from scipy.ndimage import imread
#import matplotlib.mlab as mlab
#import numpy.ma as ma
import parameters
#import image_tools as it
import gibbs_sampler as gs
#import fields_tools as ft
import seg_OTMF as sot
import SEM as sem

from scipy.ndimage import zoom
from scipy.ndimage.filters import gaussian_filter 
#from scipy.ndimage.filters import median_filter 

import gdal

import spectral.io.envi as envi

def gen_exp(experiment,x_range,S0,S1, W, m,sig,rho_1,rho_2):
    print('------- Experience '+experiment)
    

    if experiment == '1':
        ################ EXPERIMENT 1 : X,V,Y is a TMF
    
        print('------- Generate X, V')
    #    pargibbs.nb_iter = 400
    #    pargibbs.v_help==False
    #
    #    parv = gs.gen_champs_fast(pargibbs, generate_v=True, generate_x=False, use_y=False,normal=False,use_pi = True)
    #    V = parv.V_res[:,:,-1]
    #    
    #    pargibbs.nb_iter = 70
    #    pargibbs.V = V
    #    parx = gs.gen_champs_fast(pargibbs, generate_v=False, generate_x=True, use_y=False,normal=False,use_pi = True)
    ##    X = parx.X_res[:,:,-1]
    #    X = parx.X_res[:,:,-11:-1].mean(axis=2)>0.5  
        
        
        
    #    np.savez('./data/tmf_exp1_128.npz', X=X, V=V)
    #   
    #    pargibbs.v_help = False
    #    pargibbs.nb_iter = 150
    #
    #    parvx = gs.gen_champs_fast(pargibbs, generate_v=True, generate_x=True, use_y=False,normal=False,use_pi = True)
    #    V = parvx.V_res[:,:,-1]
    #    X = parvx.X_res[:,:,-1]  
    #    
    #    X_mpm_est,V_mpm_est,X_mpm,V_mpm = sot.MPM(parvx,lim=89)
    #    X = X_mpm_est
    #    V = V_mpm_est
    
        dat = np.load('./data/exp_pamiA.npz')
        X=dat['X']
        V=dat['V']
        
        Y = gen_obs(X,W, m,sig,rho_1,rho_2,corrnoise=False)
        print('------- (X,V CM)')
    elif experiment=='2':
        ################ EXPERIMENT 2 :V fixed, X, Y simulated
    #    print('------- Generate X | V')
    #    V = v_range[1] * np.ones(shape=(S0,S1))
    #    V[:S0/2,:S1/2] = v_range[0]
    #    V[S0/2:,S1/2:] = v_range[0]
    #    pi[1,:] = np.array([1e-04,   1e-04,   1e-04,1e-04,  1e-04,   1e-04, 1e-04,   1e-02,   9e-01])  
    #    pi[0,:] = np.array([1e-04,   1e-04,   1e-03, 1e-2,   5e-02,   1e-01, 3e-1,   6e-01,   8e-01])
    #    pi[1,:] /= pi[1,:].sum()
    #    pi[0,:] /= pi[0,:].sum()
    #    pargibbs.pi = pi
    #    #V[:S0/2,:] = 0
    #    
    #    
    #    pargibbs.nb_iter = 60
    #    pargibbs.V = V
    #    parx = gs.gen_champs_fast(pargibbs, generate_v=False, generate_x=True, use_y=False,normal=False,use_pi = True)
    #    X = parx.X_res[:,:,-11:-1].mean(axis=2)>0.5  
    #    X = parx.X_res[:,:,-1]
    ##    np.savez('./data/tmf_exp2_128(b).npz', X=X, V=V)
    ##    
    ##    
        print('------- (V fixe, X CM)')
        dat = np.load('./data/exp_pamiA.npz')
        X=dat['X'] > 0
#        V=dat['V']
        V = np.pi/4 * np.ones(shape=(128,128))
        V[64:,:] = 3*np.pi/4
        V[:,64:] = (V[:,64:] + np.pi/2)%np.pi
        
        Y = gen_obs(X,W,x_range, m,sig,rho_1,rho_2,corrnoise=False)
        
    elif experiment=='3':
#        dat = np.load('./results/sources/udf10/cubes/208.npz')
#        Y = dat['Y_ms']
#        S0,S1,W = Y.shape
#        X = np.zeros(shape=(S0,S1))
#        V = np.zeros_like(X)
#
#
#        Y = gen_obs(X,W, m,sig,rho_1,rho_2,corrnoise=False)
    
        dat = np.load('./data/exp_pami3v2.npz')
        X=(dat['X']>0).astype(float)
        V=dat['V']
        print('------- (X fixe, V inconnu)')
        
#        X = X[:80,:80]
#        V = V[:80,:80]
        Y = gen_obs(X,W,x_range, m,sig,rho_1,rho_2,corrnoise=False)
        
        
    elif experiment=='4':
        # cas "japan"
        X,V = gen_xv(S0,S1,v_range)
        
        Y = gen_obs(X,W, m,sig,rho_1,rho_2,corrnoise=False)
        
    elif experiment=='5':
        dat = np.load('./data/synth1b.npz') #"jap"
        dat = np.load('./data/jap.npz') #"jap"
        
        ratio = np.float(S0)/dat['X'].shape[0]
        X =  zoom(gaussian_filter(dat['X'], sigma=(ratio,ratio)), (ratio,ratio),order=0) 
        X = X.astype(float)
        V =  gs.cast_angles(zoom(dat['V'], (ratio,ratio),order=0),v_range)
#        X[:, S1/2:]*=0.5
        
        Y = gen_obs(X,W, m,sig,rho_1,rho_2,corrnoise=False)
        
    elif experiment=='6':
        dat = np.load('./data/atten1.npz')
        ratio = np.float(S0)/dat['X'].shape[0]
        
        X =  5.*zoom(gaussian_filter(dat['X'], sigma=(ratio,ratio)), (ratio,ratio),order=0) 
        
        V =  gs.cast_angles(zoom(dat['V'], (ratio,ratio),order=0),v_range)
        
        Y = gen_obs(X,W, m,sig,rho_1,rho_2,corrnoise=False)
#        X_diff = np.abs(X[:,:,np.newaxis] - 5*x_range[np.newaxis,np.newaxis,:])
#        
#        ind_x = np.argmin(X_diff,axis=2)
#        X_cast = x_range[ind_x]*5
##        for x_inst in x_range:
#        X = np.copy(X_cast)    
        
        
#        X[:, S1/2:]*=0.5
    #    X = X/10.
        
    elif experiment=='7':
        img2 = envi.open('./data/donnees_ipag/FRT39DF/psp1981_Red_2p5m_proj_warp_p2_subset.hdr','./data/donnees_ipag/FRT39DF/psp1981_Red_2p5m_proj_warp_p2_subset')
        data2 = np.copy(img2.asarray()).astype(float)
        S0 = 128
        S1 = 128
        Y = np.zeros(shape=(S0,S1,1))
        
#        Y[:,:,0] = data2[1150:1230,1050:1130,0]
#        Y[:,:,0] = data2[1100:1356,820:1076,0]
        
#        Y[:,:,0] = data2[1150:1278,750:878,0]
        Y[:,:,0] = data2[1064:1064+S0,2130:2130+S1,0]
        
#        [1064:,2130:,0]
        Y-=Y.min()
        Y /= Y.max()
        S0,S1,W = Y.shape
        X = np.zeros(shape=(S0,S1))
        V = np.zeros_like(X)
        print('------- (MRO)')
        
    elif experiment=='8':
        cube = gdal.Open('./data/donnees_sertit/extrait_vignes_pleiades_pan_20120909.tif')
        
#        img2 = envi.open('./data/donnees_ipag/FRT39DF/psp1981_Red_2p5m_proj_warp_p2_subset.hdr','./data/donnees_ipag/FRT39DF/psp1981_Red_2p5m_proj_warp_p2_subset')
        data = cube.GetRasterBand(1).ReadAsArray()
        S0 = 128
        S1 = 128
        Y = np.zeros(shape=(S0,S1,1))
        
#        Y[:,:,0] = data2[1150:1230,1050:1130,0]
#        Y[:,:,0] = data[550:630,820:900].astype(float)
#        Y[:,:,0] = data[840:920,1080:1160].astype(float)
#        Y[:,:,0] = data[360:440,240:320].astype(float)
        Y[:,:,0] = data[800:800+S0,1020:1020+S1].astype(float)
        
        
#        Y[:,:,0] = data[820:820+80,1040:1040+80].astype(float)
#        860:860+128,1110:1110+128
        Y-=Y.min()
        Y /= Y.max()
#        S0,S1,W = Y.shape
        X = np.zeros(shape=(S0,S1))
        V = np.zeros_like(X)        
        print('------- (Sentinel)')

    return X,V,Y



#%%
def plot_taux(im,ref,title):
    
    taux = (im!=ref).mean() * 100
    plt.imshow(im.T,  interpolation='nearest', origin='lower', cmap=plt.cm.bone,); 
    plt.title(title + ' - %.2f'%taux)
    plt.axis('off')
    
def erreur(A,B):
    return (A[~np.isnan(B)] != B[~np.isnan(B)] ).mean()
#%%
def plot_directions(angle, intensite,pas,taille=1):
    
    S0 = angle.shape[0]
    S1 = angle.shape[1]
    
    y,x = np.ogrid[0:S0,0:S1]
 
    angle2 = angle

    deb_x = np.tile(x,(S0,1)) - taille*np.sin(angle2) * intensite
    deb_y = np.tile(y,(1,S1)) - taille*np.cos(angle2) * intensite
    
    fin_x = np.tile(x,(S0,1)) + taille*np.sin(angle2) * intensite
    fin_y = np.tile(y,(1,S1)) + taille*np.cos(angle2) * intensite
    
    
    for i in range(int(pas/2.),S0,pas):
        for j in range(int(pas/2.),S1,pas):
#            if angle[i,j] != 0:
                plt.plot((deb_x[i,j],fin_x[i,j]), (deb_y[i,j],fin_y[i,j]) ,'k')
    plt.xlim((-0.5,S1-0.5))
    plt.ylim((-0.5,S0-0.5))     
#%% 
def gen_mu(W):
    
    if W==1:
        mu = 1.
    else:
        mu = np.zeros(shape=W)
        
        mu[int(W/2.)-1] = 0.3 
        mu[int(W/2.)] = 1 
        mu[int(W/2.)+1] = 0.5     
    
    return mu

def gen_obs(X,W,x_range, m,sig,rho_1,rho_2,corrnoise=False):
    
    S0,S1 = X.shape
     
    ###%% Creation de l'observation, hyperspectrale
    mu = gen_mu(W)
    ##
    ###%% Generation observation y
    ### Valable dans le cas ou le bruit est independant !
    # => faire des options pour bruit corrélé ?
    
    if corrnoise==True:
        Sigma = np.eye(W) * sig**2 + (np.eye(W,k=1) +  np.eye(W,k=-1)) * rho_1 + (np.eye(W,k=2) +  np.eye(W,k=-2)) * rho_2    
        
    else:
        Sigma = np.eye(W) * sig**2
    
    if W ==1 :
        Y = np.zeros(shape=(S0,S1,1))
        Y_tmp = np.zeros(shape=(S0,S1))
        for id_x in range(x_range.size):
            bruit_tout = np.random.normal(loc=0.,scale=sig[id_x],size=(S0,S1))
            Y_tout = X*mu + bruit_tout
            Y_tmp[X==x_range[id_x]] = Y_tout[X==x_range[id_x]]
        Y[:,:,0] = Y_tmp
#            Y[:,:,0] = X*mu + np.random.normal(loc=0.,scale=sig,size=(S0,S1))
    else:
        Y = X[:,:,np.newaxis]*mu[np.newaxis,np.newaxis,:] +np.random.multivariate_normal(mean=np.zeros_like(mu),cov=Sigma,size=(S0,S1))
    
    #Y = X[:,:,np.newaxis]*mu[np.newaxis,np.newaxis,:] + st.norm.rvs(loc=m,scale=sig,size=(S0,S1,W))
    
#    pargibbs.Y = Y
    
    return Y
    
def gen_xv(S0,S1,v_range,):
    
    y,x = np.ogrid[0:S0,0:S1]
    
    X = np.zeros(shape=(S0,S1))
    
    reg_ne = (x >= S0/2) * (y>= S1/2) 
    reg_nw = (x < S0/2) * (y>= S1/2) 
    reg_se = (x >= S0/2) * (y< S1/2) 
    reg_sw = (x < S0/2) * (y< S1/2) 
    
    
#    freq0 = 1/30. #frequence spatiale
    freq1 = 1/30.
    #X = (np.cos(2*np.pi*freq*(y - x*np.cos(v_range[0])))>0)*reg_sw 
    X +=  (np.cos(2*np.pi*freq1*(y - x*np.cos(v_range[1])))>0)*(reg_se)
    X +=  (np.cos(2*np.pi*freq1*(y - x*np.cos(v_range[1])))>0)*(reg_nw)
    
    X +=  (np.cos(2*np.pi*freq1*(y - x*np.cos(v_range[0])))>0)*(reg_ne)
    X +=  (np.cos(2*np.pi*freq1*(y - x*np.cos(v_range[0])))>0)*(reg_sw)
    X_n = X[:S0/2,:]

    X[S0/2:,:] = np.flipud(X_n)
#    X = np.copy(X_s)
#    X +=  (np.cos(2*np.pi*freq1*(y - x*np.cos(v_range[0])))>0)*(reg_sw)
#    
#    X = X>0
    
    V = np.zeros(shape=(S0,S1))
    
    V[reg_sw+reg_ne] = v_range[0]
    V[reg_nw+reg_se] = v_range[1]
    
    return X,V
    
#%%  


def run_exp(experiment, sigma,superv=False,mpm=True):


#experiment = '3'


    
    S0 = 128
    S1 = 128   

    ###################
#    if experiment=='2':
    pas = np.pi/2
    v_range = np.arange(pas/2., np.pi, pas)
#    else:
#        pas = np.pi/6
#        v_range = np.arange(pas/2., np.pi, pas)


    ###################
    nb_level_x =1
    x_range = np.arange(0, 1+1./nb_level_x, 1./nb_level_x)



    print('---------------------------------------')
    pargibbs = parameters.ParamsGibbs(S0 = S0,
                                 S1 = S1,
                                 type_pot = 'potts',
                                 phi_uni = 0.,
                                 thr_conv=0.001,
                                 nb_iter=100,
                                 fuzzy=False,
                                 anisotropic=True,   
                                 angle=np.zeros(shape=(S0,S1)),
                                 beta = 1.,
                                 phi_theta_0 = 0.,
                                 alpha =5,
                                 alpha_v = 10,
                                 delta = 0.,
                                 init_method = 'std',
                                 nb_fuzzy = 256. ,
                                 v_range = v_range,
                                 x_range = x_range
                                 )# beta=1.25,

    
    W=1
    pargibbs.W = W
    #pargibbs.autoconv=False
    
#    mu = gen_mu(pargibbs.W)
#sig = np.linalg.norm(mu)/np.sqrt(pargibbs.W)* 10**(-SNR/20.)

    sig = np.array([sigma,sigma])#,0.2])

    m = 0

    rho_1 = 0.5 * sig**2
    rho_2 = 0.25 * sig**2


    X,V,Y = gen_exp(experiment,x_range,S0,S1,W,m,sig,rho_1,rho_2)

    pargibbs.Y = Y
    pargibbs.S0 = S0
    pargibbs.S1 = S1

    #==============================================================================
    # Paramètres à fixer
    #==============================================================================
    # Les valeurs par défaut sont celles-ci:
    
    #nb_iter_sem=40 # nb d'iter maximum pour SEM
    #nb_rea = 100 # nombre de realisations de Gibbs differentes pour le MPM
    #taille_fen = 5 # fenetre pour la convergence de SEM
    #seuil_conv = 0.05 # convergence de SEM
    #
    #nb_iter_mpm = 100 # longueur max. pour les Gibbs dans le MPM
    #pargibbs.nb_iter = 100 
    #pargibbs.autoconv=True # convergence automatique des estimateurs de Gibbs
    #pargibbs.thr_conv = 5*1./(S0*S1) # seuil pour cette convergence, en relatif
    incert = True # Utilisation ou non de segmentation avec incertitude
    #pargibbs.Xi = 0.  # valeur de l'"incertitude" adoptee
    #tmf = False

    parseg = parameters.ParamsSeg(nb_iter_sem=40,
                                  seuil_conv = 1*(1./S0*S1),
                                  incert = incert
                                    )
    parseg.spec_snr=False #plus tard !
    parseg.multi = True # le multiclasse discret
    parseg.seuil_conv = 0.05
#nb_classe = 1
    parseg.weights=np.ones(shape=(S0,S1))
    
    
    parseg.mpm = mpm    
    
    #
    ##==============================================================================
    ## Segmentation HMF
    ##==============================================================================
    parseg.tmf = False
    #nb_classe = 1
#    parseg.seuil_conv = 0.05
    pargibbs = parameters.apply_parseg_pargibbs(parseg,pargibbs) # transfetrt a l'autre jeu de parametre
    print '---------------HMF---------------------'
    start = time.time()
    
    Y_courant = np.copy(Y)
    
    pargibbs.Y = Y_courant
    
    if superv:
        pi = sem.est_pi(X,V,pargibbs)
        real_par = parameters.ParamsChamps(mu=x_range,sig=sig)
        real_par.pi = pi
        if experiment=='3':
            real_par.pi[1,:] = real_par.pi[0,:]
        parseg.real_par = real_par
    X_mpm_hmf,dumb0,Ux_hmf,dumb1, parsem_hmf = sot.seg_otmf(parseg,pargibbs,superv)
    
    ## 
    end = time.time() - start
    print 'Temps total : %.2f s'%end  
    print '------------------------------------'
#    plt.imshow(X_mpm_hmf)
#
#        Ex_tmf = (X_mpm_est != X).mean()
    Ex_hmf = (X_mpm_hmf != X).mean()
    
    print 'Taux erreur HMF : %.7f'%(Ex_hmf*100)#,Ex_tmf*100)    
    ###
    ###==============================================================================
    ### Segmentation OTMF
    ###==============================================================================
    parseg.tmf = True
#    parseg.seuil_conv = 0.05
    pargibbs = parameters.apply_parseg_pargibbs(parseg,pargibbs) # transfetrt a l'autre jeu de parametre
    print '---------------OTMF--------------------'
    start = time.time()
    
    Y_courant = np.copy(Y)
    
    pargibbs.Y = Y_courant
    
    if superv:# useless?
        pi = sem.est_pi(X,V,pargibbs)
        real_par = parameters.ParamsChamps(mu=x_range,sig=sig)
        real_par.pi = pi
        parseg.real_par = real_par
        
    pargibbs.X_init = X
    X_mpm_est,V_mpm_est,Ux_map,Uv_map, parsem = sot.seg_otmf(parseg,pargibbs,superv)
    
    ## 
    end = time.time() - start
    X = X>0
    print 'Temps total : %.2f s'%end  
    print '------------------------------------'
    
    Ex_tmf = (X_mpm_est != X).mean()
    Ex_hmf = (X_mpm_hmf != X).mean()
    
    print 'Taux erreur HMF : %.7f, taux erreur TMF : %.7f'%(Ex_hmf*100,Ex_tmf*100)   
    if experiment=='2':
        Ev_tmf = (V_mpm_est != V).mean()
        print 'Taux erreur TMF V : %.2f'%(Ev_tmf*100)
    
    
    
    return Y, X_mpm_est,V_mpm_est,Ux_map,Uv_map, parsem, X_mpm_hmf,Ux_hmf, parsem_hmf




#%%
nom_fol = './results/pami/exp'


#experiment='3' # B dans le papier
#sigma=0.5
#superv = True#False


S0 = 128
S1 = 128   

###################
#    if experiment=='2':
pas = np.pi/2
v_range = np.arange(pas/2., np.pi, pas)
#    else:
#        pas = np.pi/6
#        v_range = np.arange(pas/2., np.pi, pas)


###################
nb_level_x =1
x_range = np.arange(0, 1+1./nb_level_x, 1./nb_level_x)



print('---------------------------------------')
pargibbs = parameters.ParamsGibbs(S0 = S0,
                             S1 = S1,
                             type_pot = 'potts',
                             phi_uni = 0.,
                             thr_conv=0.001,
                             nb_iter=100,
                             fuzzy=False,
                             anisotropic=True,   
                             angle=np.zeros(shape=(S0,S1)),
                             beta = 1.,
                             phi_theta_0 = 0.,
                             alpha =5,
                             alpha_v = 10,
                             delta = 0.,
                             init_method = 'std',
                             nb_fuzzy = 256. ,
                             v_range = v_range,
                             x_range = x_range
                             )# beta=1.25,


W=1
pargibbs.W = W
#pargibbs.autoconv=False

experiment = '3'

#    mu = gen_mu(pargibbs.W)
#sig = np.linalg.norm(mu)/np.sqrt(pargibbs.W)* 10**(-SNR/20.)
sigma = 0.5
sig = np.array([sigma,sigma])#,0.2])

m = 0

rho_1 = 0.5 * sig**2
rho_2 = 0.25 * sig**2


X,V,Y = gen_exp(experiment,x_range,S0,S1,W,m,sig,rho_1,rho_2)

pargibbs.Y = Y
pargibbs.S0 = S0
pargibbs.S1 = S1

#==============================================================================
# Paramètres à fixer
#==============================================================================
# Les valeurs par défaut sont celles-ci:

#nb_iter_sem=40 # nb d'iter maximum pour SEM
#nb_rea = 100 # nombre de realisations de Gibbs differentes pour le MPM
#taille_fen = 5 # fenetre pour la convergence de SEM
#seuil_conv = 0.05 # convergence de SEM
#
#nb_iter_mpm = 100 # longueur max. pour les Gibbs dans le MPM
#pargibbs.nb_iter = 100 
#pargibbs.autoconv=True # convergence automatique des estimateurs de Gibbs
#pargibbs.thr_conv = 5*1./(S0*S1) # seuil pour cette convergence, en relatif
incert = True # Utilisation ou non de segmentation avec incertitude
#pargibbs.Xi = 0.  # valeur de l'"incertitude" adoptee
#tmf = False

parseg = parameters.ParamsSeg(nb_iter_sem=40,
                              seuil_conv = 1*(1./S0*S1),
                              incert = incert
                                )
parseg.spec_snr=False #plus tard !
parseg.multi = True # le multiclasse discret
parseg.seuil_conv = 0.05
#nb_classe = 1
parseg.weights=np.ones(shape=(S0,S1))


parseg.mpm = True
superv = False
#
##==============================================================================
## Segmentation HMF
##==============================================================================
parseg.tmf = False
#nb_classe = 1
#    parseg.seuil_conv = 0.05
pargibbs = parameters.apply_parseg_pargibbs(parseg,pargibbs) # transfetrt a l'autre jeu de parametre
print '---------------HMF---------------------'
start = time.time()

Y_courant = np.copy(Y)

pargibbs.Y = Y_courant

if superv:
    pi = sem.est_pi(X,V,pargibbs)
    real_par = parameters.ParamsChamps(mu=x_range,sig=sig)
    real_par.pi = pi
    if experiment=='3':
        real_par.pi[1,:] = real_par.pi[0,:]
    parseg.real_par = real_par
X_mpm_hmf,dumb0,Ux_hmf,dumb1, parsem_hmf = sot.seg_otmf(parseg,pargibbs,superv)

## 
end = time.time() - start
print 'Temps total : %.2f s'%end  
print '------------------------------------'
#    plt.imshow(X_mpm_hmf)
#
#        Ex_tmf = (X_mpm_est != X).mean()
Ex_hmf = (X_mpm_hmf != X).mean()

print 'Taux erreur HMF : %.7f'%(Ex_hmf*100)#,Ex_tmf*100)    
#print 'sauvegarde sous ' + nom_can
#%%
#for numexp in range(20):
#    
#    for mpm in (False,):
#        if mpm: n0=''
#        else: n0='map'
#    
#   
#        for superv in (True,False):
#            if superv: n3 = 'known'; 
#            else: n3='unknown'
#        
#            
#            
#            for sigma in (0.5,1.):
#                if sigma==0.5: n2 = '05'; 
#                else: n2='10'
#                    
#                
#    
#                
#                for experiment in ('2','3'):#('2','3'):
#                    if experiment=='3': n1 = 'b'; 
#                    else: n1='a'
#    
#              
#                    nom_can = nom_fol+n0+n1+'_sig'+n2+'_'+n3
#                    print 'sauvegarde sous ' + nom_can
#                
#                
#                
#    
#                
#                    print "################# Exp. %.0f"%numexp
#                    
#                    try:
#                        Y, X_mpm_est,V_mpm_est,Ux_map,Uv_map, parsem, X_mpm_hmf,Ux_hmf, parsem_hmf = run_exp(experiment,sigma,superv,mpm)
#                    
#                    
#                        np.savez(nom_can+str(numexp),
#                                                                 Y = Y,
#                                                                 X_mpm_est = X_mpm_est,
#                                                                 V_mpm_est = V_mpm_est,
#                                                                 Ux_map = Ux_map, 
#                                                                 Uv_map = Uv_map,
#                                                                 parsem = parsem,
#                                                                 X_mpm_hmf = X_mpm_hmf,
#                                                                 Ux_hmf = Ux_hmf,
#                                                                 parsem_hmf=parsem_hmf )
#        
#                    
#                        print "#################"
#                    except:
#                        print "Erreur exp %.0f"%numexp
#                        print "#################"
#                        pass
