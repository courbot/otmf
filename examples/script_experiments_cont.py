# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 09:45:48 2015

@author: courbot
"""



import numpy as np 
import sys 
import matplotlib.pyplot as plt
import matplotlib.image as im
import time
from PIL import Image
#import scipy.stats as st
#import scipy.cluster.vq as cvq
#import multiprocessing as mp
#import image_tools as it
#from scipy.ndimage import imread
#import matplotlib.mlab as mlab
#import numpy.ma as ma
from otmf import parameters
#import image_tools as it
from otmf import gibbs_sampler as gs
from otmf import fields_tools as ft
from otmf import seg_OTMF as sot
from otmf import SEM as sem

from scipy.ndimage import zoom
from scipy.ndimage.filters import gaussian_filter 
from scipy.ndimage.filters import median_filter 

#import gdal

#import spectral.io.envi as envi




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
                plt.plot((deb_x[i,j],fin_x[i,j]), (deb_y[i,j],fin_y[i,j]) ,'k',linewidth=2)
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
    gen_obs(X,W, m,sig,rho_1,rho_2,corrnoise=False)

     
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
        X = X[:80,:80]
        V = V[:80,:80]        
        
#        Y = gen_obs(X,W, m,sig,rho_1,rho_2,corrnoise=False)
        Y = np.zeros(shape=(S0,S1,1))
        Y[:,:,0] = X + np.random.normal(loc=0.,scale=sig[0],size=(S0,S1))
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
#        dat = np.load('./data/exp_pamiB.npz')
#        X=dat['X']
#        V=dat['V']../../../Donnees/Otmf
        dat = np.load('../../../Donnees/Otmf/exp_pamiA.npz')
        X=dat['X'] > 0
#        V=dat['V']
        V = np.pi/4 * np.ones(shape=(128,128))
        V[64:,:] = 3*np.pi/4
        V[:,64:] = (V[:,64:] + np.pi/2)%np.pi
        
        X = X[:80,:80]
        V = V[:80,:80]        
        
        Y = np.zeros(shape=(S0,S1,1))
        Y[:,:,0] = X + np.random.normal(loc=0.,scale=sig[0],size=(S0,S1))
        #Y = gen_obs(X,W, m,sig,rho_1,rho_2,corrnoise=False)
        
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
        
        
        X = X[:80,:80]
        V = V[:80,:80]       
        print('------- (X fixe, V inconnu)')
        
#        X = X[:80,:80]
#        V = V[:80,:80]
        Y = np.zeros(shape=(S0,S1,1))
        Y[:,:,0] = X + np.random.normal(loc=0.,scale=sig[0],size=(S0,S1))
        
        
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
        Y[:,:,0] = data2[1024:1024+S0,2000:2000+S1,0] # dans le papier v1..3
        
#        im = data2[1024:1024+384,2000:2000+384,0]
#        Y[:,:,0] = data2[550:550+128,1100:1100+128,0]#zoom(gaussian_filter(im,sigma=1.0),0.333)
        #Y[:,:,0] = data2[940:940+S0,2110:2110+S1,0] # ok pour 80
        #Y[:,:,0] = data2[910:910+S0,2070:2070+S1,0] # dans la v7+
#        Y[:,:,0] = data2[920:920+S0,2080:2080+S1,0]
        #Y[:,:,0] = data2[1630:1630+S0,2840:2840+S1,0]
#        [1064:,2130:,0]
        Y-=Y.min()
#        Y /= Y.std()
        Y /= Y.max()
        S0,S1,W = Y.shape
        X = np.zeros(shape=(S0,S1))
        V = np.zeros_like(X)
        
#        X = X[:80,:80]
#        V = V[:80,:80]        
#        Y = Y[:80,:80,:]
#        
        
        print('------- (MRO)')
        
    elif experiment=='8':
        cube = gdal.Open('./data/donnees_sertit/extrait_vignes_pleiades_pan_20120909.tif')
        
        #        img2 = envi.open('./data/donnees_ipag/FRT39DF/psp1981_Red_2p5m_proj_warp_p2_subset.hdr','./data/donnees_ipag/FRT39DF/psp1981_Red_2p5m_proj_warp_p2_subset')
        data = cube.GetRasterBand(1).ReadAsArray()
        S0 = 256
        S1 = 256
        Y = np.zeros(shape=(S0,S1,1))
        
        #        Y[:,:,0] = data2[1150:1230,1050:1130,0]
        #        Y[:,:,0] = data[550:630,820:900].astype(float)
        #        Y[:,:,0] = data[840:920,1080:1160].astype(float)
        #        Y[:,:,0] = data[360:440,240:320].astype(float)
#        Y[:,:,0] = data[800:800+S0,1020:1020+S1].astype(float)
        
        
        
#        Y[:,:,0] = data[736:736+S0,956:956+S1].astype(float) # pas mal #1
        
#        Y[:,:,0] = data[800:800+S0,1020:1020+S1].astype(float)
#        
#        
#        Y[:,:,0] = data[736:736+S0,1056:1056+S1].astype(float) # pas mal #2
        
        Y[:,:,0] = data[736:736+S0,900:900+S1].astype(float) # pas mal #2
                
        
        Y[Y>800] = 800.
#        Y[:,:,0] = data[820:820+80,1040:1040+80].astype(float)
#        860:860+128,1110:1110+128
        Y-=Y.min()
        Y /= Y.max()
#        S0,S1,W = Y.shape
        X = np.zeros(shape=(S0,S1))
        V = np.zeros_like(X)        
        print('------- (Sentinel)')
        
    elif experiment=='9':
        
        dat=plt.imread('./data/empdi_2_256.png')
    
        S0 = 256
        S1 = 256
        Y = np.zeros(shape=(S0,S1,1))

        Y[:,:,0] = dat[:,:,0]
        
        
#        Y[:,:,0] = data[820:820+80,1040:1040+80].astype(float)
#        860:860+128,1110:1110+128
        Y-=Y.min()
        Y /= Y.max()
#        S0,S1,W = Y.shape
        X = np.zeros(shape=(S0,S1))
        V = np.zeros_like(X)        
        print('------- (Empreinte digitale)')
        
        
    elif experiment=='10':
#        cube = gdal.Open('./data/donnees_sertit/extrait_peupliers_Spot6_pan_20140504.tif')
        cube = gdal.Open('./data/donnees_sertit/extrait_camps_pleiades_mspan_20120719.tif')
        #gdal.Open('donnees_sertit/sertit/extrait_camps_pleiades_mspan_20120719.tif')


        slice0=cube.GetRasterBand(1).ReadAsArray()
        
        S0 = 128
        S1 = 128
        Y = np.zeros(shape=(S0,S1,1))
        
        #        deb_x = 160
        #        deb_y = 240
        deb_x = 1000
        deb_y = 780
        Y[:,:,0] = slice0[deb_x:deb_x+128,deb_y:deb_y+128].astype(float)       

        #        Y[:,:,0] = slice0[450:450+128,350:350+128].astype(float)

        Y-=Y.min()
        Y /= Y.max()
#        S0,S1,W = Y.shape
        X = np.zeros(shape=(S0,S1))
        V = np.zeros_like(X)        
        print('------- (Sentinel, peuplier)')

    
    elif experiment=='11':
#        im = Image.open('./data/brodatz_norm/D11.gif')
#        im = Image.open('./data/brodatz_norm/D11.tif')
    # vermicelle
#        im = Image.open('./data/brodatz_norm/D109.tif')
#        pi = np.array(im.getdata()).reshape(im.size[0], im.size[1])
#        data = pi[328:328+128,268:268+128]/256.#97
#        S0 = S1 = 128
#        print('------- (Brodatz 109)')


        im = Image.open('./data/brodatz_norm/D109.tif')
        pi = np.array(im.getdata()).reshape(im.size[0], im.size[1])
        data = pi[328:328+256,268:268+256]/256.#97
        S0 = S1 = 256
        print('------- (Brodatz 109)')    
        # cyclone
#        im = Image.open('./data/brodatz_norm/D113.tif')
#        pi = np.array(im.getdata()).reshape(im.size[0], im.size[1])
#        data = pi[:256,:256]
#        S0 = S1 = 256
#        print('------- (cyclone)')

#        data = pi[100:228,220:348]/256.
#    
#        S0 = 128
#        S1 = 128
#        data = pi[100:356,220:476]/256.
#    
#        S0 = 256
#        S1 = 256

#        data = pi[50:306,120:376]/256.
#        data = pi[:256,120:376]/256. # 111
#        data = pi[:128,120:248]/256. # 111
#    
#        data = pi[328:328+128,268:268+128]/256.#97
##        data = pi[:256,:256]
##        S0 = S1 = 256
#        S0 = S1 = 128
#        S1 = 256

        
        Y = np.zeros(shape=(S0,S1,1))
        
        Y[:,:,0] = data #+ np.random.normal(loc=0.,scale=0.2,size=(S0,S1))#[1150:1230,1050:1130,0]


        X = np.zeros(shape=(S0,S1))
        V = np.zeros_like(X)        
#        print('------- (Brodatz 109)')
        
    elif experiment=='12':
                # cyclone
        im = Image.open('./data/brodatz_norm/D113.tif')
        pi = np.array(im.getdata()).reshape(im.size[0], im.size[1])
        data = pi[:256,:256]
        S0 = S1 = 256
        print('------- (cyclone)')
        
        
        Y = np.zeros(shape=(S0,S1,1))
        
        Y[:,:,0] = data #+ np.random.normal(loc=0.,scale=0.2,size=(S0,S1))#[1150:1230,1050:1130,0]


        X = np.zeros(shape=(S0,S1))
        V = np.zeros_like(X)   

    return X,V,Y


    
#%%  

experiment = '2'

# 7 mars
# 8 vignes
# 2 semi markov
# 3 pas markov (rayon)
# 9 empreinte
#  10 peupliers

# parametres utiles:
#if experiment=='3':
#    S0 = 90
#    S1 = 52
##elif experiment=='7':
##    S0 = 256
##    S1 = 256   
###    
#elif experiment=='8' or experiment=='7' or experiment=='1':
#S0 = 256
#S1 = 256   
#else:
#    S0 = 80
#    S1=  80
    
#S0 = S1 = 128
#S1 =    
#
S0,S1 = 80,80
    ###################
#%%
#v_range = np.array([np.pi/4.,3*np.pi/4.])#-np.pi/6

# definition de la subdivision : finir a pi, ne pas mettre 0 d'abord !
#v_range = np.array([np.pi/3,2*np.pi/3, np.pi])

pas = np.pi/2
v_range = np.arange(pas/2., np.pi, pas)

#v_range = np.array([np.pi/2,np.pi])
#v_range = np.array([np.pi/4,3*np.pi/4])
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

SNR = 0
mu = gen_mu(pargibbs.W)
sig = np.array([1.0,1.0])*np.linalg.norm(mu)/np.sqrt(pargibbs.W)* 10**(-SNR/20.)

#sig = 1.0*np.array([1.0,1.0])#,0.2])

m = 0

rho_1 = 0.5 * sig**2
rho_2 = 0.25 * sig**2


X,V,Y = gen_exp(experiment,x_range,S0,S1,W,m,sig,rho_1,rho_2)






if experiment=='3':

    pargibbs.S0 = S0
    pargibbs.S1 = S1




pargibbs.Y = Y



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
                              seuil_conv = 0.05,
                              incert = incert
                                )
parseg.spec_snr=False #plus tard !
parseg.multi = True # le multiclasse discret
#parseg.seuil_conv = 0.025
#nb_classe = 1
parseg.weights=np.ones(shape=(S0,S1))

parseg.mpm = True

parseg.nb_iter_serie_sem = 13
parseg.use_pi = True
parseg.use_alpha = True

#
#==============================================================================
# Segmentation HMF
#==============================================================================
parseg.tmf = False
#nb_classe = 1
#parseg.seuil_conv = 0.01
pargibbs = parameters.apply_parseg_pargibbs(parseg,pargibbs) # transfetrt a l'autre jeu de parametre
#pargibbs.thr_conv = 10./(128*128)
print '---------------HMF---------------------'
start = time.time()

Y_courant = np.copy(Y)

pargibbs.Y = Y_courant
#pargibbs.X_init = None

X_mpm_hmf,V_mpm_hmf,Ux_hmf,Uv_hmf, parsem_hmf = sot.seg_otmf(parseg,pargibbs)

## 
end = time.time() - start
print 'Temps total : %.2f s'%end  
print '------------------------------------'
#
###
###
###==============================================================================
### Segmentation OTMF
#####==============================================================================
parseg.tmf = True
#parseg.seuil_conv = 0.05
pargibbs = parameters.apply_parseg_pargibbs(parseg,pargibbs) # transfetrt a l'autre jeu de parametre
print '---------------OTMF--------------------'
start = time.time()

Y_courant = np.copy(Y)

#pargibbs.X_init = X_mpm_hmf
pargibbs.Y = Y_courant


X_mpm_est,V_mpm_est,Ux_map,Uv_map, parsem = sot.seg_otmf(parseg,pargibbs)

## 
end = time.time() - start
print 'Temps total : %.2f s'%end  
print '------------------------------------'



#np.savez('/home/miv/courbot/Dropbox/res_sp_cyclone_8cl_mpm',X_mpm_est = X_mpm_est, X_mpm_hmf = X_mpm_hmf, V_mpm_est = V_mpm_est, Y = Y,Ux_map=Ux_map, Ux_hmf=Ux_hmf, Uv_map=Uv_map, parsem = parsem, parsem_hmf = parsem_hmf)


###
#==============================================================================
# Un peu d'adaptation a posteriori si il y a plus que 2 classes
#==============================================================================
##%%
#if x_range.size>2:
#    snr_tous = np.zeros_like(x_range) 
#    # snr estime de la region a 1
#    snr_tous = 10*np.log10((np.linalg.norm(x_range[:,np.newaxis]*parsem.mu[:,0],axis=1)**2) /(W*parsem.sig) )
#
#    X_snr_true = 10*np.log10((np.linalg.norm(X[:,:,np.newaxis]*mu,axis=2)**2) /(W*sig) )
#    
#    # calcul de la RMSE
#    facteur_multiplicatif = np.median(X[X_mpm_est==1]).mean()
#    ecart_moyen = np.abs( (X/5. - X_mpm_est)).mean()
##    rmse = np.sqrt(mse)
#    
##%%
#    
## ce qu'on mesure sur le résultat
#print "Erreur obtenue avec l'estimation"
#facteur = np.mean(X[X_mpm_est==1])
#    
#X_scale = X/facteur
#mse = np.mean((X_scale-X_mpm_est)**2)
#rmse = np.sqrt(mse)
#
#print 'mse: ' + str(mse)
#print 'rmse: ' + str(rmse)
#
## ce qu'on mesure sur une discrétisation de l'image
#print "Erreur obtenue avec la discretisation"
#X_diff = np.abs(X[:,:,np.newaxis] - facteur*x_range[np.newaxis,np.newaxis,:])
##        
#ind_x = np.argmin(X_diff,axis=2)
#X_cast = x_range[ind_x]
#
#mse = np.mean((X_cast-X_mpm_est)**2)
#rmse = np.sqrt(mse)
#
#print 'mse: ' + str(mse)
#print 'rmse: ' + str(rmse)

#%%
#==============================================================================
#   Affichage verite + segmentations
#==============================================================================
import matplotlib

#import cmocean
nb_li = 2
nb_col =4


X,V,dumb = gen_exp(experiment,x_range,S0,S1,W,m,sig,rho_1,rho_2)


cm_gris = matplotlib.cm.gray
cm_angl = matplotlib.cm.Spectral

cm_gris.set_bad('r',1.)
#X_mpm_est[X_mpm_est==0]+=np.nan
cm_angl.set_bad('r',1.)


plt.figure(figsize=(4.5*nb_col,4*nb_li))


plt.subplot(nb_li,nb_col,1)
plt.imshow(Y.mean(axis=2), interpolation='nearest', origin='lower', cmap=cm_gris)#,vmin = -1,vmax=1); 
#plt.axis('off')
plt.title('$\mathbf{y}$, moyenne spectrale')

plt.subplot(nb_li,nb_col,2)
# on reordonne par intensite
ind_sort = np.argsort(parsem_hmf.mu[:,0])
xr_sort = x_range[::-1]
X_hmf = np.copy(X_mpm_hmf)

#
#im = Y.mean(axis=2)
#for x_inst in x_range:    
#    msk_bon = X_mpm_hmf==x_inst
#    moy_inst = im[msk_bon].mean()
#    X_hmf[msk_bon] = moy_inst#x_range[ind[0]]
#    
#xrn = np.unique(X_hmf)
#msk0 = X_hmf == xrn.min() ;msk1 = X_hmf == xrn.max()
#msk05 = (msk0==0)*(msk1==0)
#
#X_hmf[msk0] = 0. ; X_hmf[msk05] = 0.5 ; X_hmf[msk1] = 1. ;
Xh2 = np.copy(X_hmf)
#Xh2[X_hmf==1]=0.5 
#Xh2[X_hmf==0]=0
#Xh2[X_hmf==0.5]=1
                                                
plt.imshow(1-Xh2, interpolation='nearest', origin='lower', cmap=cm_gris)#,vmin=0,vmax=1); 
plt.title('$\\hat{\mathbf{x}}_{\mathrm{MPM}}$ (HMF) ')

    
plt.subplot(nb_li,nb_col,3)
#X_tmf = np.copy(X_mpm_est)
ind_sort = np.argsort(parsem_hmf.mu[:,0])
xr_sort = x_range[::-1]
X_tmf = np.copy(X_mpm_est)

#
#im = Y.mean(axis=2)
#for x_inst in x_range:    
#    msk_bon = X_mpm_est==x_inst
#    moy_inst = im[msk_bon].mean()
#    ind = np.argsort(np.abs(x_range-moy_inst))
#    msk_est = X_mpm_est==x_inst
#    X_tmf[msk_est] = moy_inst#x_range[ind[0]]
#    
#xrn = np.unique(X_tmf)
#msk0 = X_tmf == xrn.min()
#msk1 = X_tmf == xrn.max()
#msk05 = (msk0==0)*(msk1==0)
#
#X_tmf[msk0] = 0. ; X_tmf[msk05] = 0.5 ; X_tmf[msk1] = 1. ; 
Xt2 = np.copy(X_tmf)
#Xt2[X_tmf==1]=0.33333333 
#Xt2[(X_tmf >0.3) * (X_tmf < 0.4)]=1
Xt2[X_tmf==0]=1
Xt2[X_tmf==1]=0.5
Xt2[X_tmf==0.5]=0.

plt.imshow(Xt2, interpolation='nearest', origin='lower', cmap=cm_gris)#,vmin=0,vmax=1); 
plt.title('$\\hat{\mathbf{x}}_{\mathrm{MPM}}$ ')


plt.subplot(nb_li,nb_col,4)
#plt.imshow(Y.mean(axis=2), interpolation='nearest', origin='lower', cmap=cm_gris);
plt.imshow(V_mpm_est, interpolation='nearest', origin='lower', cmap=cm_angl,vmin=0,vmax=np.pi); 
#plt.imshow(X_mpm_est, interpolation='nearest', origin='lower', cmap=cm_gris,alpha=0.5);
plot_directions(V_mpm_est, np.ones_like(V_mpm_est),pas=4,taille=1.5)

#
#plt.subplot(nb_li,nb_col,6)
#plt.imshow(Ux_hmf, interpolation='nearest', origin='lower', cmap=plt.cm.gray,vmin=0,vmax=1); 
#plt.title('$\\hat{\mathbf{u}}^x$ HMF')
#plt.colorbar(fraction=0.046,pad=0.04)
#  
#plt.subplot(nb_li,nb_col,7)
#plt.imshow(Ux_map, interpolation='nearest', origin='lower', cmap=plt.cm.gray,vmin=0,vmax=1); 
#plt.title('$\\hat{\mathbf{u}}^x$')
#plt.colorbar(fraction=0.046,pad=0.04)
#
#
#plt.subplot(nb_li,nb_col,8)
#plt.imshow(Uv_map, interpolation='nearest', origin='lower',cmap=plt.cm.gray,vmin=0, vmax=1); 
#plt.colorbar(fraction=0.046,pad=0.04)
#plt.title('$\\hat{\mathbf{u}}^v$')


plt.tight_layout()

#np.savez('./results/res_pami_mars2',X_mpm_est = X_tmf, X_mpm_hmf = X_hmf, V_mpm_est = V_mpm_est, Y = Y,Ux_map=Ux_map, Ux_hmf=Ux_hmf, Uv_map=Uv_map, parsem = parsem, parsem_hmf = parsem_hmf)
###

#np.savez('./results/res_pami_vine(map)',X_mpm_est = X_tmf, X_mpm_hmf = X_hmf, V_mpm_est = V_mpm_est, Y = Y,Ux_map=Ux_map, Ux_hmf=Ux_hmf, Uv_map=Uv_map, parsem = parsem, parsem_hmf = parsem_hmf)
#

#%% Comparaison X vrai et segmente
if  (X_hmf != X).mean() > 0.5:
    X_hmf =  1. - X_hmf
    
if  (X_tmf != X).mean() > 0.5:
    X_tmf =  1. - X_tmf
    
    
Ex_hmf = (X_hmf != X).mean()
#Ex_hmf = Ex_hmf*(Ex_hmf < 0.5) + (1-Ex_hmf)*(Ex_hmf > 0.5)

Ex_tmf = (X_tmf != X).mean()
#Ex_tmf = Ex_tmf*(Ex_tmf < 0.5) + (1-Ex_tmf)*(Ex_tmf > 0.5)



#RMSE_hmf = np.sqrt( np.mean( (X_hmf-X)**2 ) )
##RMSE_hmf = RMSE_hmf*(RMSE_hmf < 0.5) + (1-RMSE_hmf)*(RMSE_hmf > 0.5)
#
#RMSE_tmf = np.sqrt( np.mean( (X_tmf-X)**2 ) )
##RMSE_tmf = RMSE_tmf*(RMSE_tmf < 0.5) + (1-RMSE_tmf)*(RMSE_tmf > 0.5)
#
#moy_ux_hmf = np.mean(Ux_hmf[Ux_hmf<=1])
#moy_ux_tmf = np.mean(Ux_map[Ux_map<=1])
#
#std_ux_hmf = np.std(Ux_hmf[Ux_hmf<=1])
#std_ux_tmf = np.std(Ux_map[Ux_map<=1])
#%%
print 'Taux erreur HMF : %.7f, taux erreur TMF : %.7f'%(Ex_hmf*100,Ex_tmf*100)
#print 'RMSE HMF :        %.7f, RMSE TMF :        %.7f'%(RMSE_hmf,RMSE_tmf)
#print 'moy(ux) HMF :     %.7f, moy(ux) TMF :     %.7f'%(moy_ux_hmf , moy_ux_tmf)
#print 'std(ux) HMF :     %.7f, std(ux) TMF :     %.7f'%(std_ux_hmf , std_ux_tmf)


#%%
#plt.figure(figsize=(10,10))
#plt.imshow(Y.mean(axis=2), interpolation='nearest', origin='lower', cmap=cm_gris);#plt.imshow(V_mpm_est, interpolation='nearest', origin='lower', cmap=cm_angl,vmin=0,vmax=np.pi); 
##plt.imshow(X_mpm_est, interpolation='nearest', origin='lower', cmap=cm_gris,alpha=0.5);
#plot_directions(V_mpm_est, np.ones_like(V_mpm_est),pas=4,taille=1.5)
#%%
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(Y.mean(axis=2), interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r,vmin = -1,vmax=1); 
#plt.axis('off')
#plt.savefig('./figures/exp_pamiBy.png', format='png',dpi=200)
#
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(1-X_mpm_est, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r)#,vmin = -1,vmax=1); 
#plt.axis('off')
#plt.savefig('./figures/exp_pamiBx_otmf.png', format='png',dpi=200)
##
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(1-X_mpm_hmf, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r)#,vmin = -1,vmax=1); 
#plt.axis('off')
#plt.savefig('./figures/exp_pamiBx_hmf.png', format='png',dpi=200)
#
#
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(V_mpm_est, interpolation='nearest', origin='lower', cmap=plt.cm.Spectral,alpha=0.25,vmin=0,vmax=np.pi);#plt.imshow(V_mpm_est, interpolation='nearest', origin='lower', cmap=cm_angl,vmin=0,vmax=np.pi); 
#plot_directions(V_mpm_est, np.ones_like(V_mpm_est),pas=4,taille=1.0)
#plt.axis('off')
#plt.savefig('./figures/exp_pamiBv_otmf.png', format='png',dpi=200)


#
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(Ux_map, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r,vmin =0,vmax=1); 
#plt.axis('off')
#plt.savefig('./figures/exp_pamiB_ux.png', format='png',dpi=200)
##
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(Uv_map, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r,vmin = 0,vmax=1); 
#plt.axis('off')
#plt.savefig('./figures/exp_pamiB_uv.png', format='png',dpi=200)
#
#plt.close('all')



#%%

#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#im = Y.mean(axis=2)
#ax.imshow(im, interpolation='nearest', origin='lower', cmap=plt.cm.gray,vmin = im.mean() - 3*im.std(), vmax = im.mean() + 3*im.std())#,vmin = -1,vmax=1); 
#plt.axis('off')
#plt.savefig('./figures/vine2a.png', format='png',dpi=200)
#
#
##%%
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(X_tmf, interpolation='nearest', origin='lower', cmap=plt.cm.Greys,vmin=0); 
#plt.axis('off')
#plt.savefig('./figures/vine2b.png', format='png',dpi=200)
#
##%%
#
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(Ux_map, interpolation='nearest', origin='lower', cmap=plt.cm.gray,vmin=0,vmax=1); 
#plt.axis('off')
#plt.savefig('./figures/vine2c.png', format='png',dpi=200)
##%%
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(1-X_hmf, interpolation='nearest', origin='lower', cmap=plt.cm.Greys,vmin=0); 
#plt.axis('off')
#plt.savefig('./figures/vine2d.png', format='png',dpi=200)
#
##%%
#plt.close('all')
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
##ax.imshow(Y.mean(axis=2), interpolation='nearest', origin='lower', cmap=plt.cm.gray,alpha=0.25);#plt.imshow(V_mpm_est, interpolation='nearest', origin='lower', cmap=cm_angl,vmin=0,vmax=np.pi); 
#ax.imshow(V_mpm_est, vmin=0, vmax=np.pi,interpolation='nearest', origin='lower',cmap=plt.cm.hsv_r,alpha=0.25)
#plot_directions(V_mpm_est, np.ones_like(V_mpm_est),pas=4,taille=1.5)
#plt.axis('off')
#plt.savefig('./figures/vine2e.png', format='png',dpi=200)
#
##%%
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(Uv_map, interpolation='nearest', origin='lower', cmap=plt.cm.gray,vmin=0,vmax=1); 
#plt.axis('off')
#plt.savefig('./figures/vine2f.png', format='png',dpi=200)
##
##%%

#%%
#
#if x_range.size>2:
#    snr_tous = np.zeros_like(x_range) 
#    # snr estime de la region a 1
#    snr_tous = 10*np.log10((np.linalg.norm(x_range[:,np.newaxis]*parsem.mu[:,0],axis=1)**2) /(W*parsem.sig) )   
#
##plt.close('all')
#nb_li = 3
#nb_col =3
##
##plt.rc('text', usetex=True)
##plt.rc('font',family='serif')
#
#
#
#cm_gris = matplotlib.cm.gray
#cm_angl = matplotlib.cm.Spectral
#
#cm_gris.set_bad('r',1.)
##X_mpm_est[X_mpm_est==0]+=np.nan
#cm_angl.set_bad('r',1.)
###cm_gris = matplotlib.cm.gray
###cm_angl = matplotlib.cm.Spectral
##
##cm_gris.set_bad('r',1.)
##cm_angl.set_bad('r',1.)
#
#plt.figure(figsize=(4.5*nb_col,4*nb_li))
#
#
#plt.subplot(nb_li,nb_col,1)
#plt.imshow(Y.mean(axis=2), interpolation='nearest', origin='lower', cmap=cm_gris)#,vmin = -1,vmax=1); 
##plt.axis('off')
#plt.title('$\mathbf{y}$, moyenne spectrale')
#
#plt.subplot(nb_li,nb_col,2)
#plt.imshow(X, interpolation='nearest', origin='lower', cmap=cm_gris,vmin=0); 
##plt.contour(V, v_range.size,colors='g',linewidths=2,alpha=0.75)
##plt.axis('off')
#plt.title('$\mathbf{x}$')
#
#plt.subplot(nb_li,nb_col,3)
#plt.imshow(V, interpolation='nearest', origin='lower', cmap=cm_angl,vmin=0,vmax=np.pi); 
#plot_directions(V, np.ones_like(V),pas=8)
##plt.axis('off')
#plt.title('$\mathbf{v}$')
#
#
#
#
#    
#plt.subplot(nb_li,nb_col,nb_col+2)
#if x_range.size>2:
##    cm_gris_snr = matplotlib.cm.get_cmap('gray',nb_level_x+1)
##    cm_gris_snr.set_bad('w',1.)
##    bounds = snr_tous[~np.isinf(snr_tous)]
#    plt.imshow(X_mpm_est, interpolation='nearest', origin='lower', cmap=cm_gris,vmin=0,vmax=1); 
##            loc = x_range*(nb_level_x-1)/(nb_level_x) #+0.5/(nb_level_x)
##    loc = np.linspace(0.5/nb_level_x,1-0.5/nb_level_x,nb_level_x+1)
###            loc = 0.5/nb_level_x + np.arange(nb_level_x)/float(nb_level_x)
##    cb=plt.colorbar(fraction=0.046,pad=0.04,aspect='auto',shrink=float(S1)/S0)
##    cb.set_ticks(loc)
##    cb.set_label('SNR')
##    cb.set_ticklabels(['{:4.2f}'.format(l) for l in bounds])
#else:
#    plt.imshow(X_mpm_est, interpolation='nearest', origin='lower', cmap=cm_gris,vmin=0,vmax=1); 
##for vr in range(v_range.size):
##    plt.contour(V_mpm_hierarch[:,:,level]==v_range[vr],1,colors='g',linewidths=2,alpha=0.75)
##plt.axis('off')
##    plt.title('$\\hat{\mathbf{x}}_{MPM}^%.0f$'%level)
##erreur_mpm_x = (X != X_mpm_est).mean()
#plt.title('$\\hat{\mathbf{x}}_{\mathrm{MPM}}$ - %.2f '%(erreur(X,X_mpm_est)*100))
#
#
#plt.subplot(nb_li,nb_col,nb_col+3)
#plt.imshow(V_mpm_est, interpolation='nearest', origin='lower',cmap=cm_angl,vmin=0,vmax=np.pi); 
#plot_directions(V_mpm_est, np.ones_like(V_mpm_est),pas=8)
##plt.axis('off')
#plt.title('$\\hat{\mathbf{v}}_{\mathrm{MPM}}$ - %.2f '%(erreur(V,V_mpm_est)*100))
#
#
#if incert==True:
#    
#    
#    plt.subplot(nb_li,nb_col,2*nb_col+2)
#    plt.imshow(Ux_map, interpolation='nearest', origin='lower', cmap=plt.cm.gray,vmin=0,vmax=1); 
#    #plt.axis('off')
#    plt.title('$\\hat{\mathbf{u}}^x$')
#    plt.colorbar(fraction=0.046,pad=0.04)
#    
#    
#    plt.subplot(nb_li,nb_col,2*nb_col+3)
#    plt.imshow(Uv_map, interpolation='nearest', origin='lower',cmap=plt.cm.gray,vmin=0, vmax=1); 
#    plt.colorbar(fraction=0.046,pad=0.04)
#    #plt.axis('off')
#    plt.title('$\\hat{\mathbf{u}}^v$')
#
#
#plt.tight_layout()


#%% let us plot images from earth
#
#plt.figure(figsize=(6,6))
#plt.imshow(Y.mean(axis=2), interpolation='nearest', origin='lower', cmap=plt.cm.terrain)#,vmin = -1,vmax=1); 
#plt.axis('off')
#plt.tight_layout()
##plt.xlim(.50,79.5)
##plt.ylim(0.50,79.5)
#plt.savefig('./figures/vine2a.png', format='png',dpi=200)
#
#plt.figure(figsize=(6,6))
#plt.imshow(X_mpm_est, interpolation='nearest', origin='lower', cmap=plt.cm.gray,vmin=0); 
##plt.xlim(.50,79.5)
##plt.ylim(0.50,79.5)
#plt.axis('off')
#plt.tight_layout()
#plt.savefig('./figures/vine2b.png', format='png',dpi=200)
#
##f4 = plt.figure(figsize=(6,6))
#data = Y.mean(axis=2)
#sizes = np.shape(data)
#height = float(sizes[0])
#width = float(sizes[1])
#
#fig = plt.figure(figsize=(6,6))
#plt.imshow(Y.mean(axis=2), interpolation='nearest', origin='lower', cmap=plt.cm.terrain,alpha=0.5);#plt.imshow(V_mpm_est, interpolation='nearest', origin='lower', cmap=cm_angl,vmin=0,vmax=np.pi); 
#plot_directions(V_mpm_est, np.ones_like(V_mpm_est),pas=2,taille=0.75)
##plt.xlim(.50,79.5)
##plt.ylim(0.50,79.5)
#plt.axis('off')
#plt.tight_layout()
#plt.savefig('./figures/vine2e.png', format='png',dpi=200)
#
#
#fig = plt.figure(figsize=(6,6))
#plt.imshow(Ux_map, interpolation='nearest', origin='lower', cmap=plt.cm.gray,vmin=0,vmax=1); 
#plt.axis('off')
#plt.tight_layout()
#plt.savefig('./figures/vine2c.png', format='png',dpi=200)
#
#fig = plt.figure(figsize=(6,6))
#plt.imshow(Uv_map, interpolation='nearest', origin='lower', cmap=plt.cm.gray,vmin=0,vmax=1); 
#plt.axis('off')
#plt.tight_layout()
#plt.savefig('./figures/vine2f.png', format='png',dpi=200)

#%%
##%% let us plot images from mars
#plt.figure(figsize=(9,3))
#
#
#
#plt.subplot(131)
#plt.imshow(Y.mean(axis=2), interpolation='nearest', origin='lower', cmap=plt.cm.copper_r)#,vmin = -1,vmax=1); 
##plt.axis('off')
#plt.title('$Y = y$')
#plt.xlim(.50,79.5)
#plt.ylim(0.50,79.5)
#
#plt.subplot(132)
#plt.imshow(X_mpm_est, interpolation='nearest', origin='lower', cmap=plt.cm.gray,vmin=0); 
##plt.contour(V, v_range.size,colors='g',linewidths=2,alpha=0.75)
##plt.axis('off')
#plt.title('$\\hat{x}^{\mathrm{MPM}}$')
#plt.xlim(.50,79.5)
#plt.ylim(0.50,79.5)
#
#plt.subplot(133)
#plt.imshow(Y.mean(axis=2), interpolation='nearest', origin='lower', cmap=plt.cm.copper_r,vmin=0,alpha=0.5);#plt.imshow(V_mpm_est, interpolation='nearest', origin='lower', cmap=cm_angl,vmin=0,vmax=np.pi); 
#plot_directions(V_mpm_est, np.ones_like(V_mpm_est),pas=2,taille=0.75)
#plt.xlim(.50,79.5)
#plt.ylim(0.50,79.5)
#plt.title('$\\hat{v}^{\mathrm{MPM}}$')
#
#plt.tight_layout()
#plt.savefig('./figures/mars1.png', format='png',dpi=200)
#%%
#
#
#X_tpm_est = X_mpm.mean(axis=2)
#V_tpm_est  =V_mpm.sum(axis=2)
#
#plt.subplot(nb_li,nb_col,2*nb_col+2)
#plt.imshow(X_tpm_est, interpolation='nearest', origin='lower', cmap=cm_gris,vmin=0,vmax=1); 
#plt.title('$\\hat{\mathbf{x}}_{\mathrm{TPM}}$ ')
#
#
#plt.subplot(nb_li,nb_col,2*nb_col+3)
#plt.imshow(V_tpm_est, interpolation='nearest', origin='lower',cmap=cm_angl,vmin=0,vmax=np.pi); 
#plot_directions(V_tpm_est, np.ones_like(V_tpm_est),pas=8)
#plt.axis('off')


#plt.savefig('./figures/galtout.png', format='png',dpi=200)

#%%
#
#plt.figure(figsize=(4.5*3,4))
#
#X_tpm_est = X_mpm.mean(axis=2)
#V_tpm_est  =V_mpm.mean(axis=2)%np.pi
#
#ecart_x = np.abs(X_tpm_est-X/5.)
#
#plt.subplot(131)
#plt.imshow(X_tpm_est, interpolation='nearest', origin='lower', cmap=cm_gris,vmin=0,vmax=1); 
#plt.title('$\\hat{\mathbf{x}}_{\mathrm{TPM}}$ ')
#
#plt.subplot(132)
#plt.imshow(ecart_x, interpolation='nearest', origin='lower', cmap=cm_gris,vmin=0,vmax=1); 
#plt.title('ecart absolu')
#
#plt.subplot(133)
#plt.imshow(V_tpm_est, interpolation='nearest', origin='lower',cmap=cm_angl,vmin=0,vmax=np.pi); 
#plot_directions(V_tpm_est, np.ones_like(V_tpm_est),pas=8)
#plt.title('$\\hat{\mathbf{v}}_{\mathrm{TPM}}$ ')
##plt.axis('off')
#
#plt.tight_layout()

#%%
#from mpl_toolkits.mplot3d import Axes3D
#
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#
#
#X = np.arange(0,S0,1)
#Y = np.arange(0,S1,1)
#X,Y = np.meshgrid(X,Y)
#surf = ax.plot_surface(X,Y,ecart_x,cmap=plt.cm.coolwarm)
#ax.set_zlim(0,1)
#
#plt.show()

#%%
#