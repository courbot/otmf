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
import fields_tools as ft
import seg_OTMF as sot
import SEM as sem

from scipy.ndimage import zoom
from scipy.ndimage.filters import gaussian_filter 
from scipy.ndimage.filters import median_filter 

import gdal

import spectral.io.envi as envi

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
    
def permut_bruit(bruit):
# taille du bloc: 10*10*W
    W = bruit.shape[2]
    tb=np.array([10,10,W])
    ti = np.array([50,50,W])
    
    indices = np.zeros(shape=(ti/tb))
    indices = np.arange(indices.size).reshape(indices.shape)
    
    indices_permut = np.random.permutation(indices.flatten()).reshape(indices.shape)
    
    ind, ind_p = np.zeros(shape=(50,50,W)),np.zeros(shape=(50,50,W))
    
    for x in range(5):
        for y in range(5):
            ind[x*10:(x+1)*10, y*10:(y+1)*10,:] = indices[x,y]
            ind_p[x*10:(x+1)*10, y*10:(y+1)*10,:] = indices_permut[x,y]
    
    bruit_permut=np.copy(bruit)
    for i in range(indices.size):
            bruit_permut[ind_p==i] = bruit[ind==i]
    
    return bruit_permut
#
#def gen_obs(X,W,x_range, m,sig,rho_1,rho_2,corrnoise=False):
#    
#
#     
#    ###%% Creation de l'observation, hyperspectrale
#    mu = gen_mu(W)
#    ##
#    ###%% Generation observation y
#    ### Valable dans le cas ou le bruit est independant !
#    # => faire des options pour bruit corrélé ?
#    
#    if corrnoise==True:
#        Sigma = np.eye(W) * sig**2 + (np.eye(W,k=1) +  np.eye(W,k=-1)) * rho_1 + (np.eye(W,k=2) +  np.eye(W,k=-2)) * rho_2    
#        
#    else:
#        Sigma = np.eye(W) * sig**2
#    
#    if W ==1 :
#        Y = np.zeros(shape=(S0,S1,1))
#        Y_tmp = np.zeros(shape=(S0,S1))
#        for id_x in range(x_range.size):
#            bruit_tout = np.random.normal(loc=0.,scale=sig[id_x],size=(S0,S1))
#            Y_tout = X*mu + bruit_tout
#            Y_tmp[X==x_range[id_x]] = Y_tout[X==x_range[id_x]]
#        Y[:,:,0] = Y_tmp
##            Y[:,:,0] = X*mu + np.random.normal(loc=0.,scale=sig,size=(S0,S1))
#    else:
#        Y = X[:,:,np.newaxis]*mu[np.newaxis,np.newaxis,:] +np.random.multivariate_normal(mean=np.zeros_like(mu),cov=Sigma,size=(S0,S1))
#    
#    #Y = X[:,:,np.newaxis]*mu[np.newaxis,np.newaxis,:] + st.norm.rvs(loc=m,scale=sig,size=(S0,S1,W))
#    
##    pargibbs.Y = Y
#    
#    return Y
    
#%%  

experiment = '8'


synth = False
 
if synth:
    S0 = 128
    S1 = 128    
    W = 20
    
else:
    S0,S1,W,W_long  = 50,50,20,100
    

x,y = np.ogrid[0:S0,0:S1]

centre = np.array([S0/2,S1/2])
if synth:
    ecartt = 30.
else:
    ecartt = 15.
xr,yr = x-centre[0], y-centre[1]


# source en 2D : "carte d'abondance" = gaussienne tronquee
src_2d = np.exp( - (xr**2+yr**2)/ecartt**2  )
if synth:
    lim = 50.
else:
    lim = 18.
msk_pres = (xr**2+yr**2)<lim**2
src_2d[msk_pres==0]=0


# source en 3D : cube sans bruit

if synth:
    mu = np.zeros(shape=W)
    mu[7:12] = np.array([0.7, 1.,0.8, 0.6,0.3])
else:
    mu = np.zeros(shape=W_long)
    mu[45:55] = np.array([0,0.4,1.2,1.5,1.3,1.05,0.75,0.6,0.4,0.15])/1.5
src_3d = src_2d[:,:,np.newaxis] * mu[np.newaxis,np.newaxis,:]


# observation : source = bruit
#range_psnr = np.array([-15.])
#range_psnr = np.array([-13.5,-11.5])
#range_psnr=np.array([-12.,-9.,-6.,-3.,0.])

#range_psnr =  np.arange(-15.,1.5,3)
range_psnr = np.arange(-13.5,1.5,3)
#range_psnr = np.arange(-10,2,2)
nb_psnr = range_psnr.size

dirname = './results/astro_hmf/synth/'
dirname = './results/astro_hmf/udf10/'

std_ref = 1.
rho_1,rho_2 = 0.05,0.025
covmat=np.eye(W)*std_ref**2 + (np.eye(W,k=1)+np.eye(W,k=-1))*rho_1 + (np.eye(W,k=2)+np.eye(W,k=-2))*rho_2


nb_essai = 100
for essai in range(nb_essai):
    
    if synth:
        bruit_ref = np.random.multivariate_normal(mean=np.zeros_like(mu),cov=covmat,size=src_2d.shape)
    else:
        
        dat = np.load('./data/emptyudf0.npz')
        bruit_long = dat['Y']
        #bruit_ref = dat['Y']
        bruit_long = bruit_long/bruit_long.std()
        # permutation de bloc du bruit, taille 10x10
        bruit_long = permut_bruit(bruit_long)

    for id_psnr in range(nb_psnr):

            psnr=range_psnr[id_psnr]
            
            if synth:
                std_spec = 10.**(-psnr/20.) * np.linalg.norm(mu)/np.sqrt(W)
            else:
                std_spec = 10.**(-psnr/20.) * np.linalg.norm(mu[40:60])/np.sqrt(W)
           
            Y_long = src_3d + std_spec * bruit_long
            
            
            # soustraction moyenne...?
            Y_med = np.median(Y_long,axis=2)
            Y = Y_long[:,:,40:60] - Y_med[:,:,np.newaxis]
#            Ymax = Y.max()            
#            Y = Y/Ymax# Y.mean()            
            
            snr_map = 10*np.log10((np.linalg.norm(src_3d,axis=2)**2) /(W*std_spec**2) )
            
            
            pas = np.pi/6
            v_range = np.arange(pas/2., np.pi, pas)
            
    #            nb_level_x =2
                
            for nb_level_x in (1,2,3,4,):
                print 'Version numero %.0f'%essai
                print 'Peak SNR : %.1f dB'%psnr
                print 'Nb level x : %.1f'%nb_level_x
                
                nblx = nb_level_x
                x_range = np.arange(0, 1+1./nblx, 1./nblx)
                
                
                
                print('---------------------------------------')
                pargibbs = parameters.ParamsGibbs(S0 = S0,                              S1 = S1,
                                             type_pot = 'potts',                              phi_uni = 0.,
                                             thr_conv=0.001,                             nb_iter=100,
                                             fuzzy=False,                             anisotropic=True,   
                                             angle=np.zeros(shape=(S0,S1)),
                                             beta = 1.,                             phi_theta_0 = 0.,
                                             alpha =5,                             alpha_v = 10,
                                             delta = 0.,                             init_method = 'std',
                                             nb_fuzzy = 256. ,                             v_range = v_range,
                                             x_range = x_range                             )# beta=1.25,
                
                pargibbs.S0 = S0
                pargibbs.S1 = S1
                pargibbs.W = W
                
                pargibbs.Y = Y
                
                
                
                #==============================================================================
                # Paramètres à fixer
                #==============================================================================
                
                incert = True # Utilisation ou non de segmentation avec incertitude
                
                parseg = parameters.ParamsSeg(nb_iter_sem=40,
                                              seuil_conv = 1*(1./S0*S1),
                                              incert = incert
                                                )
                parseg.spec_snr=True
#                valmin_erg = 1./nb_level_x
#                valmax_erg = valmin_erg*nblx
#                parseg.facteur = valmax_erg/Ymax
                parseg.facteur=1.
                parseg.multi = False # le multiclasse discret
                parseg.seuil_conv = 0.05
                parseg.weights = np.ones(shape=(Y.shape[0],Y.shape[1]))
                
                #
                ##==============================================================================
                ## Segmentation HMF
                ##==============================================================================
                parseg.tmf = False
                try:
                    pargibbs = parameters.apply_parseg_pargibbs(parseg,pargibbs) # transfetrt a l'autre jeu de parametre
                    print '---------------HMF---------------------'
                    start = time.time()
                    
                    X_mpm_hmf,V_mpm_hmf,Ux_hmf,Uv_hmf, parsem_hmf = sot.seg_otmf(parseg,pargibbs)
                    
                    end = time.time() - start
                    print 'Temps total : %.2f s'%end  
                    print '------------------------------------'
                    filename=dirname+'nb'+str(nb_level_x)+'_psnr'+str(psnr)+'_fa'+str(parseg.facteur)+'_v'+str(essai)
                    np.savez(filename, X_mpm_hmf=X_mpm_hmf, Ux_hmf=Ux_hmf,parsem_hmf=parsem_hmf,snr_map=snr_map,msk_pres=msk_pres,nb_level_x=nb_level_x,psnr=psnr)
                except:
                    print 'erreur'
                    pass

#%%

nb_li=2
nb_col=3
cg = plt.cm.gray_r
plt.figure(figsize=(nb_col*4,nb_li*4))
plt.subplot(nb_li,nb_col,1)
plt.imshow(src_2d, interpolation='nearest', origin='lower', cmap=cg,vmin=0,vmax=1)#,vmin=0,vmax=1);
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(nb_li,nb_col,2)
plt.imshow(snr_map, interpolation='nearest', origin='lower', cmap=plt.cm.Spectral,vmin=-20)#,vmin=0-sig,vmax=1+sig);
plt.colorbar(fraction=0.046, pad=0.04)


plt.subplot(nb_li,nb_col,(3))
plt.plot(mu)
#plt.plot(parsem_hmf.mu,'r')
plt.plot(parsem_hmf.mu[0,:],'r')


plt.subplot(nb_li,nb_col,4)
plt.imshow(Y.mean(axis=2), interpolation='nearest', origin='lower', cmap=cg)#,vmin=0-sig,vmax=1+sig);
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(nb_li,nb_col,5)
plt.imshow(X_mpm_hmf, interpolation='nearest', origin='lower', cmap=cg)#,vmin=0-sig,vmax=1+sig);
#plt.imshow(msk_pres, interpolation='nearest', origin='lower', cmap=plt.cm.jet,alpha=0.5)#,vmin=0-sig,vmax=1+sig);
plt.colorbar(fraction=0.046, pad=0.04)
plt.imshow(snr_map, interpolation='nearest', origin='lower', cmap=plt.cm.Spectral,vmin=-20,alpha=0.05)

plt.tight_layout()

#%%
#
#synth = False
# 
#if synth:
#    S0 = 128
#    S1 = 128    
#    W = 20
#    
#else:
#    S0,S1,W,W_long  = 50,50,20,100
#    
#
#x,y = np.ogrid[0:S0,0:S1]
#
#centre = np.array([S0/2,S1/2])
#if synth:
#    ecartt = 30.
#else:
#    ecartt = 10.
#xr,yr = x-centre[0], y-centre[1]
#
#%%
# source en 2D : "carte d'abondance" = gaussienne tronquee
#src_2d = np.exp( - (xr**2+yr**2)/ecartt**2  )
#fig=plt.figure(figsize=(4,4))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#plt.axis('off')
#plt.imshow(src_2d, origin='lower',vmin=-1,vmax=1,cmap=plt.cm.gray_r,interpolation='nearest')
#
#plt.savefig('./figures/musesynth_source.png', format='png',dpi=200)
#
##%%
#mu = np.zeros(shape=W)
#mu[5:15]=np.array([0,0.4,1.2,1.5,1.3,1.05,0.75,0.6,0.4,0.15])/1.5
#std_ref = 1.
#rho_1,rho_2 = 0.05,0.025
#covmat=np.eye(W)*std_ref**2 + (np.eye(W,k=1)+np.eye(W,k=-1))*rho_1 + (np.eye(W,k=2)+np.eye(W,k=-2))*rho_2
#bruit_ref = np.random.multivariate_normal(mean=np.zeros_like(mu),cov=covmat,size=src_2d.shape)
#
#fig=plt.figure(figsize=(4,4))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#plt.axis('off')
#plt.imshow(bruit_ref.mean(axis=2), origin='lower',cmap=plt.cm.gray_r,interpolation='nearest',vmin=-1,vmax=1)
#plt.savefig('./figures/musesynth_noise.png', format='png',dpi=200)

#%%
#fig=plt.figure(figsize=(4,4))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#plt.axis('off')
#plt.hist(bruit_ref.flatten(),50)
#plt.savefig('./figures/musesynth_histnoise.png', format='png',dpi=200)
#%%

#dat = np.load('./data/emptyudf0.npz')
#bruit_long = dat['Y']
##bruit_ref = dat['Y']
#bruit_long = bruit_long/bruit_long.std()
#
#
#
#bruit_ref = bruit_long[:,:,40:60]
#
#fig=plt.figure(figsize=(4,4))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#plt.axis('off')
##vmin,vmax = bruit_ref.mean(axis=2).T.min(),bruit_ref.mean(axis=2).T.max()
#plt.imshow(bruit_ref.mean(axis=2).T, origin='lower',cmap=plt.cm.gray_r,interpolation='nearest',vmin=-1,vmax=1)
#plt.savefig('./figures/museudf_noise.png', format='png',dpi=200)

#%%

#dat = np.load('./data/emptyudf1.npz')
#bruit_long = dat['Y']
##bruit_ref = dat['Y']
#bruit_long = bruit_long/bruit_long.std()
#
#
#
#bruit_ref = bruit_long[:,:,40:60]
#
#fig=plt.figure(figsize=(4,4))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#plt.axis('off')
#plt.imshow(bruit_ref.mean(axis=2).T, origin='lower',cmap=plt.cm.gray_r,vmin=-1,vmax=1,interpolation='nearest')#,vmin=vmin,vmax=vmax)
#plt.savefig('./figures/museudf_noise2.png', format='png',dpi=200)

#%%
#fig=plt.figure(figsize=(4,4))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#plt.axis('off')
#plt.hist(bruit_ref.flatten(),50)
#plt.savefig('./figures/museudf_histnoise.png', format='png',dpi=200)
#%%
#fig=plt.figure(figsize=(4,2))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#plt.plot(mu,linewidth=4)
#plt.xlim(0,19)
#plt.axis('off')
#plt.savefig('./figures/musesynth_mu.png', format='png',dpi=200)
#



#%%



#%% On construit une courbe ROC
#
#true = msk_pres
#clas = X_mpm_hmf > 0
#
#true_positive = np.mean((true==1)*(clas==1))/np.mean(true==1)
#false_positive = np.mean((true==0)*(clas==1))/np.mean(true==0)
#true_negative = np.mean((true==0)*(clas==0))/np.mean(true==0)
#false_negative = np.mean((true==1)*(clas==0))/np.mean(true==1)
#
#found = np.mean(true==clas)
#missed = np.mean(true!=clas)
#
#print 'True positive: %.2f ; False positive: %.2f'%(true_positive,false_positive)
#print 'False negative: %.2f ; True negative: %.2f'%(false_negative,true_positive)
#print ' '
#print 'Prop retrouve: %.2f ;   Prop. manque: %.2f'%(found,missed)
#
#
##%%
#nb_level=10
#smax = snr_map[msk_pres].max()
#smin = snr_map[msk_pres].min()
#mid_step = (smax-smin)/nb_level
#bornes = np.arange(smin,smax+mid_step,mid_step)
#
#tp,fp,tn,fn = np.zeros(shape=nb_level),np.zeros(shape=nb_level),np.zeros(shape=nb_level),np.zeros(shape=nb_level)
#found, missed = np.zeros(shape=nb_level),np.zeros(shape=nb_level)
#
#for level in range(nb_level):
#    
#    msk_level = (snr_map > bornes[level])*(snr_map<=bornes[level+1])
#    true = msk_pres[msk_level]
#    clas = X_mpm_hmf[msk_level] > 0
#    
#    tp[level] = np.mean((true==1)*(clas==1))/np.mean(true==1)
#    fp[level] = np.mean((true==0)*(clas==1))/np.mean(true==0)
#    tn[level] = np.mean((true==0)*(clas==0))/np.mean(true==0)
#    fn[level] = np.mean((true==1)*(clas==0))/np.mean(true==1)
#    
#    found[level] = np.mean(true==clas)
#    missed[level] = np.mean(true!=clas)
#
##%%
#plt.figure(figsize=(8,6))
#
##for id_psnr in range(nb_psnr):
##    psnr=range_psnr[id_psnr]
##
##    snraxis = bornes[:-1]+1.5 # car pas de 3 dans ces intervalles
#plt.plot(range_psnr,found,'.:',linewidth=4)
###plt.plot(snraxis,tp,'b',label='true positive',)
###plt.plot(snraxis,fn,'g',label='false negative',)
####plt.plot(snraxis,missed,'r',label='missed',)
#plt.ylim((0,1))
#plt.grid()
###plt.xlim(snraxis[0],snraxis[-1])
#plt.xlabel('peak-SNR (dB)')
##plt.legend(loc='best',title='Peak SNR')
#
#%%
#plt.figure(figsize=(8,6))
#snraxis = bornes[:-1]+1.5
#for id_psnr in range(nb_psnr):
#    psnr=range_psnr[id_psnr]
#
#    snraxis = bornes[:-1]+1.5 # car pas de 3 dans ces intervalles
#    plt.plot(snraxis,found[id_psnr,:],'.:',label='%.0f dB'%psnr,linewidth=4)
##plt.plot(snraxis,tp,'b',label='true positive',)
##plt.plot(snraxis,fn,'g',label='false negative',)
###plt.plot(snraxis,missed,'r',label='missed',)
#plt.ylim((0,1))
#plt.grid()
##plt.xlim(snraxis[0],snraxis[-1])
#plt.xlabel('SNR (dB)')
#plt.legend(loc='best',title='Peak SNR')
#

#%%

#%%
#real = msk_pres
#classif = X_mpm_hmf>0.
#
#pd = np.mean((classif==1)*(real==1))/np.mean(real==1)
#pf = np.mean((classif==1)*(real==0))/np.mean(real==1)
#print 'PD : %.3f ; PF : %.3f'%(pd,pf)
#Pd = np.zeros(shape=nb_level_x)
#Pf = np.copy(Pd)
#i=0
#for x_inst in x_range[x_range!=0]:
#    msk = Xres==x_inst
#    classif = (Xres==x_inst)
#    real = msk_pres +0.
##    msk_ok =  
##    msk_fa =  (Xres[msk]>0)*(msk_pres[msk]==0)
#    Pd[i] = np.mean(classif[msk]*real[msk])/np.mean(real[msk])
#    Pf[i] = np.mean(classif*(real==0))/np.mean(real)
#    i+=1
#    
#plt.figure()
#plt.plot(Pf,Pd,'.',markersize=10)
#plt.xlim((0,0.25))
#plt.ylim((0.5,1.))
#%% random permutations

#
#dat = np.load('./data/emptyudf0.npz')
#bruit_long = dat['Y']
##bruit_ref = dat['Y']
#bruit_long = bruit_long/bruit_long.std()
#
#bruit = bruit_long[:,:,40:60]
##
##
##
##
#
## copie du contenu de la fonction permut_bruit
#cmi = plt.cm.jet
#tb=np.array([10,10,W])
#ti = np.array([50,50,W])
#
#indices = np.zeros(shape=(ti/tb))
#indices = np.arange(indices.size).reshape(indices.shape)
#
#indices_permut = np.random.permutation(indices.flatten()).reshape(indices.shape)
#
#ind, ind_p = np.zeros(shape=(50,50,W)),np.zeros(shape=(50,50,W))
#
#for x in range(5):
#    for y in range(5):
#        ind[x*10:(x+1)*10, y*10:(y+1)*10,:] = indices[x,y]
#        ind_p[x*10:(x+1)*10, y*10:(y+1)*10,:] = indices_permut[x,y]
#
#bruit_permut=np.copy(bruit)
#for i in range(indices.size):
#        bruit_permut[ind_p==i] = bruit[ind==i]
##%%
#fig=plt.figure(figsize=(4,4))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#plt.imshow(bruit.mean(axis=2).T,cmap=plt.cm.gray_r,vmin=-1,vmax=1,interpolation='nearest',origin='lower',extent=[0.,50,0.,50])
#plt.imshow(ind.mean(axis=2),cmap=cmi,interpolation='nearest',alpha=0.2,origin='lower',extent=[0.,50,0.,50])
#
#plt.xlim(0,49);plt.ylim(0,49)
##plt.axis('off')
#plt.grid(linestyle='-', linewidth=2)
#plt.savefig('./figures/musenoise_noperm2.png', format='png',dpi=200)
#
#
#
#fig=plt.figure(figsize=(4,4))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#plt.imshow(bruit_permut.mean(axis=2).T,cmap=plt.cm.gray_r,vmin=-1,vmax=1,interpolation='None',origin='lower',extent=[0.,50,0.,50])
#plt.imshow(ind_p.mean(axis=2),cmap=cmi,interpolation='None',alpha=0.2,origin='lower',extent=[0.,50,0.,50])
#plt.grid(linestyle='-', linewidth=2)
#plt.xlim(0,49);plt.ylim(0,49)
#
##plt.axis('off')
#plt.savefig('./figures/musenoise_perm2.png', format='png',dpi=200)



