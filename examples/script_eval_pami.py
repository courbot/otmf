# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 08:53:52 2016

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

import scipy.ndimage.filters as fi
from os import listdir
import glob


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
nom_fol = './results/pami/PAMImapmpm/exp'


experiment='3' # B dans le papier
sigma=0.5
superv = True#False

#


li = 0
tout_res = np.zeros(shape=(16,14))

    
for superv in (False,):
    if superv: n3 = 'known'; 
    else: n3='unknown'

    for experiment in ('3',):#('2','3'):
        if experiment=='3': n1 = 'b'; 
        else: n1='a'
    

        if experiment=='3':
            dat = np.load('./data/exp_pami3v2.npz')
            X=(dat['X']>0).astype(float)
            V=dat['V']
        else:
            dat = np.load('./data/exp_pamiA.npz')
            X=(dat['X']>0).astype(float)
            V = np.pi/4 * np.ones(shape=(128,128))
            V[64:,:] = 3*np.pi/4
            V[:,64:] = (V[:,64:] + np.pi/2)%np.pi
        
        for sigma in (1.0,):
            if sigma==0.5: n2 = '05'; 
            else: n2='10'
            
            for mpm in (True,):
                if mpm: n0=''
                else: n0='map'
                
                print ' '
                print 'superv'+str(superv) +' exp.' + experiment +  ' sig' + str(sigma)

          
                nom_can = nom_fol+n0+n1+'_sig'+n2+'_'+n3
                print nom_can
    #            print 'sauvegarde sous ' + nom_can
    
                listfile = glob.glob(nom_can+'*')
                numel = np.size(listfile)
                
                res_tout = np.zeros(shape=(numel,13) )
                i = 0
                for filename in listfile:
    #                print filename
                
                    d =  np.load(filename)
                    
                    X_tmf = d['X_mpm_est']
                    V_tmf = d['V_mpm_est']
                    X_hmf = d['X_mpm_hmf']
                    
                    Ux_tmf = d['Ux_map']
                    Uv_tmf = d['Uv_map']
                    
                    Ux_hmf = d['Ux_hmf']
                    
                    
                    
                    if  (X_hmf != X).mean() > 0.5:
                        X_hmf =  1. - X_hmf
                        
                    if  (X_tmf != X).mean() > 0.5:
                        X_tmf =  1. - X_tmf
                        
                        
                    Ex_hmf = (X_hmf != X).mean()
                    
                    moy_ux_hmf = np.mean(Ux_hmf[Ux_hmf<=1])
                    std_ux_hmf = np.std(Ux_hmf[Ux_hmf<=1])
                    #Ex_hmf = Ex_hmf*(Ex_hmf < 0.5) + (1-Ex_hmf)*(Ex_hmf > 0.5)
                    
                    
                    moy_ux_tmf = np.mean(Ux_tmf[Ux_tmf<=1])
                    std_ux_tmf = np.std(Ux_tmf[Ux_tmf<=1])
                    
                    moy_uv_tmf = np.mean(Uv_tmf[Uv_tmf<=1])
                    std_uv_tmf = np.std(Uv_tmf[Uv_tmf<=1])                
                    
                    
                    
                    Ex_tmf = (X_tmf != X).mean()
                    msk = (Ux_tmf > moy_ux_tmf) * (Ux_tmf <= 1)
                    prop_corr_x = msk.sum()/(128.**2)
                    Ex_cor = (X_tmf[msk] != X[msk]).mean()
                #    Ev_tmf = (V_tmf != V).mean()
                    if experiment=='2':
                        Ev_tmf = (V_tmf != V).mean()
                        msk = (Uv_tmf > moy_uv_tmf) * (Uv_tmf <= 1)
                        prop_corr_v = msk.sum()/(128.**2)
                        Ev_cor = (V_tmf[msk] != V[msk]).mean()
                        
                #        ecart_relatif = (V_tmf-V)/V
                #        Rmv_tmf = np.sqrt((ecart_relatif**2).mean())
                    else:
                        Ev_tmf = 0
                        Rmv_tmf = 0
                        prop_corr_v = 0
                        Ev_cor = 0
                        
                        
                   
    
                    
                    
                    res_tout[i,:] = [Ex_hmf, moy_ux_hmf, std_ux_hmf,Ex_tmf, moy_ux_tmf, std_ux_tmf, Ev_tmf,moy_uv_tmf,std_uv_tmf, Ex_cor,prop_corr_x,Ev_cor,prop_corr_v]    
                    i+=1
    
                rtm = res_tout.mean(axis=0)
                tout_res[li,1:] = rtm
                tout_res[li,0] = res_tout.shape[0]
                li +=1
                print 'nb. elt: %.0f'%res_tout.shape[0]
                print 'Taux erreur HMF : %.2f'%(rtm[0]*100)
                print 'moy(ux) HMF :     %.3f'%rtm[1]
                print 'std(ux) HMF :     %.3f'%rtm[2]
                
                print 'Taux erreur TMF : %.2f'%(rtm[3]*100)
                #    print 'RMSE HMF :        %.7f, RMSE TMF :        %.7f'%(RMSE_hmf,RMSE_tmf)
                #print 'moy(ux) HMF :     %.7f'%rtm[1]
                print 'moy(ux) TMF :     %.3f'%(rtm[4])
                #print 'std(ux) HMF :     %.7f'%rtm[2]
                print 'std(ux) TMF :     %.3f'%(rtm[5])
                print 'Taux erreur TMF V : %.2f'%(rtm[6]*100)
                print ' '
                print 'moy(uv) TMF :     %.3f'%(rtm[7])
                print 'std(uv) TMF :     %.3f'%(rtm[8])
                print ' '
                print 'Taux erreur TMF X corr : %.2f (garde=%.0f)'%(rtm[9]*100,rtm[10]*100)
                print 'Taux erreur TMF V corr : %.2f (garde=%.0f)'%(rtm[11]*100,rtm[12]*100)

#
#print 'Erreur corrigee : %.7f'%(rtm[9]*100)
tout_res[:,1] *=100 ;  tout_res[:,4] *=100 ; tout_res[:,7] *=100 ; tout_res[:,12] *=100 
#%%
#np.savetxt('./results/results_pami.txt', tout_res,fmt='%.3f', delimiter = '\t',
#           header="\t E_hmf\t Ux_m \t Ux_s \t E_tmf \t Ux_m \t Ux_s \t Ev_tmf\t Uv_m \t Uv_s \t")




#filename = nom_fol + 'b_sig10_unknown'+str(id_hmf)+'.npz'

#%%
#==============================================================================
# Quelques affichages
#==============================================================================


#==============================================================================
# # Sauvegarde resultats experience B
#==============================================================================
#id_hmf = np.argmin(np.abs(res_tout[:,0]-rtm[0]))
#filename = nom_fol + 'b_sig10_unknown'+str(id_hmf)+'.npz'
#d =  np.load(filename)
##    
#
##    
#Y = d['Y']
##            
##X_tmf = d['X_mpm_est']
##V_tmf = d['V_mpm_est']
##X_hmf = d['X_mpm_hmf']
##
##Ux_tmf = d['Ux_map']
##Uv_tmf = d['Uv_map']
##
##Ux_hmf = d['Ux_hmf']
##
##
#plt.close('all')
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(Y.mean(axis=2), interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r,vmin = -3,vmax=4); 
#plt.axis('off')
#plt.savefig('./figures/exp_pamiBy.png', format='png',dpi=200)
###
##
#
#
#
#X_hmf = d['X_mpm_hmf']
#Ux_hmf = d['Ux_hmf']
#
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(1-X_hmf, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r)#,vmin = -1,vmax=1); 
#plt.axis('off')
#plt.savefig('./figures/exp_pamiBx_hmf.png', format='png',dpi=200)
##
#
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(Ux_hmf, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r,vmin =0,vmax=1); 
#plt.axis('off')
#plt.savefig('./figures/exp_pamiB_ux_hmf.png', format='png',dpi=200)
#
##%%
#
#import cmocean
id_tmf = np.argmin(np.abs(res_tout[:,3]-rtm[3]))
filename = nom_fol + 'b_sig10_unknown'+str(id_tmf)+'.npz'
d =  np.load(filename)
##    
##            
#X_tmf = d['X_mpm_est']
V_tmf = d['V_mpm_est']
#
#
#Ux_tmf = d['Ux_map']
#Uv_tmf = d['Uv_map']
##
#
##
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(1-X_tmf, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r)#,vmin = -1,vmax=1); 
#plt.axis('off')
#plt.savefig('./figures/exp_pamiBx_otmf.png', format='png',dpi=200)
####
#
###
fig = plt.figure(figsize=(6,6))
ax = plt.Axes(fig, [0.,0.,1.,1.])
fig.add_axes(ax)
ax.imshow(V_tmf, vmin=0, vmax=np.pi,interpolation='nearest', origin='lower',cmap=plt.cm.inferno,alpha=0.5)

V_fi = fi.median_filter(V_tmf, size=8)
plot_directions(V_fi, np.ones_like(V_tmf),pas=8,taille=2)
plt.axis('off')
plt.savefig('./figures/exp_pamiBv_otmf.png', format='png',dpi=200)


#
###
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(Ux_tmf, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r,vmin =0,vmax=1); 
#plt.axis('off')
#plt.savefig('./figures/exp_pamiB_ux.png', format='png',dpi=200)
####
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(Uv_tmf, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r,vmin = 0,vmax=1); 
#plt.axis('off')
#plt.savefig('./figures/exp_pamiB_uv.png', format='png',dpi=200)

#plt.close('all')

#%%

#==============================================================================
# Images experience A
#==============================================================================
#
#plt.close('all')
#id_hmf = np.argmin(np.abs(res_tout[:,0]-rtm[0]))
#filename = nom_fol + 'a_sig10_unknown'+str(id_hmf)+'.npz'
#d =  np.load(filename)
#    

##    
#Y = d['Y']
#
##
##
##plt.close('all')
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(Y.mean(axis=2), interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r,vmin = -3,vmax=4); 
#plt.axis('off')
#plt.savefig('./figures/exp_pamiAy.png', format='png',dpi=200)
##
#
#
#
#
#X_hmf = d['X_mpm_hmf']
#Ux_hmf = d['Ux_hmf']
#
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(1-X_hmf, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r)#,vmin = -1,vmax=1); 
#plt.axis('off')
#plt.savefig('./figures/exp_pamiAx_hmf.png', format='png',dpi=200)
##
#
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(Ux_hmf, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r,vmin =0,vmax=1); 
#plt.axis('off')
#plt.savefig('./figures/exp_pamiA_ux_hmf.png', format='png',dpi=200)
#
##%%
#import cmocean

#id_tmf = np.argmin(np.abs(res_tout[:,3]-rtm[3]))
#filename = nom_fol + 'a_sig10_unknown'+str(id_tmf)+'.npz'
#d =  np.load(filename)
##    
##            
#X_tmf = d['X_mpm_est']
#V_tmf = d['V_mpm_est']
#
##
##Ux_tmf = d['Ux_map']
##Uv_tmf = d['Uv_map']
##
#
#V = np.pi/4 * np.ones(shape=(128,128))
#V[64:,:] = 3*np.pi/4
#V[:,64:] = (V[:,64:] + np.pi/2)%np.pi
#
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(V, interpolation='nearest', origin='lower', cmap=plt.cm.inferno,alpha=0.5,vmin=0,vmax=np.pi);#plt.imshow(V_mpm_est, interpolation='nearest', origin='lower', cmap=cm_angl,vmin=0,vmax=np.pi); 
#plot_directions(V, np.ones_like(V),pas=8,taille=2)
#plt.axis('off')
#plt.savefig('./figures/exp_pamiAv.png', format='png',dpi=200)

##
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(X_tmf, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r)#,vmin = -1,vmax=1); 
#plt.axis('off')
#plt.savefig('./figures/exp_pamiAx_otmf.png', format='png',dpi=200)
####
#
###
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(V_tmf, vmin=0, vmax=np.pi,interpolation='nearest', origin='lower',cmap=plt.cm.inferno,alpha=0.5)
#
#V_fi = fi.median_filter(V_tmf, size=8)
#plot_directions(V_fi, np.ones_like(V_tmf),pas=8,taille=2)
#
##ax.imshow(V_tmf, interpolation='nearest', origin='lower', cmap=cmocean.cm.phase,alpha=0.25,vmin=0,vmax=np.pi);#plt.imshow(V_mpm_est, interpolation='nearest', origin='lower', cmap=cm_angl,vmin=0,vmax=np.pi); 
##plot_directions(V_tmf, np.ones_like(V_tmf),pas=8,taille=2.0)
#plt.axis('off')
#plt.savefig('./figures/exp_pamiAv_otmf.png', format='png',dpi=200)
#

###
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(Ux_tmf, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r,vmin =0,vmax=1); 
#plt.axis('off')
#plt.savefig('./figures/exp_pamiA_ux.png', format='png',dpi=200)
####
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(Uv_tmf, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r,vmin = 0,vmax=1); 
#plt.axis('off')
#plt.savefig('./figures/exp_pamiA_uv.png', format='png',dpi=200)
#
