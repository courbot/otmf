# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 13:54:47 2017

@author: courbot
"""

import numpy as np
import matplotlib.pyplot as plt

#%%
def plot_directions(angle, intensite,pas,taille=1,couleur='b'):
    
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
        
                plt.plot((deb_x[i,j],fin_x[i,j]), (deb_y[i,j],fin_y[i,j]) ,couleur, linewidth=1)
    plt.xlim((-0.5,S1-0.5))
    plt.ylim((-0.5,S0-0.5))    
 
def plot_stream(angle,dens,couleur):
    taille = 1.
    intensite = 1.
    S0 = angle.shape[0]
    S1 = angle.shape[1]

    y,x = np.ogrid[0:S0,0:S1]

    angle2 = angle

    deb_x = np.tile(x,(S0,1)) - taille*np.sin(angle2) * intensite
    deb_y = np.tile(y,(1,S1)) - taille*np.cos(angle2) * intensite

    fin_x = np.tile(x,(S0,1)) + taille*np.sin(angle2) * intensite
    fin_y = np.tile(y,(1,S1)) + taille*np.cos(angle2) * intensite



    plt.streamplot(x,y,fin_x-deb_x,fin_y-deb_y,color=couleur,arrowsize=0.0001,density=dens,linewidth=1)

    plt.xlim((-0.5,S1-0.5))
    plt.ylim((-0.5,S0-0.5)) 


for i in range(9):
    
    nom = 'brodatz'+str(i)
    dat=np.load('./results/banc_test/'+str(i)+'_MPM.npz')
    dat_map = np.load('./results/banc_test/'+str(i)+'_MPM.npz')

    Y, X_mpm_est, Ux_map, X_mpm_hmf, V_mpm_est, Uv_map = dat['Y'], dat['X_mpm_tmf'], dat['Ux_tmf'], dat['X_mpm_hmf'], dat['V_mpm_tmf'], dat['Uv_tmf']


    X_map_est,V_map_est, X_map_hmf = dat_map['X_mpm_tmf'], dat_map['V_mpm_tmf'], dat['X_mpm_hmf']

#cm_gris = plt.cm.Greys
#
##%%
    
    fig = plt.figure(figsize=(6,6))
    ax = plt.Axes(fig, [0.,0.,1.,1.])
    fig.add_axes(ax)
    im = Y.mean(axis=2)
    ax.imshow(im, interpolation='nearest', origin='lower', cmap=plt.cm.gray,vmin = im.mean() - 3*im.std(), vmax = im.mean() + 3*im.std())#,vmin = -1,vmax=1); 
    plt.axis('off')
    plt.savefig('./figures/brodatz/'+nom+'_y.png', format='png',dpi=256/6.,cmap='gray')
#
#
##%%
    fig = plt.figure(figsize=(6,6))
    ax = plt.Axes(fig, [0.,0.,1.,1.])
    fig.add_axes(ax)
    Xt = np.copy(X_mpm_est)    
    c_new = np.array([0,0.5,1])
    i=0
    for c in (0,0.5,1):
        c_new[i] = Y[X_mpm_est==c].mean()
        Xt[X_mpm_est==c] = c_new[i]
    
    ax.imshow(Xt, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r); 
    plt.axis('off')
    plt.savefig('./figures/brodatz/'+nom+'_x_otmf_mpm.png', format='png',dpi=256/6.)
    
    fig = plt.figure(figsize=(6,6))
    ax = plt.Axes(fig, [0.,0.,1.,1.])
    fig.add_axes(ax)
    
    Xt = np.copy(X_mpm_hmf)    
    c_new = np.array([0,0.5,1])
    i=0
    for c in (0,0.5,1):
        c_new[i] = Y[X_mpm_hmf==c].mean()
        Xt[X_mpm_hmf==c] = c_new[i]
    ax.imshow(Xt, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r); 
    plt.axis('off')
    plt.savefig('./figures/brodatz/'+nom+'_x_hmf_mpm.png', format='png',dpi=256/6.)
    
    
#
##%%
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(1-X_map_est, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r); 
#plt.axis('off')
#plt.savefig('./figures/'+nom+'_x_otmf_map.png', format='png',dpi=200)
#
##%%
#
    fig = plt.figure(figsize=(6,6))
    ax = plt.Axes(fig, [0.,0.,1.,1.])
    fig.add_axes(ax)
    ax.imshow(Ux_map, interpolation='nearest', origin='lower', cmap=plt.cm.gray,vmin=0,vmax=1); 
    plt.axis('off')
    plt.savefig('./figures/brodatz/'+nom+'_ux_otmf_mpm.png', format='png',dpi=256/6.)
    
#%%
    fig = plt.figure(figsize=(6,6))
    ax = plt.Axes(fig, [0.,0.,1.,1.])
    fig.add_axes(ax)
    ax.imshow(Uv_map, interpolation='nearest', origin='lower', cmap=plt.cm.gray,vmin=0,vmax=1); 
    plt.axis('off')
    plt.savefig('./figures/brodatz/'+nom+'_uv_otmf_mpm.png', format='png',dpi=256/6.)
    #%%

##%%
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#
#ax.imshow(X_map_hmf, interpolation='nearest', origin='lower', cmap=plt.cm.Greys_r); 
#plt.axis('off')
#plt.savefig('./figures/'+nom+'_x_hmf_map.png', format='png',dpi=200)
#
    #%%
#    import cmocean
    fig = plt.figure(figsize=(6,6))
    ax = plt.Axes(fig, [0.,0.,1.,1.])
    fig.add_axes(ax)
    ax.imshow(V_mpm_est, vmin=0, vmax=np.pi,interpolation='nearest', origin='lower',cmap=plt.cm.inferno,alpha=0.5)

    V_fi = fi.median_filter(V_mpm_est, size=4)
#    plot_directions(V_fi, np.ones_like(V_mpm_est),pas=8,taille=2,couleur='firebrick')
    plot_stream(V_fi,dens=0.7*128./30,couleur='firebrick') 
    plt.axis('off')
    plt.savefig('./figures/brodatz/'+nom+'_v_otmf_mpm.png', format='png',dpi=100)
    
    plt.close('all')
    #%%
#
#fig = plt.figure(figsize=(6,6))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(Y.mean(axis=2),interpolation='nearest', origin='lower',cmap=plt.cm.gray,alpha=0.75)
#
#%%
#fig = plt.figure(figsize=(2.5,2.5))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(V_mpm_est, vmin=0, vmax=np.pi,interpolation='nearest', origin='lower',cmap=plt.cm.inferno,alpha=0.5)
#plot_directions(V_mpm_est, np.ones_like(V_mpm_est),pas=4,taille=1,couleur='firebrick')
##ax.imshow(Y.mean(axis=2),interpolation='nearest', origin='lower',cmap=plt.cm.gray,alpha=0.5)
##
##plot_stream(V_mpm_est,dens=0.55*128./30,couleur='firebrick') 
#plt.axis('off')
##plt.savefig('./figures/'+nom+'2e1.png', format='png',dpi=200)
#plt.savefig('./figures/'+nom+'_v_otmf_mpm.png', format='png',dpi=200)
#
##%%
#fig = plt.figure(figsize=(2.5,2.5))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(Y.mean(axis=2),interpolation='nearest', origin='lower',cmap=plt.cm.gray,alpha=0.5)
##
#plot_stream(V_mpm_est,dens=0.7*128./30,couleur='firebrick') 
#plt.axis('off')
##plt.savefig('./figures/'+nom+'2e1.png', format='png',dpi=200)
#plt.savefig('./figures/'+nom+'_v_otmf_mpm2.png', format='png',dpi=200)
#
##%%
#fig = plt.figure(figsize=(2.5,2.5))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(V_map_est, vmin=0, vmax=np.pi,interpolation='nearest', origin='lower',cmap=plt.cm.inferno,alpha=0.5)
#plot_directions(V_mpm_est, np.ones_like(V_mpm_est),pas=4,taille=1,couleur='firebrick')
##ax.imshow(Y.mean(axis=2),interpolation='nearest', origin='lower',cmap=plt.cm.gray,alpha=0.5)
##
##plot_stream(V_mpm_est,dens=0.55*128./30,couleur='firebrick') 
#plt.axis('off')
##plt.savefig('./figures/'+nom+'2e1.png', format='png',dpi=200)
#plt.savefig('./figures/'+nom+'_v_otmf_map.png', format='png',dpi=200)
#
##%%
#fig = plt.figure(figsize=(2.5,2.5))
#ax = plt.Axes(fig, [0.,0.,1.,1.])
#fig.add_axes(ax)
#ax.imshow(Y.mean(axis=2),interpolation='nearest', origin='lower',cmap=plt.cm.gray,alpha=0.5)
##
#plot_stream(V_map_est,dens=0.7*128./30,couleur='firebrick') 
#plt.axis('off')
##plt.savefig('./figures/'+nom+'2e1.png', format='png',dpi=200)
#plt.savefig('./figures/'+nom+'_v_otmf_map2.png', format='png',dpi=200)
##
##