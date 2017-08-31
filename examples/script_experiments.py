# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 09:45:48 2015

@author: courbot
"""



import numpy as np 
import sys 
import matplotlib.pyplot as plt
import time
import scipy.stats as st
import scipy.cluster.vq as cvq
import multiprocessing as mp
import image_tools as it
from scipy.ndimage import imread
import matplotlib.mlab as mlab
import numpy.ma as ma
import parameters
#import image_tools as it
import gibbs_sampler as gs
import fields_tools as ft
import seg_OTMF as sot
import SEM as sem

from scipy.ndimage import zoom
from scipy.ndimage.filters import gaussian_filter 
from scipy.ndimage.filters import median_filter 



def gen_exp(experiment,S0,S1):

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
    
        dat = np.load('./data/tmf_exp1_128.npz')
        X=dat['X']
        V=dat['V']
        
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
        dat = np.load('./data/tmf_exp2_128(b).npz')
        X=dat['X']
        V=dat['V']
        
        
    elif experiment=='3':
        dat = np.load('./results/sources/udf10/cubes/208.npz')
        Y = dat['Y_ms']
        S0,S1,W = Y.shape
        X = np.zeros(shape=(S0,S1))
        V = np.zeros_like(X)
#        Y = Y[:,8:,:]
#        Y = Y[:S0,:S1,120:170]
        
    elif experiment=='4':
        # cas "japan"
        X,V = gen_xv(S0,S1,v_range)
        
    elif experiment=='5':
        dat = np.load('./data/synth1b.npz') #"jap"
        dat = np.load('./data/jap.npz') #"jap"
        
        ratio = np.float(S0)/dat['X'].shape[0]
        X =  zoom(gaussian_filter(dat['X'], sigma=(ratio,ratio)), (ratio,ratio),order=0) 
        X = X.astype(float)
        V =  gs.cast_angles(zoom(dat['V'], (ratio,ratio),order=0),v_range)
#        X[:, S1/2:]*=0.5
        
        
    elif experiment=='6':
        dat = np.load('./data/atten1.npz')
        ratio = np.float(S0)/dat['X'].shape[0]
        X =  5.*zoom(gaussian_filter(dat['X'], sigma=(ratio,ratio)), (ratio,ratio),order=0) 
        
#        X_diff = np.abs(X[:,:,np.newaxis] - 5*x_range[np.newaxis,np.newaxis,:])
#        
#        ind_x = np.argmin(X_diff,axis=2)
#        X_cast = x_range[ind_x]*5
##        for x_inst in x_range:
#        X = np.copy(X_cast)    
        
        
#        X[:, S1/2:]*=0.5
    #    X = X/10.
        V =  gs.cast_angles(zoom(dat['V'], (ratio,ratio),order=0),v_range)
        
    return X,V



def raise_window(figname=None):
    """
    Raise the plot window for Figure figname to the foreground.  If no argument
    is given, raise the current figure.

    This function will only work with a Qt graphics backend.  It assumes you
    have already executed the command 'import matplotlib.pyplot as plt'.
    """

    if figname: plt.figure(figname)
    cfm = plt.get_current_fig_manager()
    cfm.window.activateWindow()
    cfm.window.raise_()

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
    
    mu = np.zeros(shape=W)
    mu[4] = 0.3 ; mu[5] = 1 ; mu[6] = 0.5     
    
    
    return mu

def gen_obs(X,W, m,sig,rho_1,rho_2,corrnoise=False):
    

        
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
    
    Y = X[:,:,np.newaxis]*mu[np.newaxis,np.newaxis,:] +np.random.multivariate_normal(mean=np.zeros_like(mu),cov=Sigma,size=(S0,S1))
    
    #Y = X[:,:,np.newaxis]*mu[np.newaxis,np.newaxis,:] + st.norm.rvs(loc=m,scale=sig,size=(S0,S1,W))
    
    pargibbs.Y = Y
    
    return pargibbs,Y
    
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

experiment = '5'


# parametres utiles:
if experiment=='3':
    S0 = 90
    S1 = 52
    
else:
    S0 = 80
    S1=  80
    

    ###################
#%%
v_range = np.array([np.pi/4.,3*np.pi/4.])#-np.pi/6

# definition de la subdivision : finir a pi, ne pas mettre 0 d'abord !
#v_range = np.array([np.pi/3,2*np.pi/3, np.pi])
#v_range = np.array([np.pi/4,np.pi/2, 3*np.pi/4,np.pi])
#v_range = np.array([np.pi/2,np.pi])
#v_range = np.array([np.pi/4,np.pi,3*np.pi/4])
###################
nb_level_x = 1
x_range = np.arange(0, 1+1./nb_level_x, 1./nb_level_x)


X,V = gen_exp(experiment,S0,S1)

if experiment=='3':
    dat = np.load('./results/sources/udf10/cubes/208.npz')
    Y = dat['Y_ms']
    S0,S1,W = Y.shape
    X = np.zeros(shape=(S0,S1))
    V = np.zeros_like(X)
##%%
#

print('---------------------------------------')
pargibbs = parameters.ParamsGibbs(S0 = S0,
                             S1 = S1,
                             type_pot = 'potts',
                             phi_uni = 0.,
                             thr_conv=0.005,
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


#%% Essai simulation
#
pi = np.zeros(shape=(2,9))


pi[0,:] = np.array([1e-04,   1e-04,   1e-03, 1e-2,   5e-02,   1e-01, 5e-02,   1e-01,   9.9e-01])
pi[1,:] = np.array([1e-04,   1e-04,   1e-03, 1e-2,   5e-02,   1e-01, 5e-02,   1e-01,   9.9e-01])



pi[1,:] /= pi[1,:].sum()
pi[0,:] /= pi[0,:].sum()


#pi = np.copy(pi_est)

pargibbs.pi = pi

#%%
#------- Generate V
#print('------- Generate V')


#------- Generate Y | X
if experiment=='3':
    W = Y.shape[2]
else:
    W = 20
pargibbs.W = W
m = 0

if experiment=='3':

    pargibbs.S0 = S0
    pargibbs.S1 = S1



#==============================================================================
# Generation de l'observation Y
#==============================================================================
SNR = -7


mu = gen_mu(pargibbs.W)
sig = np.linalg.norm(mu)/np.sqrt(pargibbs.W)* 10**(-SNR/20.)

rho_1 = 0.5 * sig**2
rho_2 = 0.25 * sig**2

print('------- Generation donnees : sig2=%.4f (SNR = %.2f dB), rho1= %.4f, rho2 = %.4f'%(sig**2,SNR,rho_1,rho_2))

if experiment !='3':
    pargibbs, Y = gen_obs(X,pargibbs.W,m,sig,rho_1,rho_2,corrnoise=True)
    
else:
    X = np.zeros(shape=(S0,S1))
    V = np.zeros(shape=(S0,S1))

pargibbs.Y = Y



#==============================================================================
# Paramètres à fixer
#==============================================================================

nb_iter_sem=40 # nb d'iter maximum pour SEM
nb_rea = 200 # nombre de realisations de Gibbs differentes pour le MPM
taille_fen = 5 # fenetre pour la convergence de SEM
seuil_conv = 0.05 # convergence de SEM

nb_iter_mpm = 200 # longueur max. pour les Gibbs dans le MPM
pargibbs.nb_iter = 100 
pargibbs.autoconv=True # convergence automatique des estimateurs de Gibbs
pargibbs.thr_conv = 5*1./(S0*S1) # seuil pour cette convergence, en relatif
incert = True # Utilisation ou non de segmentation avec incertitude
pargibbs.Xi = 0.  # valeur de l'"incertitude" adoptee
tmf = True

#==============================================================================
# Segmentation
#==============================================================================
print '------------------------------------'
start = time.time()

Y_courant = np.copy(Y)

pargibbs.Y = Y_courant


X_mpm_est,V_mpm_est,Ux_map,Uv_map, parsem = sot.seg_otmf(pargibbs,nb_iter_sem,nb_iter_mpm,nb_rea,seuil_conv, taille_fen,longueur_mpm=1,incert=incert,tmf=tmf)

## 
end = time.time() - start
print 'Temps total : %.2f s'%end  
print '------------------------------------'


#==============================================================================
# Un peu d'adaptation a posteriori si il y a plus que 2 classes
#==============================================================================

if x_range.size>2:
    snr_tous = np.zeros_like(x_range) 
    # snr estime de la region a 1
    snr_tous = 10*np.log10((np.linalg.norm(x_range[:,np.newaxis]*parsem.mu,axis=1)**2) /(W*parsem.sig) )

    X_snr_true = 10*np.log10((np.linalg.norm(X[:,:,np.newaxis]*mu,axis=2)**2) /(W*sig) )
    
    # calcul de la RMSE
    facteur_multiplicatif = np.median(X[X_mpm_est==1]).mean()
    ecart_moyen = np.abs( (X/5. - X_mpm_est)).mean()
#    rmse = np.sqrt(mse)
    
#%%
#==============================================================================
#   Affichage verite + segmentations
#==============================================================================
import matplotlib

if x_range.size>2:
    snr_tous = np.zeros_like(x_range) 
    # snr estime de la region a 1
    snr_tous = 10*np.log10((np.linalg.norm(x_range[:,np.newaxis]*parsem.mu,axis=1)**2) /(W*parsem.sig) )   

#plt.close('all')
if incert == True:
    nb_li = 3
else : 
    nb_li = 2
    
nb_col =3
#
#plt.rc('text', usetex=True)
#plt.rc('font',family='serif')



cm_gris = matplotlib.cm.gray
cm_angl = matplotlib.cm.Spectral

cm_gris.set_bad('r',1.)
#X_mpm_est[X_mpm_est==0]+=np.nan
cm_angl.set_bad('r',1.)
##cm_gris = matplotlib.cm.gray
##cm_angl = matplotlib.cm.Spectral
#
#cm_gris.set_bad('r',1.)
#cm_angl.set_bad('r',1.)

plt.figure(figsize=(4.5*nb_col,4*nb_li))


plt.subplot(nb_li,nb_col,1)
plt.imshow(Y.mean(axis=2), interpolation='nearest', origin='lower', cmap=cm_gris,vmin = -1,vmax=1); 
#plt.axis('off')
plt.title('$\mathbf{y}$, moyenne spectrale')

plt.subplot(nb_li,nb_col,2)
plt.imshow(X, interpolation='nearest', origin='lower', cmap=cm_gris,vmin=0); 
#plt.contour(V, v_range.size,colors='g',linewidths=2,alpha=0.75)
#plt.axis('off')
plt.title('$\mathbf{x}$')

plt.subplot(nb_li,nb_col,3)
plt.imshow(V, interpolation='nearest', origin='lower', cmap=cm_angl,vmin=0,vmax=np.pi); 
plot_directions(V, np.ones_like(V),pas=8)
#plt.axis('off')
plt.title('$\mathbf{v}$')




    
plt.subplot(nb_li,nb_col,nb_col+2)
if x_range.size>2:
#    cm_gris_snr = matplotlib.cm.get_cmap('gray',nb_level_x+1)
#    cm_gris_snr.set_bad('w',1.)
#    bounds = snr_tous[~np.isinf(snr_tous)]
    plt.imshow(X_mpm_est, interpolation='nearest', origin='lower', cmap=cm_gris,vmin=0,vmax=1); 
#            loc = x_range*(nb_level_x-1)/(nb_level_x) #+0.5/(nb_level_x)
#    loc = np.linspace(0.5/nb_level_x,1-0.5/nb_level_x,nb_level_x+1)
##            loc = 0.5/nb_level_x + np.arange(nb_level_x)/float(nb_level_x)
#    cb=plt.colorbar(fraction=0.046,pad=0.04,aspect='auto',shrink=float(S1)/S0)
#    cb.set_ticks(loc)
#    cb.set_label('SNR')
#    cb.set_ticklabels(['{:4.2f}'.format(l) for l in bounds])
else:
    plt.imshow(X_mpm_est, interpolation='nearest', origin='lower', cmap=cm_gris,vmin=0,vmax=1); 
#for vr in range(v_range.size):
#    plt.contour(V_mpm_hierarch[:,:,level]==v_range[vr],1,colors='g',linewidths=2,alpha=0.75)
#plt.axis('off')
#    plt.title('$\\hat{\mathbf{x}}_{MPM}^%.0f$'%level)
#erreur_mpm_x = (X != X_mpm_est).mean()
plt.title('$\\hat{\mathbf{x}}_{\mathrm{MPM}}$ - %.2f '%(erreur(X,X_mpm_est)*100))


plt.subplot(nb_li,nb_col,nb_col+3)
plt.imshow(V_mpm_est, interpolation='nearest', origin='lower',cmap=cm_angl,vmin=0,vmax=np.pi); 
plot_directions(V_mpm_est, np.ones_like(V_mpm_est),pas=8)
#plt.axis('off')
plt.title('$\\hat{\mathbf{v}}_{\mathrm{MPM}}$ - %.2f '%(erreur(V,V_mpm_est)*100))


if incert==True:
    
    
    plt.subplot(nb_li,nb_col,2*nb_col+2)
    plt.imshow(Ux_map, interpolation='nearest', origin='lower', cmap=plt.cm.gray,vmin=0,vmax=1); 
    #plt.axis('off')
    plt.title('$\\hat{\mathbf{u}}^x$')
    plt.colorbar(fraction=0.046,pad=0.04)
    
    
    plt.subplot(nb_li,nb_col,2*nb_col+3)
    plt.imshow(Uv_map, interpolation='nearest', origin='lower',cmap=plt.cm.gray,vmin=0, vmax=1); 
    plt.colorbar(fraction=0.046,pad=0.04)
    #plt.axis('off')
    plt.title('$\\hat{\mathbf{u}}^v$')





plt.tight_layout()


#plt.savefig('./figures/galtout.png', format='png',dpi=200)


#%%
#
#plt.figure(figsize=(5,5))
#plt.imshow(Y.mean(axis=2), interpolation='nearest', origin='lower', cmap=cm_gris,vmin = -1,vmax=1);
##plt.axis('off')
#plt.tight_layout()
#plt.savefig('./figures/galnoise.png', format='png',dpi=200)
#
#plt.figure(figsize=(5,2.5))
#plt.plot(mu, linewidth=2,color='#8234cb')
#plt.xlabel('$\\lambda$')
#plt.xlim((0,19))
#plt.ylim((-1,1))
#plt.grid()
#plt.tight_layout()
#plt.savefig('./figures/galmu.png', format='png',dpi=200)
#
#
#plt.figure(figsize=(5,2.5))
#plt.plot(Y[20,20,:], linewidth=2,color='gray')
#plt.xlabel('$\\lambda$')
#plt.xlim((0,19))
#plt.ylim((-1,1))
#plt.grid()
#plt.tight_layout()
#plt.savefig('./figures/galmunoise.png', format='png',dpi=200)
#
#
#
#plt.figure(figsize=(5,5))
#
#plt.imshow(V, interpolation='nearest', origin='lower', cmap=cm_angl,vmin=0,vmax=np.pi); 
#plot_directions(V, np.ones_like(V),pas=8)
##plt.axis('off')
#plt.tight_layout()
#plt.savefig('./figures/galangle.png', format='png',dpi=200)
#
#
#plt.figure(figsize=(5,5))
#plt.imshow(X, interpolation='nearest', origin='lower', cmap=cm_gris,vmin=0); 
##plt.axis('off')
#plt.tight_layout()
#plt.savefig('./figures/gal.png', format='png',dpi=200)
#
#
#
#plt.figure(figsize=(5,5))
#
#plt.imshow(V_mpm_est, interpolation='nearest', origin='lower', cmap=cm_angl,vmin=0,vmax=np.pi); 
#plot_directions(V_mpm_est, np.ones_like(V_mpm_est),pas=8)
##plt.axis('off')
#plt.tight_layout()
#plt.savefig('./figures/galangleest.png', format='png',dpi=200)
#
#
#plt.figure(figsize=(5,5))
#plt.imshow(X_mpm_est, interpolation='nearest', origin='lower', cmap=cm_gris,vmin=0); 
##plt.axis('off')
#plt.tight_layout()
#plt.savefig('./figures/galest.png', format='png',dpi=200)
#%%
#plt.figure(figsize=(5,5))
#plt.imshow(X_mpm_est, interpolation='nearest', origin='lower', cmap=cm_gris,vmin=0); 
##plt.axis('off')
#plt.tight_layout()
#plt.savefig('./figures/galcache.png', format='png',dpi=200)
##

#plt.savefig('./figures/seg_uncert2.pdf', format='eps',dpi=100)
#plt.savefig('./figures/atten_seg_uncert8.pdf', format='eps',dpi=100)
#plt.savefig('./figures/exray_seg_uncert4.pdf', format='eps',dpi=100)


#%%
#plt.figure(figsize=(5,3))
#cmcop = plt.cm.get_cmap('gray',x_range.size)
##ax1.set_color_cycle(plt.cm.get_cmap('copper',x_range.size+2))
##mu_sem_tous = parsem.mu[:,np.newaxis] * x_range[np.newaxis,:]
#i=0
#for x_inst in x_range[x_range!=0]:
#    plt.plot(parsem.mu*x_inst,color=cmcop(i-1),linewidth=2)
#    i+=1
##ax1.errorbar(1,parsem.mu[1],yerr=parsem.sig,color='r',linewidth=2)#np.array([parsem.mu[0]-parsem.sig,parsem.mu[0]+parsem.sig]))
#plt.xlim([0,W-1])
#plt.ylim([-0.5,3.5])
#plt.grid()
##if x_range.size!=2:
##    plt.legend(np.round(bounds*100)/100.)
#plt.xlabel('$\\lambda$')
#plt.ylabel('I')
#plt.tight_layout()
#plt.savefig('./figures/galmuref.png', format='png',dpi=200)
#
##%%
#import numpy.ma as ma
#plt.figure(figsize=(5,3))
#cmcop = plt.cm.get_cmap('gray',x_range.size)
##ax1.set_color_cycle(plt.cm.get_cmap('copper',x_range.size+2))
##mu_sem_tous = parsem.mu[:,np.newaxis] * x_range[np.newaxis,:]
#i=0
#for x_inst in x_range[x_range!=0]:
#    msk = (X_mpm_est==x_inst)    
#    Y_msk = ma.masked_array(Y, np.tile(1-msk[:,:,np.newaxis],(1,1,W)))
#    sp_mean = ma.mean(ma.mean(Y_msk,axis=0),axis=0)
#    plt.plot(sp_mean,color=cmcop(i-1),linewidth=2)
#    i+=1
##ax1.errorbar(1,parsem.mu[1],yerr=parsem.sig,color='r',linewidth=2)#np.array([parsem.mu[0]-parsem.sig,parsem.mu[0]+parsem.sig]))
#plt.xlim([0,W-1])
#plt.ylim([-0.5,3.5])
#plt.grid()
##if x_range.size!=2:
##    plt.legend(np.round(bounds*100)/100.)
#plt.xlabel('$\\lambda$')
#plt.ylabel('I')
#plt.tight_layout()
#plt.savefig('./figures/galmuest.png', format='png',dpi=200)