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

import parameters
#import image_tools as it
import gibbs_sampler as gs
import fields_tools as ft
import seg_OTMF as sot
import SEM as sem

from scipy.ndimage import zoom
from scipy.ndimage.filters import gaussian_filter 
from scipy.ndimage.filters import median_filter 



def gen_exp(experiment):

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
        dat = np.load('./data/lyabig.npz')
        Y = dat['Y']
        Y = Y[:,8:,:]
        Y = Y[:S0,:S1,120:170]
        
    elif experiment=='4':
        X,V = gen_xv(S0,S1,v_range)
        
    elif experiment=='5':
        dat = np.load('./data/synth1b.npz')
        
        ratio = np.float(S0)/dat['X'].shape[0]
        X =  zoom(gaussian_filter(dat['X'], sigma=(ratio,ratio)), (ratio,ratio),order=0) 
        X = X.astype(float)
        V =  gs.cast_angles(zoom(dat['V'], (ratio,ratio),order=0),v_range)
#        X[:, S1/2:]*=0.5
        
        
    elif experiment=='6':
        dat = np.load('./data/atten1.npz')
        ratio = np.float(S0)/dat['X'].shape[0]
        X =  5*zoom(gaussian_filter(dat['X'], sigma=(ratio,ratio)), (ratio,ratio),order=0) 
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
def plot_directions(angle, intensite,pas):
    
    S0 = angle.shape[0]
    S1 = angle.shape[1]
    
    y,x = np.ogrid[0:S0,0:S1]
    
    angle2 = angle

    deb_x = np.tile(x,(S0,1)) - 0.5*np.sin(angle2) * intensite
    deb_y = np.tile(y,(1,S1)) - 0.5*np.cos(angle2) * intensite
    
    fin_x = np.tile(x,(S0,1)) + 0.5*np.sin(angle2) * intensite
    fin_y = np.tile(y,(1,S1)) + 0.5*np.cos(angle2) * intensite
    
    
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
    S0 = 128
    S1=  128
    

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


X,V = gen_exp(experiment)
##%%
#
alpha = 5.
alpha_v = 10.
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
                             alpha =alpha,
                             alpha_v = alpha_v,
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
#sig = 0.75
#%% Options
SNR = -10


#
#%%
mu = gen_mu(pargibbs.W)
sig = np.linalg.norm(mu)/np.sqrt(pargibbs.W)* 10**(-SNR/20.)
#sig = 0.25
rho_1 = 0.5 * sig**2
rho_2 = 0.25 * sig**2

print('------- Generation donnees : sig2=%.4f (SNR = %.2f dB), rho1= %.4f, rho2 = %.4f'%(sig**2,SNR,rho_1,rho_2))

if experiment !='3':
    pargibbs, Y = gen_obs(X,pargibbs.W,m,sig,rho_1,rho_2,corrnoise=True)
    
else:
    X = np.zeros(shape=(S0,S1))
    V = np.zeros(shape=(S0,S1))

#pargibbs.Y = Y

###%%
nb_iter_sem=20

nb_rea = 200 # il faudrait 100 ici, pour estimer des frequences proprement
taille_fen = 5
seuil_conv = 0.05
pargibbs.Xi = 0.
##
##
hierarch = False
incert = True

start = time.time()
####
range_level=np.array([0]) # pas de hierarchie pour l'instant
X_mpm_hierarch = np.zeros(shape=(S0,S1,range_level.size))
V_mpm_hierarch = np.zeros(shape=(S0,S1,range_level.size))


# Convergence des estimateurs de Gibbs
pargibbs.autoconv=True
pargibbs.thr_conv = 1e-4
#
for level in range_level:#(range_level[0],range_level[1]):
    

    print 'niveau %.0f...'%level
    if level==range_level[0]: # premiere iteration
        pargibbs.v_help = True
        nb_iter_mpm = 150
        pargibbs.nb_iter = 100 
    else :
        pargibbs.v_help = True
        nb_iter_mpm = 150
        pargibbs.nb_iter = 100 
        
        
    zoom_ratio = 1./(2**level)
    
    if level !=0 :
        Y_fi = np.copy(Y)
        for w in range(W):
            Y_fi[:,:,w] = gaussian_filter(Y[:,:,w], sigma=((2**level),(2**level)))

            
        Y_zoom = zoom(Y_fi, (zoom_ratio,zoom_ratio,1))#,prefilter=False)
    else :
        Y_zoom = np.copy(Y)
#    
#    Y_courant = zoom(Y, (zoom_ratio,zoom_ratio,1))
    Y_courant = np.copy(Y_zoom)
    
    pargibbs.Y = Y_courant
    pargibbs.S0 = Y_courant.shape[0]
    pargibbs.S1 = Y_courant.shape[1]
    
    Vois = np.zeros(shape=(pargibbs.S0,pargibbs.S1,8))
    for i in xrange(pargibbs.S0):
        for j in xrange(pargibbs.S1):
            Vois[i,j,:] = it.get_num_voisins(i,j,np.zeros(shape=(pargibbs.S0,pargibbs.S1)))
    pargibbs.Vois = Vois
    
    # Transfert info de l'etage superieur
    if hierarch==True :
        if level != range_level[0]:
            pargibbs.X_init = zoom(X_mpm_hierarch[:,:,level+1 ], (zoom_ratio,zoom_ratio),order=2) 
            pargibbs.V_init = zoom(V_mpm_hierarch[:,:,level+1 ], (zoom_ratio,zoom_ratio),order=2) 
    
    
    X_mpm_est,V_mpm_est,Ux_map,Uv_map, parsem = sot.seg_otmf(pargibbs,nb_iter_sem,nb_iter_mpm,nb_rea,seuil_conv, taille_fen,longueur_mpm=1,incert=incert,tmf=True)
    
    X_mpm_hierarch[:,:,level] =  zoom(X_mpm_est, (1./zoom_ratio,1./zoom_ratio),order=0)
    V_mpm_hierarch[:,:,level] =  zoom(V_mpm_est, (1./zoom_ratio,1./zoom_ratio),order=0)
    print '-----'
## 
end = time.time() - start
print 'Temps total : %.2f s'%end  
print '------------------------------------'


#%% Un peu d'adaptation a posteriori si il y a plus que 2 classes

if x_range.size>2:
    snr_tous = np.zeros_like(x_range) 
    # snr estime de la region a 1
    snr_tous = 10*np.log10((np.linalg.norm(x_range[:,np.newaxis]*parsem.mu,axis=1)**2) /(W*parsem.sig) )
#    decal =  (x_range[2]-x_range[1])/2.
#    x_range_decal = np.append(x_range -decal,x_range[-1]+decal )
#    snr_tous_decal = 10*np.log10((np.linalg.norm(x_range_decal[:,np.newaxis]*parsem.mu,axis=1)**2) /(W*parsem.sig) )
##    snr_tous = snr_tous[~np.isinf(snr_tous)]
#    indices = (X_mpm_hierarch[:,:,level] *nb_level_x).astype(int)
    
    X_snr_true = 10*np.log10((np.linalg.norm(X[:,:,np.newaxis]*mu,axis=2)**2) /(W*sig) )
    

#%%  Affichage MPM sur V
import matplotlib



#plt.close('all')
nb_li = 3
nb_col =3

plt.rc('text', usetex=True)
plt.rc('font',family='serif')

cm_gris = matplotlib.cm.gray
cm_angl = matplotlib.cm.Spectral

cm_gris.set_bad('r',1.)
cm_angl.set_bad('r',1.)

plt.figure(figsize=(4.5*nb_col,4*nb_li))


plt.subplot(nb_li,nb_col,1)
plt.imshow(Y.mean(axis=2), interpolation='nearest', origin='lower', cmap=cm_gris,vmin = -1,vmax=1); 
plt.axis('off')
plt.title('$\mathbf{y}$, moyenne spectrale')

plt.subplot(nb_li,nb_col,2)
plt.imshow(X, interpolation='nearest', origin='lower', cmap=cm_gris,vmin=0); 
#plt.contour(V, v_range.size,colors='g',linewidths=2,alpha=0.75)
plt.axis('off')
plt.title('$\mathbf{x}$')

plt.subplot(nb_li,nb_col,3)
plt.imshow(V, interpolation='nearest', origin='lower', cmap=cm_angl,vmin=0,vmax=np.pi); 
plot_directions(V, np.ones_like(V),pas=8)
plt.axis('off')
plt.title('$\mathbf{v}$')




    
plt.subplot(nb_li,nb_col,nb_col+2)
if x_range.size>2:
    cm_gris_snr = matplotlib.cm.get_cmap('gray',nb_level_x+1)
    cm_gris_snr.set_bad('k',1.)
    bounds = snr_tous#[~np.isinf(snr_tous)]
    plt.imshow(X_mpm_hierarch[:,:,level], interpolation='nearest', origin='lower', cmap=cm_gris_snr); 
    loc = x_range*(nb_level_x-1)/nb_level_x +0.5/(nb_level_x)
    cb=plt.colorbar(fraction=0.046,pad=0.04)
    cb.set_ticks(loc)
    cb.set_label('SNR')
    cb.set_ticklabels(['{:4.2f}'.format(l) for l in bounds])
else:
    plt.imshow(X_mpm_hierarch[:,:,level], interpolation='nearest', origin='lower', cmap=cm_gris,vmin=0,vmax=1); 
#for vr in range(v_range.size):
#    plt.contour(V_mpm_hierarch[:,:,level]==v_range[vr],1,colors='g',linewidths=2,alpha=0.75)
plt.axis('off')
#    plt.title('$\\hat{\mathbf{x}}_{MPM}^%.0f$'%level)
#erreur_mpm_x = (X != X_mpm_est).mean()
plt.title('$\\hat{\mathbf{x}}_{\mathrm{MPM}}$ - %.2f '%(erreur(X,X_mpm_hierarch[:,:,level])*100))


plt.subplot(nb_li,nb_col,nb_col+3)
plt.imshow(V_mpm_hierarch[:,:,level], interpolation='nearest', origin='lower',cmap=cm_angl,vmin=0,vmax=np.pi); 
plot_directions(V_mpm_hierarch[:,:,level], np.ones_like(V_mpm_hierarch[:,:,level]),pas=8)
plt.axis('off')
plt.title('$\\hat{\mathbf{v}}_{\mathrm{MPM}}$ - %.2f '%(erreur(V,V_mpm_hierarch[:,:,level])*100))




plt.subplot(nb_li,nb_col,2*nb_col+2)
plt.imshow(Ux_map, interpolation='nearest', origin='lower', cmap=plt.cm.hot_r); 
plt.axis('off')
plt.title('$\\hat{\mathbf{u}}^x$')
plt.colorbar(fraction=0.046,pad=0.04)


plt.subplot(nb_li,nb_col,2*nb_col+3)
plt.imshow(Uv_map, interpolation='nearest', origin='lower',cmap=plt.cm.hot_r); 
plt.colorbar(fraction=0.046,pad=0.04)
plt.axis('off')
plt.title('$\\hat{\mathbf{u}}^v$')





plt.tight_layout()
plt.savefig('./figures/seg_uncert2.pdf', format='eps',dpi=100)
#plt.savefig('./figures/atten_seg_uncert8.pdf', format='eps',dpi=100)
#plt.savefig('./figures/exray_seg_uncert4.pdf', format='eps',dpi=100)
#%%
###############################################################################
#
#for level in range_level:
#    
#    zoom_ratio = 1./(2**level)
#    
#    if level !=0 :
#        Y_fi = np.copy(Y)
#        for w in range(W):
#            Y_fi[:,:,w] = gaussian_filter(Y[:,:,w],sigma=((2**level),(2**level)))
#
#            
#        Y_zoom = zoom(Y_fi, (zoom_ratio,zoom_ratio,1))
#    else :
#        Y_zoom = np.copy(Y)
#    
#    if level !=0 :
#        plt.subplot(nb_li,nb_col,nb_col*(nb_li-level-1)+1)
#        plt.imshow(Y_zoom.mean(axis=2), interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin = -1,vmax=1); 
#        plt.axis('off')
#        plt.title('$\mathbf{y}^%.0f$, moyenne spectrale'%level)
#    
#    plt.subplot(nb_li,nb_col,nb_col*(nb_li-level-1)+2)
#    plt.imshow(X_mpm_hierarch[:,:,level], interpolation='nearest', origin='lower', cmap=cm_gris,vmin=0,vmax=1); 
#    plt.contour(V_mpm_hierarch[:,:,level]==v_range[0], 1,colors='g',linewidths=5,alpha=0.75)
#    plt.axis('off')
#    #    plt.title('$\\hat{\mathbf{x}}_{MPM}^%.0f$'%level)
#    #erreur_mpm_x = (X != X_mpm_est).mean()
#    plt.title('$\\hat{\mathbf{x}}_{\mathrm{MPM}}^%.0f$ - %.2f, contours de $\\hat{\mathbf{v}}_{\mathrm{MPM}}^%.0f$ '%(level,erreur(X,X_mpm_hierarch[:,:,level])*100,level))
#    
#    
#    plt.subplot(nb_li,nb_col,nb_col*(nb_li-level-1)+3)
#    plt.imshow(V_mpm_hierarch[:,:,level], interpolation='nearest', origin='lower',cmap=cm_angl,vmin=0,vmax=np.pi); 
#    plot_directions(V_mpm_hierarch[:,:,level], np.ones_like(V_mpm_hierarch[:,:,level]),pas=8)
#    plt.axis('off')
#    plt.title('$\\hat{\mathbf{v}}_{\mathrm{MPM}}^%.0f$ - %.2f '%(level,erreur(V,V_mpm_hierarch[:,:,level])*100))
#
#
#
#
#
#plt.tight_layout()




#%%




#
##plt.close('all')
#nb_li = 3
#nb_col = 1+range_level.size
#
#plt.rc('text', usetex=True)
#plt.rc('font',family='serif')
#
#cm_gris = matplotlib.cm.bone
#cm_angl = matplotlib.cm.Spectral
#
#cm_gris.set_bad('r',1.)
#cm_angl.set_bad('r',1.)
#
#plt.figure(figsize=(5*nb_col,5*nb_li))
#
#
#plt.subplot(nb_li,nb_col,1)
#plt.imshow(Y.mean(axis=2), interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin = -1,vmax=1); 
#plt.axis('off')
#plt.title('$\mathbf{y}^0$ spectral average')
#
#plt.subplot(nb_li,nb_col,nb_col+1)
#plt.imshow(X, interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin=0,vmax=1); 
#plt.axis('off')
#plt.title('$\mathbf{x}$')
#
#plt.subplot(nb_li,nb_col,2*nb_col+1)
#plt.imshow(V, interpolation='nearest', origin='lower', cmap=plt.cm.Spectral,vmin=0,vmax=np.pi); 
#plot_directions(V, np.ones_like(V),pas=8)
#plt.axis('off')
#plt.title('$\mathbf{v}$')
#
#
################################################################################
#
#for level in range_level:
#    
#    zoom_ratio = 1./(2**level)
#    
#    if level !=0 :
#        Y_fi = np.copy(Y)
#        for w in range(W):
#            Y_fi[:,:,w] = gaussian_filter(Y[:,:,w],sigma=((2**level),(2**level)))
#
#            
#        Y_zoom = zoom(Y_fi, (zoom_ratio,zoom_ratio,1))
#    else :
#        Y_zoom = np.copy(Y)
#    
#    
#    plt.subplot(nb_li,nb_col,(nb_col-level))
#    plt.imshow(Y_zoom.mean(axis=2), interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin = -1,vmax=1); 
#    plt.axis('off')
#    plt.title('$\mathbf{y}^%.0f$ spectral average'%level)
#    
#    plt.subplot(nb_li,nb_col,nb_col + (nb_col-level))
#    plt.imshow(X_mpm_hierarch[:,:,level], interpolation='nearest', origin='lower', cmap=cm_gris,vmin=0,vmax=1); 
#    plt.axis('off')
#    #    plt.title('$\\hat{\mathbf{x}}_{MPM}^%.0f$'%level)
#    #erreur_mpm_x = (X != X_mpm_est).mean()
#    plt.title('$\\hat{\mathbf{x}}_{\mathrm{MPM}}^%.0f$ - %.2f '%(level,erreur(X,X_mpm_hierarch[:,:,level])*100))
#    
#    
#    plt.subplot(nb_li,nb_col,2*nb_col+ (nb_col-level))
#    plt.imshow(V_mpm_hierarch[:,:,level], interpolation='nearest', origin='lower',cmap=cm_angl,vmin=0,vmax=np.pi); 
#    plot_directions(V_mpm_hierarch[:,:,level], np.ones_like(V_mpm_hierarch[:,:,level]),pas=8)
#    plt.axis('off')
#    plt.title('$\\hat{\mathbf{v}}_{\mathrm{MPM}}^%.0f$ - %.2f '%(level,erreur(V,V_mpm_hierarch[:,:,level])*100))
#
#
#
#
#
#plt.tight_layout()








#erreur_mpm_v = (V != V_mpm_est).mean()
#plt.title('$\\hat{\mathbf{v}}_{\mathrm{MPM}}$ - %.2f '%(erreur_mpm_v*100))
#raise_window('Resultats')

#%%

#
#
#
#
#from scipy import interpolate
#
#par = pargibbs
#S0 = par.S0
#S1 = par.S1 
#v_range = par.v_range
#
#nb_nn = par.nb_nn_v_help# number of nearest neighbor
##nb_nn = 19
#dx,dy = np.gradient(X)
#an = np.arctan2(dy,dx)
#an = (an%np.pi)#+np.pi/2)%np.pi
##
#ux, uy = np.ogrid[0:S0,0:S1]
#ux = np.tile(ux,(1,S1))
#uy = np.tile(uy,(S0,1))
##
#
#
#image = np.copy(an)
#image[image==0]+=np.nan
##
##image[mask] = np.nan
#
#valid_mask = ~np.isnan(image)
#coords = np.array(np.nonzero(valid_mask)).T
#values = image[valid_mask]
#
#it = interpolate.LinearNDInterpolator(coords, values, fill_value=0)
#
#an_interp = it(list(np.ndindex(image.shape))).reshape(image.shape)
#
#an_interp = (an_interp+np.pi/2)%np.pi
#an_interp2 = np.copy(an_interp)#(an_interp)%np.pi      
#  
#v_range_new = v_range[v_range!=0]
#for i in range(an.shape[0]):
#    for j in range(an.shape[1]):
#        if np.isnan(an_interp[i,j])==0:
#            ecart = np.abs(an_interp2[i,j] - v_range)
#            ind_min = np.argmin(ecart)
#    
#            an_interp2[i,j] =v_range[ind_min]
#
#
#
#plt.figure()
#plt.subplot(141)
#plt.imshow(X, interpolation='nearest', origin='lower', cmap=plt.cm.bone)
#plt.subplot(142)
#plt.imshow(an, interpolation='nearest', origin='lower', cmap=plt.cm.jet)
#plt.subplot(143)
#plt.imshow(an_interp, interpolation='nearest', origin='lower', cmap=plt.cm.jet)
#plt.subplot(144)
##plt.imshow(an_interp2, interpolation='nearest', origin='lower', cmap=plt.cm.jet)
#plt.imshow(an_interp2, interpolation='nearest', origin='lower',cmap=plt.cm.Spectral,vmin=0,vmax=np.pi);
#
#
#plot_directions(an_interp2, np.ones_like(an_interp2),pas=8) #%%





#%%














##%%
#pargibbs.S0 = 80
#
#pargibbs.S1 = 80
#par = pargibbs
#
#
#
#S0 = par.S0
#S1 = par.S1 
#v_range = par.v_range
#
#nb_nn = par.nb_nn_v_help# number of nearest neighbor
#nb_nn = 19
#dx,dy = np.gradient(X)
#an = np.arctan2(dy,dx)
##
#ux, uy = np.ogrid[0:S0,0:S1]
#ux = np.tile(ux,(1,S1))
#uy = np.tile(uy,(S0,1))
##
#
#an_flat = an.flatten()
#an_interp = np.copy(an)
##
#            #        if (np.isnan(an[i,j])):
#for i in range(an.shape[0]):
#    for j in range(an.shape[1]):
#        dist = np.sqrt((i-ux)**2 + (j-uy)**2)
#        dist[an==0] = 10000
#        
#        dist_flat = dist.flatten()
#
#
#        #            dist_flat[(an_flat)==0] = 10000
#        ind_min = np.argmin(dist_flat)
#        ind_sort = np.argsort(dist_flat)
#        an_interp[i,j] = 0
#        for nn in range(nb_nn):
#                an_interp[i,j] += an_flat[ind_sort[nn]]/nb_nn
##        an_interp[i,j] = an_flat[ind_sort[0]]#/nb_nn
#
#
#an_interp2 = (an_interp+np.pi/2)%np.pi      
#  
#v_range_new = v_range[v_range!=0]
#for i in range(an.shape[0]):
#    for j in range(an.shape[1]):
#        if np.isnan(an_interp[i,j])==0:
#            ecart = np.abs(an_interp2[i,j] - v_range)
#            ind_min = np.argmin(ecart)
#    
#            an_interp2[i,j] =v_range[ind_min]
##an_interp[np.isnan(an_interp)] = 0      
#
#plt.figure()
#plt.subplot(141)
#plt.imshow(X, interpolation='nearest', origin='lower', cmap=plt.cm.bone)
#plt.subplot(142)
#plt.imshow(an, interpolation='nearest', origin='lower', cmap=plt.cm.jet)
#plt.subplot(143)
#plt.imshow(an_interp, interpolation='nearest', origin='lower', cmap=plt.cm.jet)
#plt.subplot(144)
##plt.imshow(an_interp2, interpolation='nearest', origin='lower', cmap=plt.cm.jet)
#plt.imshow(an_interp2, interpolation='nearest', origin='lower',cmap=plt.cm.Spectral,vmin=0,vmax=np.pi);
#plot_directions(an_interp2, np.ones_like(an_interp2),pas=8) #%%
###############################################################################
#
#plt.subplot(nb_li,nb_col,3)
#plt.imshow(Y_1.mean(axis=2), interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin = -1,vmax=1); 
#plt.axis('off')
#plt.title('$\mathbf{y}^1$ spectral average')
#
#plt.subplot(nb_li,nb_col,nb_col+3)
#plt.imshow(X_mpm_1, interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin=0,vmax=1); 
#plt.axis('off')
#plt.title('$\\hat{\mathbf{x}}_{\mathrm{MPM}}^1$ - %.2f '%(erreur(X,X_mpm_1_rescale)*100))
#
##erreur_mpm_x = (X != X_mpm_est).mean()
##plt.title('$\\hat{\mathbf{x}}_{\mathrm{MPM}}$ - %.2f '%(erreur_mpm_x*100))
#
#
#plt.subplot(nb_li,nb_col,2*nb_col+3)
#plt.imshow(V_mpm_1, interpolation='nearest', origin='lower',cmap=plt.cm.Spectral,vmin=0,vmax=np.pi); 
#plot_directions(V_mpm_1, np.ones_like(V_mpm_1),pas=8)
#plt.axis('off')
#plt.title('$\\hat{\mathbf{v}}_{\mathrm{MPM}}^1$ - %.2f '%(erreur(V,V_mpm_1_rescale)*100))
#
#
################################################################################
#
#plt.subplot(nb_li,nb_col,4)
#plt.imshow(Y.mean(axis=2), interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin = -1,vmax=1); 
#plt.axis('off')
#plt.title('$\mathbf{y}^0$ spectral average')
#
#
#plt.subplot(nb_li,nb_col,nb_col+4)
#plt.imshow(X_mpm_0, interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin=0,vmax=1); 
#plt.axis('off')
#plt.title('$\\hat{\mathbf{x}}_{MPM}$')
#plt.title('$\\hat{\mathbf{x}}_{\mathrm{MPM}}^0$ - %.2f '%(erreur(X,X_mpm_0)*100))
#
##erreur_mpm_x = (X != X_mpm_est).mean()
##plt.title('$\\hat{\mathbf{x}}_{\mathrm{MPM}}$ - %.2f '%(erreur_mpm_x*100))
#
#
#plt.subplot(nb_li,nb_col,2*nb_col+4)
#plt.imshow(V_mpm_0, interpolation='nearest', origin='lower',cmap=plt.cm.Spectral,vmin=0,vmax=np.pi); 
#plot_directions(V_mpm_0, np.ones_like(V_mpm_0),pas=8)
#plt.title('$\\hat{\mathbf{v}}_{\mathrm{MPM}}^0$ - %.2f '%(erreur(V,V_mpm_0)*100))
#plt.axis('off')
#


#%%
##--------------------------
#
#plt.subplot(nb_li,nb_col,nb_col+1)
#plt.imshow(X, interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin=0,vmax=1); 
#plt.axis('off')
#plt.title('$\mathbf{x}$')
#
#
#
#plt.subplot(nb_li,nb_col,2*nb_col+1)
#plt.imshow(V, interpolation='nearest', origin='lower', cmap=plt.cm.Spectral,vmin=0,vmax=np.pi); 
##X_aff = np.copy(X)
##X_aff[X_aff==0]+=np.nan
##plt.imshow(X_aff.T, interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin=0,vmax=1,alpha=0.25); 
#plot_directions(V, np.ones_like(V),pas=8)
#plt.axis('off')
#plt.title('$\mathbf{v}$')
#
##--------------------------
#
#plt.subplot(nb_li,nb_col,nb_col+2)
#plt.imshow(X_mpm_est, interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin=0,vmax=1); 
#plt.axis('off')
#plt.title('$\\hat{\mathbf{x}}_{MPM}$')
#erreur_mpm_x = (X != X_mpm_est).mean()
#plt.title('$\\hat{\mathbf{x}}_{\mathrm{MPM}}$ - %.2f '%(erreur_mpm_x*100))
#
#
#
#plt.subplot(nb_li,nb_col,2*nb_col+2)
#plt.imshow(V_mpm_est, interpolation='nearest', origin='lower',cmap=plt.cm.Spectral,vmin=0,vmax=np.pi); 
##X_aff = np.copy(X_mpm_est)
##X_aff[X_aff==0]+=np.nan
##plt.imshow(X_aff.T, interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin=0,vmax=1,alpha=0.25); 
#plot_directions(V_mpm_est, np.ones_like(V),pas=8)
#plt.axis('off')
#erreur_mpm_v = (V != V_mpm_est).mean()
#plt.title('$\\hat{\mathbf{v}}_{\mathrm{MPM}}$ - %.2f '%(erreur_mpm_v*100))
##
#
##--------------------------
#
#
##plt.subplot(nb_li,nb_col,4)
##plt.imshow((X_mpm==1).mean(axis=2), interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin=0,vmax=1); 
##plt.axis('off')
##plt.title('$\\hat{\mathbf{x}}_{MPM}$')
##erreur_mpm_x = (X != X_mpm_est).mean()
##plt.title('$\\hat{\mathbf{x}}_{\mathrm{MPM}}$ - %.2f '%(erreur_mpm_x*100))
#
#from scipy import stats
#
#
#V_mpm_1 = (V_mpm==v_range[1]).mean(axis=2)
#V_mpm_0 = (V_mpm==v_range[0]).mean(axis=2)
#
#std_1 = np.std(V_mpm_1[(V_mpm_est==v_range[1])])
#mean_1 = np.mean(V_mpm_1[(V_mpm_est==v_range[1])])
##n1 = np.size(V_mpm_1[(V_mpm_est==v_range[1])])
#
#
#seuil_1 = stats.norm.isf(0.999,mean_1,std_1) # ca-a-d une p-valeur de 0.01
## C'est le seuil pour lequel on peut rejeter l'appartenance a la classe 1 pour n'importe quelle autre valeur.
#
#
#std_0 = np.std(V_mpm_0[(V_mpm_est==v_range[0])])
#mean_0 = np.mean(V_mpm_0[(V_mpm_est==v_range[0])])
#seuil_0 = stats.norm.isf(0.999,mean_0,std_0) # ca-a-d une p-valeur de 0.05
## C'est le seuil pour lequel on peut rejeter l'appartenance a la classe 1 pour n'importe quelle autre valeur.
#
#
#
#plt.subplot(nb_li,nb_col,3)
#plt.imshow(V_mpm_0, interpolation='nearest', origin='lower',cmap=plt.cm.Oranges,vmin=0,vmax=1,alpha=0.75); 
#cs2 = plt.contour(V_mpm_0, levels=(0.8,0.7,0.6,0.5))
#plt.clabel(cs2,fmt='%1.1f')
#plt.title('Carte, niveaux de $\\hat{p}^{\\mathrm{MPM}}_0$')
#plt.axis('off')
#
#
#
#plt.subplot(nb_li,nb_col,nb_col+3)
#plt.imshow(V_mpm_1, interpolation='nearest', origin='lower',cmap=plt.cm.Greens,vmin=0,vmax=1,alpha=0.75); 
#cs = plt.contour(V_mpm_1, levels=(0.8,0.7,0.6,0.5))
#plt.clabel(cs,fmt='%1.1f')
#plt.title('Carte, niveaux de $\\hat{p}^{\\mathrm{MPM}}_1$')
#plt.axis('off')
#
#plt.subplot(nb_li,nb_col,2*nb_col+3)
#plt.imshow((V_mpm==v_range[2]).mean(axis=2), interpolation='nearest', origin='lower',cmap=plt.cm.Reds,vmin=0,vmax=1,alpha=0.75); 
#cs = plt.contour((V_mpm==v_range[2]), levels=(0.8,0.7,0.6,0.5))
#plt.clabel(cs,fmt='%1.1f')
#plt.title('Carte, niveaux de $\\hat{p}^{\\mathrm{MPM}}_1$')
#plt.axis('off')
#
##
#
###-------------------------------------------------##
#plt.subplot(nb_li, nb_col,4)
#n,bins,patches = plt.hist(V_mpm_0[(V_mpm_est==v_range[0])].flatten(),25,normed=1,alpha=0.25,facecolor='orange')
#plt.plot(bins,mlab.normpdf(bins,mean_0,std_0),'orange',linewidth=2)
#plt.title('Histogramme de $\\hat{p}^{\\mathrm{MPM}}_0$ pour $\\hat{v}^{\\mathrm{MPM}} = v_0$')
#plt.axvline(x=(1-seuil_1), color='g', linestyle='--',linewidth=2 )
#plt.text((1-seuil_1 ), 4, ' rejet $\\mathcal{H}_0^{(1)}$',color='g')
#
#
#plt.subplot(nb_li, nb_col,nb_col+4)
#n,bins,patches = plt.hist(V_mpm_1[(V_mpm_est==v_range[1])].flatten(),25,normed=1,alpha=0.25,facecolor='green')
#plt.plot(bins,mlab.normpdf(bins,mean_1,std_1),'green',linewidth=2)
#plt.title('Histogramme de $\\hat{p}^{\\mathrm{MPM}}_1$ pour $\\hat{v}^{\\mathrm{MPM}} = v_1$')
#plt.axvline(x=(1-seuil_0), color='r', linestyle='--',linewidth=2 )
#plt.text((1-seuil_0),4, 'rejet $\\mathcal{H}_0^{(0)}$',color='r')
#
#
#plt.subplot(nb_li, nb_col,2*nb_col+4)
#n,bins,patches = plt.hist(V_mpm_1.flatten(),50,alpha=0.25,normed=1, facecolor='gray')
#
#y0 = mlab.normpdf(bins,mean_1,std_1)/2.
#y1 = mlab.normpdf(bins,1-mean_0,std_0)/2.
#y = y0 + y1
#
#plt.plot(bins,y0,'green',linewidth=2,alpha=0.5)
#plt.plot(bins,y1,'orange',linewidth=2,alpha=0.5)
#plt.plot(bins,y,'gray',linewidth=2)
#plt.title('Histogramme de $\\hat{p}^{\\mathrm{MPM}}_1$')
#plt.axvline(x=(1-seuil_0), color='r', linestyle='--',linewidth=2 )
#plt.axvline(x=(seuil_1), color='g', linestyle='--',linewidth=2 )
#
#
###-------------------------------------------------##
#
#V_mpm_conf = np.zeros_like(V)
#V_mpm_conf[V_mpm_1 > (1-seuil_0)] = v_range[1]
#V_mpm_conf[V_mpm_0 >(1-seuil_1)] = v_range[0]
#
#
#plt.subplot(nb_li,nb_col,2*nb_col+3)
#plt.imshow(V_mpm_conf, interpolation='nearest', origin='lower',cmap=plt.cm.Spectral,vmin=0,vmax=np.pi); 
#
#erreur_conf = (V_mpm_conf[V_mpm_conf!=0] != V[V_mpm_conf!=0]).mean()
#plot_directions(V_mpm_conf, np.ones_like(V),pas=8)
#plt.axis('off')
#plt.title('$\\hat{\mathbf{v}}_{\mathrm{MPM}}$ + conf - %.2f '%(erreur_conf*100))
##



#plt.savefig('./figures/instance_seg_uncert.eps', format='eps',dpi=100)

#%%
#from scipy.stats import norm
#def fixed_point_func(mu,sig,a,b):
#    
#    alpha = (a-mu)/sig; # Attention en realite il faudrait utiliser les vraies valeurs
#    beta = (b-mu)/sig;
#    
#    F_0 = (norm.pdf(beta)-norm.pdf(alpha))/(norm.cdf(beta)-norm.cdf(alpha))
#    F_1 = 1 - ((beta*norm.pdf(beta)-alpha*norm.pdf(alpha))/(norm.cdf(beta)-norm.cdf(alpha))) - F_0**2
#    
#    sig_est = sig/F_1
#    
#    mu_est = mu + sig_est*F_0
#    
#    return mu_est,sig_est
#
#    
#    
##%%
#nb_iter = 9
#a = 0.5
#b = 1
#mu = np.zeros(nb_iter)
#sig = np.zeros(nb_iter)
#
#
#sig[0] = np.std(V_mpm_1[(V_mpm_est==v_range[1])])
#mu[0] = np.mean(V_mpm_1[(V_mpm_est==v_range[1])])
#
#for i in range(1,nb_iter):
#    
#    mu[i], sig[i] = fixed_point_func(mu[i-1], sig[i-1], a,b)
#
##%%
##sig_1 = np.copy(sig_est)
#sig_1 = np.std(V_mpm_1[(V_mpm_est==v_range[1])])
#mu_1 = np.mean(V_mpm_1[(V_mpm_est==v_range[1])])
#med_1 = np.median(V_mpm_1[(V_mpm_est==v_range[1])])
#
#alpha = (a-med_1)/sig_1; # Attention en realite il faudrait utiliser les vraies valeurs
#beta = (b-med_1)/sig_1;
#
#F_0 = (norm.pdf(beta)-norm.pdf(alpha))/(norm.cdf(beta)-norm.cdf(alpha))
#F_1 = 1 - ((beta*norm.pdf(beta)-alpha*norm.pdf(alpha))/(norm.cdf(beta)-norm.cdf(alpha))) - F_0**2
#
#sig_est = sig_1/F_1
#
#mu_est = mu_1 + sig_est*F_0

#%%
#std_1 = np.std(V_mpm_1[(V_mpm_est==v_range[1])])
#lim_1 = 0.5+2*std_1
#V_mpm_1_conf = V_mpm_1 > lim_1
#
#std_0 = np.std(V_mpm_2[(V_mpm_est==v_range[0])])
#lim_0 = 0.5+2*std_0
#V_mpm_0_conf = V_mpm_2 > lim_0
#
#
#nb_li = 2
#nb_col = 4
##plt.close('all')
#plt.figure(figsize=(4*nb_col,4*nb_li))
#plt.subplot(nb_li, nb_col,1)
#plt.hist(V_mpm_1.flatten(),25,normed=True)
#plt.title('Histogramme de $\\hat{p}^{\\mathrm{MPM}}_1$')
#
#plt.subplot(nb_li, nb_col,2)
#plt.hist(V_mpm_1[(V_mpm_est==v_range[1])].flatten(),25,normed = True)
#plt.title('Histogramme de $\\hat{p}^{\\mathrm{MPM}}_1$ pour $\\hat{v}^{\\mathrm{MPM}} = v_1$')
#plt.axvline(x=lim_1, color='r', linestyle='--',linewidth=2 )
#
#plt.subplot(nb_li, nb_col,3)
#plt.hist(V_mpm_2[(V_mpm_est==v_range[0])].flatten(),25,normed = True)
#plt.title('Histogramme de $\\hat{p}^{\\mathrm{MPM}}_2$ pour $\\hat{v}^{\\mathrm{MPM}} = v_2$')
#plt.axvline(x=lim_0, color='r', linestyle='--',linewidth=2 )
##
#plt.subplot(nb_li, nb_col,nb_col+1)
#plt.hist(V_mpm_2.flatten(),25,normed=True)
#plt.title('Classe 2 : Toutes les valeurs')
#
#plt.subplot(nb_li, nb_col,nb_col+2)
#plt.hist(V_mpm_2[(V_mpm_est==v_range[1])].flatten(),25,normed = True)
#plt.title('val 1')
#
#plt.subplot(nb_li, nb_col,nb_col+3)
#plt.hist(V_mpm_2[(V_mpm_est==v_range[0])].flatten(),25,normed = True)
#plt.title('val 2')


#
#plt.subplot(nb_li, nb_col,nb_col+4)
#plt.imshow(V_mpm_1_conf, interpolation='nearest', origin='lower',cmap=plt.cm.gray,vmin=0,vmax=1); 
#plt.tight_layout()
#plt.show()


#%% Affichage seg + est param


#plt.close('all')
#
#nb_li = 3
#nb_col = 5
#
#plt.rc('text', usetex=True)
#plt.rc('font',family='serif')
#
#plt.figure(figsize=(20,15))
#
#
#
#plt.subplot(nb_li,nb_col,1)
#plt.imshow(Y.mean(axis=2).T, interpolation='nearest', origin='lower', cmap=plt.cm.bone,); 
#plt.axis('off')
#plt.title('$\mathbf{y}$ spectral average')
#
#plt.subplot(nb_li,nb_col,2)
#plt.imshow(Y[:,:,5].T, interpolation='nearest', origin='lower', cmap=plt.cm.bone,); 
#plt.axis('off')
#plt.title('$\mathbf{y}$ at $\\lambda=5$')
#
##--------------------------
#
#plt.subplot(nb_li,nb_col,nb_col+1)
#plt.imshow(X.T, interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin=0,vmax=1); 
#plt.axis('off')
#plt.title('$\mathbf{x}$')
#
#
#
#plt.subplot(nb_li,nb_col,2*nb_col+1)
#plt.imshow(V.T, interpolation='nearest', origin='lower', cmap=plt.cm.Spectral,vmin=0,vmax=np.pi); 
##X_aff = np.copy(X)
##X_aff[X_aff==0]+=np.nan
##plt.imshow(X_aff.T, interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin=0,vmax=1,alpha=0.25); 
#plot_directions(V.T, np.ones_like(V.T),pas=5)
#plt.axis('off')
#plt.title('$\mathbf{v}$')
#
##--------------------------
#
#plt.subplot(nb_li,nb_col,nb_col+2)
#plt.imshow(X_mpm_est.T, interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin=0,vmax=1); 
#plt.axis('off')
#plt.title('$\\hat{\mathbf{x}}_{MPM}$')
#erreur_mpm_x = (X != X_mpm_est).mean()
#plt.title('$\\hat{\mathbf{x}}_{\mathrm{MPM}}$ - %.2f '%(erreur_mpm_x*100))
#
#
#
#plt.subplot(nb_li,nb_col,2*nb_col+2)
#plt.imshow(V_mpm_est.T, interpolation='nearest', origin='lower',cmap=plt.cm.Spectral,vmin=0,vmax=np.pi); 
##X_aff = np.copy(X_mpm_est)
##X_aff[X_aff==0]+=np.nan
##plt.imshow(X_aff.T, interpolation='nearest', origin='lower', cmap=plt.cm.bone,vmin=0,vmax=1,alpha=0.25); 
#plot_directions(V_mpm_est.T, np.ones_like(V.T),pas=5)
#plt.axis('off')
#erreur_mpm_v = (V != V_mpm_est).mean()
#plt.title('$\\hat{\mathbf{v}}_{\mathrm{MPM}}$ - %.2f '%(erreur_mpm_v*100))
##
#
#sig_sem= parsem.sig_sem
#rho_1sem= parsem.rho1_sem
#rho_2sem= parsem.rho2_sem
#mu_sem = parsem.mu_sem
#pi_sem = parsem.pi_sem
#
#residus = mu_sem - mu[np.newaxis,:]
#eqm = (residus**2).sum(axis=1)
#rmse = np.sqrt(eqm)
#
#rmse_final = np.sqrt( ((parsem.mu-mu)**2 ).sum())
#
#plt.subplot(nb_li,nb_col,3)
#plt.plot(rmse,'b', label = 'RMSE')
#plt.plot(np.ones_like(sig_sem)*rmse_final,'-g',linewidth=1.5, label='RMSE de $\\hat{\\mu}^{\mathrm{SEM}}$ ')
#plt.ylim((0,None))
#plt.xlabel('$k$ (iter. SEM)')
#plt.title('RMSE sur $\\hat{\\mu}^k$')
#plt.legend()
#plt.ylim((0,0.1))
#plt.grid()
#
#plt.subplot(nb_li,nb_col,4)
#plt.plot(sig_sem,'b')
#plt.plot(sig*np.ones_like(sig_sem),'--k', label = 'True value')
#plt.plot(np.ones_like(sig_sem)*parsem.sig,'-g',linewidth=1.5, label='$\\hat{\\sigma}^{\mathrm{SEM}}$ ')
#plt.xlabel('$k$ (iter. SEM)')
#plt.title('$\\hat{\\sigma}^k$')
#plt.legend()
#plt.grid()
#plt.ylim((0.99*sig,1.01*sig))
#
#
#plt.subplot(nb_li,nb_col,5)
#plt.plot(rho_1sem,'b')
#plt.plot(rho_1*np.ones(np.size(sig_sem)),'--b', label='True value')
#plt.plot(parsem.rho_1*np.ones(np.size(sig_sem)),'-g',linewidth=1.5, label='$\\hat{\\rho}_1^{\mathrm{SEM}}$')
#plt.legend()
#plt.title('$\\hat{\\rho}_1^{k}$')
#plt.ylim((0.95*rho_1,1.05*rho_1))
#plt.grid()
#
#plt.subplot(nb_li,nb_col,nb_col+5)
#plt.plot(rho_2sem,'b')
#plt.plot(rho_2*np.ones(np.size(sig_sem)),'--k', label='True value')
#plt.plot(parsem.rho_2*np.ones(np.size(sig_sem)),'-g',linewidth=1.5, label='$\\hat{\\rho}_2^{\mathrm{SEM}}$')
#plt.legend()
#plt.title('$\\hat{\\rho}_2^{k}$')
#plt.ylim((0.95*rho_2,1.05*rho_2))
#plt.grid()
#
#plt.subplot(nb_li,nb_col,nb_col+3)
#plt.semilogy(parsem.pi[0,:],'b',label='$\\pi^x$')
#plt.plot(pi[0,:],'--b')
#
#plt.plot(parsem.pi[1,:],'r',label='$\\pi^v$')
#plt.plot(pi[1,:],'--r')
#plt.xlabel('Config')
#plt.legend(loc='bottom right')
#plt.grid()
#
#
#
#plt.subplot(nb_li,nb_col,2*nb_col+3)
#plt.plot(pi_sem[:,0,8],'b',label='$\\pi_8^x$')
#plt.plot(pi[0,8]*np.ones(shape=pi_sem.shape[0]),'--b')
#
#plt.plot(pi_sem[:,1,8],'r',label='$\\pi_8^v$')
#plt.plot(pi[1,8]*np.ones(shape=pi_sem.shape[0]),'--r')
#plt.xlabel('$k$ (iter. SEM)')
#plt.legend(loc='bottom right')
#plt.grid()
#
#
#plt.subplot(nb_li,nb_col,2*nb_col+4)
#plt.plot(pi_sem[:,0,7],'b',label='$\\pi_7^x$')
#plt.plot(pi[0,7]*np.ones(shape=pi_sem.shape[0]),'--b')
#
#plt.plot(pi_sem[:,1,7],'r',label='$\\pi_7^v$')
#plt.plot(pi[1,7]*np.ones(shape=pi_sem.shape[0]),'--r')
#plt.xlabel('$k$ (iter. SEM)')
#plt.legend(loc='bottom right')
#plt.grid()
#
#
#
#plt.subplot(nb_li,nb_col,2*nb_col+5)
#plt.plot(pi_sem[:,0,6],'b',label='$\\pi_6^x$')
#plt.plot(pi[0,6]*np.ones(shape=pi_sem.shape[0]),'--b')
#
#plt.plot(pi_sem[:,1,6],'r',label='$\\pi_6^v$')
#plt.plot(pi[1,6]*np.ones(shape=pi_sem.shape[0]),'--r')
#plt.xlabel('$k$ (iter. SEM)')
#plt.legend(loc='bottom right')
#plt.grid()
#
#
#plt.tight_layout()
#
##plt.savefig('./figures/instance_paramest.eps', format='eps',dpi=100)
##%%

#from scipy import interpolate


##    
#    nb_nn = par.nb_nn_v_help# number of nearest neighbor
#
#    dx,dy = np.gradient(X)
#    an = np.arctan2(dy,dx)
    

##nb_nn = 19
#dx,dy = np.gradient(X.astype('float'))
#an = np.arctan2(dy,dx)+np.pi/2

#an = (an)%np.pi
    
##    an = np.arctan(dy/dx)
#an = np.zeros_like(X)
#
##    q13 = (dx*dy) > 0
#q13 = (dx>0)*(dy>0)s
#q24 = (dx*dy) <0    
#
#an[q13] = np.arctan(dy[q13].astype(float)/dx[q13])+np.pi/2


#    an[q13] +=np.pi/2

#    an[q24] +=np.pi/2
#an=  np.arctan(dy/dx) + np.pi/2
#    an = np.angle(dx + 1j*dy)
#    an = (an%(2*np.pi))#+np.pi/2)%np.pi
#
#an = np.zeros(shape=(S0,S1))
#q13 = (dx>0)*(dy>0)
#
#
#an[q13] = np.arctan2(dy[q13],dx[q13])
#

#
#dx,dy = np.gradient(X.astype('float'))
#an = np.arctan2(dx,dy)
#
#
#an = (an%np.pi+np.pi/2)%np.pi
#mask = (an==np.pi/2)+(an==0)
##an[an==0] = np.pi
##
#ux, uy = np.ogrid[0:S0,0:S1]
#ux = np.tile(ux,(1,S1))
#uy = np.tile(uy,(S0,1))
##
#
#
#image = np.copy(an)
#image[mask]+=np.nan
##image = (image%np.pi+np.pi/2)%np.pi
##
##image[mask] = np.nan
#
#
########### Actual interpolation
#valid_mask = ~np.isnan(image)
#coords = np.array(np.nonzero(valid_mask)).T
#values = image[valid_mask]
#
#it = interpolate.LinearNDInterpolator(coords, values, fill_value=0)
#an_interp = it(list(np.ndindex(image.shape))).reshape(image.shape)
#
#
#
#
#
############## Recasting into the known range
##an_interp = (an_interp+np.pi/2)%np.pi
#an_interp2 = np.copy(an_interp)#(an_interp)%np.pi      
#
#  
#v_range_new = v_range[v_range!=0]
#for i in range(an.shape[0]):
#    for j in range(an.shape[1]):
#        if np.isnan(an_interp[i,j])==0:
#            ecart = np.abs(an_interp2[i,j] - v_range_new)%np.pi
#            ind_min = np.argmin(ecart)
#    
#            an_interp2[i,j] =v_range_new[ind_min]    
#
#
#dx,dy = np.gradient(X.astype('float'))
#an = np.arctan2(dx,dy)
#
#
#an = (an%np.pi+np.pi/2)%np.pi
#mask = (an==np.pi/2)+(an==0)
##an[an==0] = np.pi
##
#ux, uy = np.ogrid[0:S0,0:S1]
#ux = np.tile(ux,(1,S1))
#uy = np.tile(uy,(S0,1))
##
#
#
#image = np.copy(an)
#image[mask]+=np.nan
##image = (image%np.pi+np.pi/2)%np.pi
##
##image[mask] = np.nan
#
#
########### Actual interpolation
#valid_mask = ~np.isnan(image)
#coords = np.array(np.nonzero(valid_mask)).T
#values = image[valid_mask]
#
#it = interpolate.LinearNDInterpolator(coords, values, fill_value=0)
#
#
#
#an_interp = it(list(np.ndindex(image.shape))).reshape(image.shape)
#
#ani0 = an_interp[an_interp==0]
#an_interp[an_interp==0] = np.random.choice(v_range,size=ani0.size)
#
#
############## Recasting into the known range
##an_interp = (an_interp+np.pi/2)%np.pi
#an_interp2 = np.copy(an_interp)#(an_interp)%np.pi      
#
#  
#v_range_new = v_range[v_range!=0]
#for i in range(an.shape[0]):
#    for j in range(an.shape[1]):
#        if np.isnan(an_interp[i,j])==0:
#            ecart = np.abs(an_interp2[i,j] - v_range_new)
#            ind_min = np.argmin(ecart)
#    
#            an_interp2[i,j] =v_range_new[ind_min]    
#
#plt.close('all')
#Vbis = np.copy(an_interp)
#
#%%
#from scipy import interpolate
###
#v_range= np.array([np.pi/2,np.pi])
#X_fil = gaussian_filter(X.astype(float), sigma=(2,2))
#dx,dy = np.gradient(X_fil.astype('float'))
#an = np.arctan2(dy,dx)
#    
#
#an = (an+np.pi/2)%np.pi
##an = an%np.pi
##an = gs.cast_angles(an,v_range)
##mask = (an==np.pi/2)+(an==0)
#mask = (((dx==0)*(dy==0))) > 0
##an[an==0] = np.pi
##
#ux, uy = np.ogrid[0:S0,0:S1]
#ux = np.tile(ux,(1,S1))
#uy = np.tile(uy,(S0,1))
##
#
#
#image = np.copy(an)
#image[mask]+=np.nan
#
########### Actual interpolation
#valid_mask = ~np.isnan(image)
#coords = np.array(np.nonzero(valid_mask)).T
#values = image[valid_mask]
#
#it = interpolate.LinearNDInterpolator(coords, values)
#
#an_interp = it(list(np.ndindex(image.shape))).reshape(image.shape)
#
##an_interp = an_interp%np.pi
#
## Interpolation fails outside of a convex hull. Misisng values are randomly filled.
#ani0 =an_interp[np.isnan(an_interp)]
#an_interp[np.isnan(an_interp)] = np.random.choice(v_range,size=ani0.size)
#
############### Recasting into the known range
#an_interp_fil = gaussian_filter(an_interp.astype(float), sigma=(1,1))
#
#
#an_interp2 = gs.cast_angles(an_interp_fil, v_range)
#
#plt.figure(figsize=(15,5))
#plt.subplot(131)
#plt.imshow(image, interpolation='nearest', origin='lower',cmap=cm_angl,vmin=0,vmax=np.pi); 
#plot_directions(an, np.ones_like(V_mpm_hierarch[:,:,level]),pas=8)
#plt.axis('off')
#
#plt.subplot(132)
#plt.imshow(an_interp_fil, interpolation='nearest', origin='lower',cmap=cm_angl,vmin=0,vmax=np.pi,alpha=0.75); 
#plot_directions(an_interp, np.ones_like(V_mpm_hierarch[:,:,level]),pas=8)
#plt.axis('off')
#
#
#plt.subplot(133)
#plt.imshow(an_interp2, interpolation='nearest', origin='lower',cmap=cm_angl,vmin=0,vmax=np.pi); 
#plot_directions(an_interp2, np.ones_like(V_mpm_hierarch[:,:,level]),pas=8)
#plt.axis('off')
#
#plt.tight_layout()
#%%
#Vc = dat['V']
#v_range = np.array([np.pi/3,2*np.pi/3,np.pi])
#v_range = np.array([np.pi/4,np.pi/2,3*np.pi/4,np.pi])
#decal = (v_range[1]-v_range[0])/2
#
## il nous faut construire des intervalles sur 0,pi
#
#V_new = np.zeros_like(Vc)
#for i in range(v_range.size):
#    
#
#    vmin = (v_range[i] - decal)
#    vmax = (v_range[i]+decal)
#    Vcb = Vc%np.pi
#    
#    print vmin, vmax
#    if vmin < 0 :
#        # intervalle supplementaire du cote de pi
#        vmin_bis = vmin+np.pi
#        vmax_bis = np.pi
#        
#        V_new[(Vcb>=vmin)*(Vcb<vmax)] = v_range[i]  
#        V_new[(Vcb>=vmin_bis)*(Vcb<vmax_bis)] = v_range[i]  
#        print vmin_bis, vmax_bis
#    elif vmax > np.pi:
#        vmin_bis = 0
#        vmax_bis = vmax-np.pi
#        
#        V_new[(Vcb>=vmin)*(Vcb<vmax)] = v_range[i]  
#        V_new[(Vcb>=vmin_bis)*(Vcb<vmax_bis)] = v_range[i]  
#        print vmin_bis, vmax_bis
#        
#    else:
#        V_new[(Vcb>=vmin)*(Vcb<vmax)] = v_range[i]  
#        
#
#
#plt.imshow(V_new, interpolation='nearest', origin='lower',cmap=cm_angl,vmin=0,vmax=np.pi); 
#plot_directions(V_new, np.ones_like(V_new),pas=50)
#

