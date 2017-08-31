
# coding: utf-8

# In[1]:

import numpy as np 
import sys 
import matplotlib.pyplot as plt
import time
import scipy.stats as st
import scipy.cluster.vq as cvq
import multiprocessing as mp
import image_tools as it
from scipy.ndimage import imread


import numpy.ma as ma
import scipy.stats as st
import scipy.signal as si
from astropy.io import fits
from astropy.table import Table
import scipy.ndimage.morphology as morph
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
from scipy.ndimage import zoom

import parameters
#import image_tools as it
import gibbs_sampler as gs
import fields_tools as ft
import seg_OTMF as sot


# In[2]:

#sys.path.insert(0,'/home/courbot/Programmes/mpdaf-1.2')
#import mpdaf
#
#from mpdaf.obj import WCS
#from mpdaf.obj import WaveCoord
#from mpdaf.obj import Image
#from mpdaf.obj import Spectrum
#from mpdaf.obj import Cube
##from mpdaf.obj import CubeDisk
#
#from mpdaf.sdetect import Source, SourceList


# In[3]:

# Version information
version = 0.2
# source dir
nom_dossier='./results/sources/hdfs/'


# In[4]:

def plot_directions(angle, intensite,pas):
    
    S0 = angle.shape[0]
    S1 = angle.shape[1]
    
    y,x = np.ogrid[0:S0,0:S1]
    

    deb_x = np.tile(x,(S0,1)) - 0.5*np.sin(angle) * intensite
    deb_y = np.tile(y,(1,S1)) - 0.5*np.cos(angle) * intensite
    
    fin_x = np.tile(x,(S0,1)) + 0.5*np.sin(angle) * intensite
    fin_y = np.tile(y,(1,S1)) + 0.5*np.cos(angle) * intensite
    
    
    for i in range(int(pas/2.),S0,pas):
        for j in range(int(pas/2.),S1,pas):
            if angle[i,j] != 0:
                plt.plot((deb_x[i,j],fin_x[i,j]), (deb_y[i,j],fin_y[i,j]) ,'k')
    plt.xlim((-0.5,S1-0.5))
    plt.ylim((-0.5,S0-0.5))    


# In[10]:


#src_list = SourceList.from_path(nom_dossier)


# In[11]:

# pour avoir les numeros des objets ( methode depuis l'objet SourceList ?)
#nb_obj = 0
#for s in src_list:
#    nb_obj+=1
#num_src = np.zeros(shape=(nb_obj))
#i=0
#for s in src_list:
#    if s.id==554:
#        continue
#        
#    num_src[i] = s.id
#    i+=1
#
#num_src_sort = np.copy(num_src)
#num_src_sort.sort()
#print num_src
#print num_src_sort


# In[12]:

lambda_0 = 4750    # see Bacon et al, 2015
lambda_lya = 1216
pas_spectral = 1.25 # Angstrom / spectral band

W = 20
W_aff = 300
S = 50


# In[ ]:

def calc_S(src,marge):
    if src.id in (489,):
        S = 100 + 2*marge
    elif src.id in (139,):
        S = 80 + 2*marge

    elif src.id in (144,92):
        S = 50 + 2*marge

    elif src.id in (43,40,139,257,294,308,311,334,363,430,449,469,474,489,500,514,546,552,568,577,580,581):
        S = 60+2*marge

    else:
        S = 40 + 2*marge
    return S

def median_filtering(Y_src):
    ss_cube_medfilt = si.medfilt(Y_src,(1,1,301))
    Y_ms = Y_src - ss_cube_medfilt
    
    return Y_ms

def get_subcube_large(cube,W,S,lam_ang,pas_spectral,lambda_0,src):
    
    
    deb_lar = lam_ang - W_aff/2 * pas_spectral
    pos_ligne = W_aff/2 # en bande
    if deb_lar < lambda_0: 
        deb_lar = lambda_0 +1
        pos_ligne = lam_ang*pas_spectral
        
    fin_lar = deb_lar+W_aff*pas_spectral
    
    sp_range_large = np.arange(deb_lar,fin_lar+1,pas_spectral)
    sub_cube_large = cube.subcube((src.dec,src.ra),S,lbda=(deb_lar,fin_lar),pix=True)
    
    Y_src = np.swapaxes(sub_cube_large.get_np_data(),2,0)
    
    return Y_src,sub_cube_large


def get_subcube_tight(sub_cube_large,W,lam_ang,pas_spectral,lambda_0,src) :  
    
    deb = lam_ang - W/2 * pas_spectral
    pos_ligne = W/2 # en bande
    if deb < lambda_0: 
        deb = lambda_0 +1
        pos_ligne = lam_ang*pas_spectral
        
    fin = deb+(W-1)*pas_spectral

    sub_cube_tight = sub_cube_large.subcube((src.dec,src.ra),S,lbda=(deb,fin),pix=True)
    Y_ms = np.swapaxes(sub_cube_tight.get_np_data(),0,2)
    return Y_ms

def make_square(Y_src,Y_ms):
    # Ensuring Y_src cube is "square" (after median filtering + subcube extraction - preserving wcs)
    diff_size = Y_src.shape[0]-Y_src.shape[1]
    if diff_size!=0:
        taille = max( Y_src.shape[0],Y_src.shape[1])
        
        Y_src_new = np.zeros((taille,taille,Y_src.shape[2])) + np.nan
        Y_ms_new = np.zeros((taille,taille,Y_ms.shape[2])) + np.nan
        
        Y_src_new[:Y_src.shape[0],:Y_src.shape[1],:] = Y_src
        Y_ms_new[:Y_src.shape[0],:Y_src.shape[1],:] = Y_ms
        
        Y_src = Y_src_new
        Y_ms = Y_ms_new
        
    return Y_src,Y_ms

def cut_nan(Y_ms):
    nanmap=  np.isnan(Y_ms).any(axis=2)
    nanmap = nanmap+nanmap.T

    nonanmap = nanmap==0

    # remove rows + cols containing only nans
    xmax = (nanmap).all(axis=1).argmax()
    xmin = nanmap.all(axis=1).argmin()

    ymax = (nanmap).all(axis=0).argmax()
    ymin = nanmap.all(axis=0).argmin()

    Y_ms2 = Y_ms[xmin:xmax,ymin:ymax,:]

    # keep rows + cols containing only non-nans
    nanmap=  np.isnan(Y_ms2).any(axis=2)
    nonanmap = nanmap==0

    xmax = np.where((nonanmap).all(axis=1))[0].max()
    xmin = np.where((nonanmap).all(axis=1))[0].min()

    ymax = np.where((nonanmap).all(axis=0))[0].max()
    ymin = np.where((nonanmap).all(axis=0))[0].min()

    Y_ms2 = Y_ms2[xmin:xmax,ymin:ymax,:]
    
    return Y_ms2


# In[ ]:

############## A faire
#  Sauvegarder systématiquement les résultats, les logs, les images/figures dans une seule structure
#  Faire un script / des codes pour l'appli astro. Dedans :
#       - nettoyage de la "segmentation initiale"
#       - donner des intervalles de confiance sur le résultat final ?
#       - associer les niveaux de segmentation à un RSB (mu/(lambda*sig) )       
#       - l'incertitude comme une surface 3D ?
#
#
######################

#
##doc_pdf = PdfPages('./results/LAE-HMF-HDFS-1.24(sample)2.pdf')
#i=1
#num_obj = 1
nom_cube='udf10'
#nb. UDF10 : 0 n'existe pas
liste_cube_udf = (11724,149,153,180,184,204,214,252,364,400,547,559,579,675,797)
liste_cube_hdfs = (0,1,4,5,7,9)
for num_obj in (208,):#liste_cube_udf:#1,4,5,7,9):#range(10,20):#range(12,20):#range(10): #range(10):#(3,):#(0,1,4,5,7,9):#range(num_src.size):,
    
        if nom_cube=='hdfs':
            src = src_list[np.where(num_src==num_src_sort[num_obj])[0]]   
            
    
            if src.id==554 or src.id==144 or src.id==89 or src.id==159 or src.id==181:
                continue
            print '#----#'
            print '| '+ str(src.id) +' |'
            print '#----#'

            
            dat = np.load('./results/sources/'+nom_cube+'/cubes/%.0f.npz'%src.id)
        elif nom_cube=='udf10':
            if num_obj==184:
                continue
            
            print '#----#'
            print '| '+ str(num_obj) +' |'
            print '#----#'
            
            dat = np.load('./results/sources/'+nom_cube+'/cubes/%.0f.npz'%num_obj)
            
        Y_ms = dat['Y_ms']

#        Y_ms = zoom(Y_ms, (2,2,1)) 
        


        
# quelques correctifs objectwise
        if nom_cube=='hdfs':
            if src.id==95:  
                Y_ms = Y_ms[:,:,:16]
               
                
            if src.id==139:
                Y_ms = Y_ms[:-10, :-10,:]
                
        elif nom_cube=='udf10':
            if num_obj==547:
                Y_ms = Y_ms[:, :-10,:]
                Y_ms = Y_ms[:,:,:15]
               
            elif num_obj==208:
                Y_ms = Y_ms[30:,30:70,:]
                
        if np.isnan(Y_ms).any():
            continue
        
        Y = np.copy(Y_ms)
        
        alpha = 2.5
        alpha_v = 5.
        S0, S1, W = Y_ms.shape
        
#        v_range = np.array([np.pi/4,np.pi/2,3*np.pi/4,np.pi])
        v_range = np.array([np.pi/4,3*np.pi/4])
        nb_level_x = 5
        x_range = np.arange(0, 1+1./nb_level_x, 1./nb_level_x)


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


        pargibbs.Y = Y_ms
        pargibbs.W  = W
        
        #==============================================================================
        # Paramètres à fixer
        #==============================================================================
        
        nb_iter_sem=40 # nb d'iter maximum pour SEM
        nb_rea = 200 # nombre de realisations de Gibbs differentes pour le MPM
        taille_fen = 5 # fenetre pour la convergence de SEM
        seuil_conv = 0.05 # convergence de SEM
        
        nb_iter_mpm = 200 # longueur max. pour les Gibbs dans le MPM
        pargibbs.nb_iter = 200 
        pargibbs.autoconv=True # convergence automatique des estimateurs de Gibbs
        pargibbs.thr_conv = 3*1./(S0*S1) # seuil pour cette convergence, en relatif
        incert = True # Utilisation ou non de segmentation avec incertitude
        pargibbs.Xi = 0.  # valeur de l'"incertitude" adoptee
        tmf = True
        #==============================================================================
        # Segmentation
        #==============================================================================
        
        print '------------------------------------'
        start = time.time()
        X_mpm_est,V_mpm_est,Ux_map,Uv_map, parsem = sot.seg_otmf(pargibbs,nb_iter_sem,nb_iter_mpm,nb_rea,seuil_conv, taille_fen,longueur_mpm=1,incert=incert,tmf=tmf,spec_snr=True)

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
        #==============================================================================
        #     Affichage    
        #==============================================================================
        #plt.close('all')
#%%
        if tmf==True:
            nb_li = 3
        else: 
            nb_li = 2
        nb_col =3
        
#        plt.rc('text', usetex=True)
#        plt.rc('font',family='serif')
        
        cm_gris = matplotlib.cm.coolwarm
        cm_angl = matplotlib.cm.Spectral
        
        cm_gris.set_bad('w',1.)
        X_mpm_est[X_mpm_est==0]+=np.nan
        cm_angl.set_bad('r',1.)
        
        plt.figure(figsize=(4*nb_col,3.5*nb_li))
        
        
        plt.subplot(nb_li,nb_col,1)
        plt.imshow(Y.mean(axis=2).T, interpolation='nearest', origin='lower', cmap=cm_gris,vmin = -1,vmax=1); 
        #plt.axis('off')
        plt.title('$\mathbf{y}$, moyenne spectrale')
        
        #
        #plt.subplot(nb_li,nb_col,nb_col+1)
        #plt.imshow(Y_ms.mean(axis=2).T, interpolation='nearest', origin='lower', cmap=cm_gris,vmin = -1,vmax=1); 
        ##plt.axis('off')
        #plt.title('$\mathbf{y}$, version employee')
        
            
        plt.subplot(nb_li,nb_col,2)
        if x_range.size>2:
            cm_gris_snr = matplotlib.cm.get_cmap('coolwarm',nb_level_x)
            cm_gris_snr.set_bad('w',1.)
            bounds = snr_tous[~np.isinf(snr_tous)]
            plt.imshow(X_mpm_est.T, interpolation='nearest', origin='lower', cmap=cm_gris_snr,vmin=min(x_range[x_range!=0]),vmax=max(x_range)); 
#            loc = x_range*(nb_level_x-1)/(nb_level_x) #+0.5/(nb_level_x)
            loc = np.linspace(0.5/nb_level_x,1-0.5/nb_level_x,nb_level_x+1)
#            loc = 0.5/nb_level_x + np.arange(nb_level_x)/float(nb_level_x)
            cb=plt.colorbar(fraction=0.046,pad=0.04,aspect='auto',shrink=float(S1)/S0)
            cb.set_ticks(loc)
            cb.set_label('SNR')
            cb.set_ticklabels(['{:4.2f}'.format(l) for l in bounds])
        else:
            plt.imshow(X_mpm_est.T, interpolation='nearest', origin='lower', cmap=cm_gris,vmin=0,vmax=1); 
        
        plt.title('$\\hat{\mathbf{x}}_{\mathrm{MPM}}$')
        

        
        
        
        plt.subplot(nb_li,nb_col,3)
        plt.imshow(Ux_map.T, interpolation='nearest', origin='lower', cmap=plt.cm.gray,vmin=0,vmax=1); 
        #plt.axis('off')
        plt.title('$\\hat{\mathbf{u}}^x$')
        plt.colorbar(fraction=0.046,pad=0.04,shrink=float(S1)/S0)
        
        
        if tmf==True:
            plt.subplot(nb_li,nb_col,nb_col+2)
            plt.imshow(V_mpm_est.T, interpolation='nearest', origin='lower',cmap=cm_angl,vmin=0,vmax=np.pi); 
            plot_directions(V_mpm_est.T, np.ones_like(V_mpm_est.T),pas=8)
            
            plt.title('$\\hat{\mathbf{v}}_{\mathrm{MPM}}$ ')
                    
            
            
            plt.subplot(nb_li,nb_col,nb_col+3)
            plt.title('$\\hat{\mathbf{u}}^v$')
            plt.imshow(Uv_map.T, interpolation='nearest', origin='lower', cmap=plt.cm.gray,vmin=0,vmax=1); 
            plt.colorbar(fraction=0.046,pad=0.04,shrink=float(S1)/S0)
        
        
        
        
        ax1 =plt.subplot(nb_li,nb_col,((nb_li-1)*nb_col+1,(nb_li-1)*nb_col+2))
        cmcop = plt.cm.get_cmap('coolwarm',x_range.size)
        #ax1.set_color_cycle(plt.cm.get_cmap('copper',x_range.size+2))
        #mu_sem_tous = parsem.mu[:,np.newaxis] * x_range[np.newaxis,:]
        for id_x in range(x_range.size):
            plt.plot(parsem.mu*x_range[id_x],color=cmcop(id_x),linewidth=2)
        ax1.errorbar(1,parsem.mu[1],yerr=parsem.sig,color='r',linewidth=2)#np.array([parsem.mu[0]-parsem.sig,parsem.mu[0]+parsem.sig]))
#        plt.xlim([0,W-1])
        if x_range.size!=2:
            plt.legend(np.round(bounds*100)/100.)
        plt.xlabel('$\\lambda$')
        plt.ylabel('I')
        
        
        plt.tight_layout()
#        if nom_cube=='hdfs':
#            if x_range.size==2:
#                plt.savefig('./results/'+nom_cube+'_hmf/cat%.0f-bin.png'%src.id, format='png',dpi=150)
#            else:
#                plt.savefig('./results/'+nom_cube+'_hmf/cat%.0f-xmu.png'%src.id, format='png',dpi=150)
#                
#        else:
#            if x_range.size==2:
#                plt.savefig('./results/'+nom_cube+'_hmf/cat%.0f-bin.png'%num_obj, format='png',dpi=150)
#            else:
#                plt.savefig('./results/'+nom_cube+'_hmf/cat%.0f-xmu.png'%num_obj, format='png',dpi=150)
#        plt.close()
        
        
        
        
        
        #%%
#        plt.rc('text', usetex=True)
#        plt.rc('font',family='serif')
#        nb_col = 4
#        nb_li = 2
#        Fig = plt.figure(figsize=(nb_col*5,5.5*nb_li))
#        
#        plt.subplot(nb_li,nb_col,1)
#        plt.imshow(Y_ms.mean(axis=2).T,cmap=plt.cm.gray_r,interpolation='nearest',origin='lower')
#        plt.title('ID %.0f - White image, median sub'%src.id)
#        
#        plt.subplot(nb_li,nb_col,3)
#        plt.imshow(X_mpm_est.T,cmap=plt.cm.gray_r,interpolation='nearest',origin='lower')
#        plt.title('$\\hat{X}^{\mathrm{MPM}}$ pour $\\hat{\\mu}^{\mathrm{MPM}}$')
#        
#        
#        
#        plt.subplot(nb_li,nb_col,2)
#        
#        plt.fill_between(np.arange(W),parchamp.mu+parchamp.sig,parchamp.mu-parchamp.sig,facecolor='red',alpha=0.5)
#        plt.plot(parchamp.mu,'k',linewidth=2)
#        #plt.plot(parchamp.mu_sem[0,:],'b',linewidth=1)
#        plt.title('$\\hat{\\mu}^{\mathrm{MPM}} \pm \\hat{\sigma}^{\mathrm{MPM}}$  ')
#        plt.xlim((0,W-1))
#        
#        
#        for c in range(4):
#            plt.subplot(nb_li,nb_col,nb_col+c+1)
#            plt.imshow(X_mpm_mult[:,:,3-c].T,cmap=plt.cm.coolwarm,interpolation='nearest',origin='lower',vmin=0,vmax=1)
#            plt.title('$\\overline{X^{MPM}}$ pour $%.1f \\times \\hat{\\mu}^{\mathrm{MPM}}$'%range_coeff[3-c])
#        
#        plt.subplot(nb_li,nb_col,4)  
#        
#        carte = X_est_mult.sum(axis=2)
#        carte[carte==0] +=np.nan
#        plt.imshow(Y_ms.mean(axis=2).T,cmap=plt.cm.gray_r,interpolation='nearest',origin='lower')
#        plt.contourf(carte.T,10,labels=range_coeff,antialiased=False,alpha=0.5,cmap=plt.cm.jet)
#        plt.xlim((0.5,S0-1.5))
#        plt.ylim((0.5,S1-1.5))
#        plt.title('Segmentations avec $\\lbrace 0.1, 0.2, \\ldots, 1 \\rbrace \\times \\hat{\\mu}^{\mathrm{MPM}}$ ')
#        
#        
#        
#        
#        
#        if HMF==False:
#            plt.subplot(nb_li,nb_col,3)
#            plt.imshow(V_mpm_est,cmap=plt.cm.Spectral,interpolation='nearest',origin='lower',vmin=0,vmax=np.pi)
#            auxmap = np.copy(X_mpm_est).astype(float);     auxmap[X_mpm_est==0]=np.nan
#            plt.imshow(auxmap,cmap=plt.cm.gray,interpolation='nearest',origin='lower', alpha=0.33)
#            plot_directions(V_mpm_est, np.ones_like(V_mpm_est),pas=4)
#            plt.title('$\\hat{V}^{\mathrm{MPM}}$')
#        
#            plt.subplot(nb_li,nb_col,nb_col+3)
#            plt.imshow(V_mpm.mean(axis=2),cmap=plt.cm.Spectral,interpolation='nearest',origin='lower', vmin=0,vmax=np.pi)
#            plt.imshow(auxmap,cmap=plt.cm.gray,interpolation='nearest',origin='lower', alpha=0.33)
#            plot_directions(V_mpm.mean(axis=2), np.ones_like(V_mpm_est),pas=4)
#            plt.title('$\\overline{V^{MPM}}$')
#        
#        plt.tight_layout()
#        
#        plt.savefig('./results/cat%.0f-xmu.png'%src.id, format='png',dpi=150)

#%%
        
        #doc_pdf.savefig(Fig)
#        plt.close(Fig)


#doc_pdf.close()     
#
#nb_li = 1
#nb_col=1
#%%
#Fig = plt.figure(figsize=(nb_col*4.5,4*nb_li))
#im_norm = np.linalg.norm(Y_ms,axis=2)
#ecart_type = np.std(im_norm)
#plt.imshow(im_norm)
#plt.contour(im_norm > 3*ecart_type,1,colors='r')

#%%
#X = np.copy(X_mpm_est)
#Y = np.copy(Y_ms)
#Y[:20,:5,:] += np.nan
#
#W = Y.shape[2]
#liste_vec = np.reshape(Y,(Y.shape[0]*Y.shape[1],W))
#nanmap = np.isnan(Y).any(axis=2)
#msk = (nanmap).reshape(Y.shape[0]*Y.shape[1])  
#Y_msk = ma.masked_array(Y, np.tile(nanmap, (1,1,W)))
#
#
#mu = ma.mean(ma.mean(Y_msk,axis=0),axis=0)
##liste_msk = ma.masked_array(liste_vec, np.tile(msk[:,np.newaxis]==0,(1,W)))
##liste_sans_nan = ma.compress_rows(liste_msk)
##mean - vector
##mu = (Y*X[:,:,np.newaxis]).sum(axis=(0,1))/(X).sum()
##mu = ma.mean(liste_msk,axis=0)
##print mu.shape
## standard deviation - real
#Y_manip = np.copy(Y)
#for x_inst in (0,1):
#    
#    Y_manip  = Y_manip - x_inst * mu[np.newaxis,np.newaxis,:] *  (X[:,:,np.newaxis]==x_inst)
#
#liste_vec = np.reshape(Y_manip,(Y.shape[0]*Y.shape[1],W))
##    nanmap = np.isnan(Y).any(axis=2)
##    msk = (nanmap).reshape(Y.shape[0]*Y.shape[1])  
#
#liste_msk = ma.masked_array(liste_vec, np.tile(msk[:,np.newaxis],(1,W)))
#liste_sans_nan = ma.compress_rows(liste_msk)
#
#
##sig = np.std(Y_manip)    
#
#Sigma = np.ma.cov(liste_msk,rowvar=False)
#
#sig = np.sqrt(Sigma[np.eye(W)==1].mean())
#        #doc_pdf.savefig(Fig)
#        #plt.close(Fig)
#print sig
#
##doc_pdf.close()
#rho_1= Sigma[ (np.eye(W,k=-1)+np.eye(W,k=1))==1].mean()
#rho_2= Sigma[ (np.eye(W,k=-2)+np.eye(W,k=2))==1].mean()

# In[1]:


#Voici quelques questions :
#    - que se passe-t-il si on diminue/augmente la taille des spectres ?
#    - idem taille de l'image, hors effets d'objets brillants
#    - traiter/ignorer les nans dans les algos directement. Permettra le masquage !
# In[24]:

#------- Spectra retrieving ----#

# In[ ]:


#
#
#plt.figure(figsize=(5,5))
#plt.imshow(Y.mean(axis=2).T, interpolation='nearest', origin='lower', cmap=plt.cm.gray,vmin = -1,vmax=1);
##plt.axis('off')
#plt.tight_layout()
#plt.savefig('./figures/udfnoise.png', format='png',dpi=200)
#
#
##
##plt.figure(figsize=(5,2.5))
##plt.plot(Y[20,20,:], linewidth=2,color='gray')
##plt.xlabel('$\\lambda$')
##plt.xlim((0,19))
##plt.ylim((-1,1))
##plt.grid()
##plt.tight_layout()
##plt.savefig('./figures/udfmunoise.png', format='png',dpi=200)
#
#
plt.figure(figsize=(5,5))

plt.imshow(V_mpm_est.T, interpolation='nearest', origin='lower', cmap=cm_angl,vmin=0,vmax=np.pi); 
plot_directions(V_mpm_est.T, np.ones_like(V_mpm_est.T),pas=8)
#plt.axis('off')
plt.tight_layout()
plt.savefig('./figures/udfangleest.png', format='png',dpi=200)

#
#

#%%
plt.figure(figsize=(5,5))
X_mpm_est[np.isnan(X_mpm_est)] = 0
plt.imshow(X_mpm_est.T, interpolation='nearest', origin='lower', cmap=plt.cm.gray); 
#plt.axis('off')
plt.tight_layout()
plt.savefig('./figures/udfest.png', format='png',dpi=200)


##%%
#plt.figure(figsize=(5,3))
#X_mpm_est[0==(X_mpm_est)] = np.nan
#plt.imshow(X_mpm_est.T, interpolation='nearest', origin='lower', cmap=cm_gris_snr); 
#plt.axis('off')
#plt.tight_layout()
#plt.savefig('./figures/udfest_noaxis.png', format='png',dpi=200)
#
#%%
plt.figure(figsize=(5,3))
cmcop = plt.cm.get_cmap('gray',x_range.size)
  
i=0
for x_inst in x_range[x_range!=0]:
    plt.plot(parsem.mu*x_inst,color=cmcop(i),linewidth=2)
    i+=1
    
#ax1.errorbar(1,parsem.mu[1],yerr=parsem.sig,color='r',linewidth=2)#np.array([parsem.mu[0]-parsem.sig,parsem.mu[0]+parsem.sig]))
plt.xlim([0,W-1])
plt.grid()
#if x_range.size!=2:
#    plt.legend(np.round(bounds*100)/100.)
plt.xlabel('$\\lambda$')
plt.ylabel('I')
plt.tight_layout()
plt.savefig('./figures/udfmuref.png', format='png',dpi=200)
#
#
#%%
plt.figure(figsize=(5,3))
cmcop = plt.cm.get_cmap('gray',x_range.size)
#ax1.set_color_cycle(plt.cm.get_cmap('copper',x_range.size+2))
#mu_sem_tous = parsem.mu[:,np.newaxis] * x_range[np.newaxis,:]
m = [':','--','-','-.']
i=0
for x_inst in x_range[x_range!=0]:
    msk = (X_mpm_est==x_inst)    
    Y_msk = ma.masked_array(Y, np.tile(1-msk[:,:,np.newaxis],(1,1,W)))
    sp_mean = ma.mean(ma.mean(Y_msk,axis=0),axis=0)
    plt.plot(sp_mean,color=cmcop(i),linewidth=2,)
    i+=1
plt.xlim([0,W-1])
plt.grid()
#if x_range.size!=2:
#    plt.legend(np.round(bounds*100)/100.)
plt.xlabel('$\\lambda$')
plt.ylabel('I')
plt.tight_layout()
plt.savefig('./figures/udfmuest.png', format='png',dpi=200)

#%%


cmcop = plt.cm.get_cmap('gray',x_range.size*2)
plt.figure(figsize=(5,5))
X_mpm_est[np.isnan(X_mpm_est)] = 0
plt.imshow(X_mpm_est.T, interpolation='nearest', origin='lower', cmap=plt.cm.coolwarm); 

plt.contourf(V_mpm_est.T==v_range[0],1,colors='none',hatches=['\\','/'],extend='lower')
#plt.axis('off')
plt.tight_layout()
plt.savefig('./figures/udfestjoint.png', format='png',dpi=200)