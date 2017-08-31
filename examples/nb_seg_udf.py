
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


import parameters
#import image_tools as it
import gibbs_sampler as gs
import fields_tools as ft
import seg_OTMF as sot


# In[2]:

sys.path.insert(0,'/home/courbot/Programmes/mpdaf-1.2')
import mpdaf

from mpdaf.obj import WCS
from mpdaf.obj import WaveCoord
from mpdaf.obj import Image
from mpdaf.obj import Spectrum
from mpdaf.obj import Cube
#from mpdaf.obj import CubeDisk

from mpdaf.sdetect import Source, SourceList


# In[3]:

# Version information
version = 0.2
# source dir
nom_dossier='./results/sources/udf10/'


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


#cube = Cube('../data/DATACUBE-PROPVAR-ZAP_UDF-10.fits')

# In[10]:


src_list = SourceList.from_path(nom_dossier)


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
#
#
#
######################
lambda_0 = 4750    # see Bacon et al, 2015
lambda_lya = 1216
pas_spectral = 1.25 # Angstrom / spectral band

W = 20
W_aff = 300
S = 50
#
##doc_pdf = PdfPages('./results/LAE-HMF-HDFS-1.24(sample)2.pdf')
#i=1
#num_obj = 1
#for num_obj in range(10,20):#range(12,20):#range(10): #range(10):#(3,):#(0,1,4,5,9):#range(num_src.size):,
#    
#        
#        src = src_list[np.where(num_src==num_src_sort[num_obj])[0]]   
#        
#
#    

