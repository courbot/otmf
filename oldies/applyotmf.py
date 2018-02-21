# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:11:19 2017

@author: courbot
"""

# -*- coding: utf-8 -*-


__version__ = '02'

#__all__ = ['hmf', 'hmf_src']

import astropy.units as u
import numpy as np
import os.path
#import time

from mpdaf.obj import Image, Spectrum 
from mpdaf.sdetect import Source

#from .parameters import ParamsSeg, ParamsGibbs, apply_parseg_pargibbs
#from .segmentation import segment

#from .segmentation_arbre_triplet import seg_arbre_triplet
#import sys
#sys.path.insert(0,'../../HEOLHT_1.4/lib')

from heolht import parameters_heolht as pat
from heolht import strategy_detection_pose as sdp

#from matplotlib.backends.backend_pdf import PdfPages

from otmf import parameters as param_otmf
from otmf import seg_OTMF as sot

def apply_ht(S,Y):
    
    FWHM = 0.66*1/0.2
    
    pfa_bright = 0.001
    pfa_faint = 0.001        
    
     # to change if the object center is not the subcube center :
    centre=np.array([int(S/2),int(S/2)])
    params = pat.Params(Y,
                               centre,
                               pfa_bright=pfa_bright,
                               pfa_faint=pfa_faint,
                               FWHM=FWHM)
                               
                               #    params.Y_sig =  empty_cube
    params.confident=False
    
    
    Xe1,ve1,vem1,Xi,vi = sdp.detection_strategy(params) 
    
    
    return Xe1
    
    
    
def apply_OTMF(Y,X_init=None):
    print 'Segmentation OTMF...'
    
     
    S0,S1,W = Y.shape
    

    
    
    pas = np.pi/2
    v_range = np.arange(pas/2., np.pi, pas)
    
    #v_range = np.array([np.pi/2,np.pi])
    #v_range = np.array([np.pi/4,3*np.pi/4])
    ###################
    nb_level_x =1
    x_range = np.arange(0, 1+1./nb_level_x, 1./nb_level_x)
    
    
    
#    print('---------------------------------------')
    pargibbs = param_otmf.ParamsGibbs(S0 = S0,
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
    
    
    if X_init is not None:
        pargibbs.X_init=X_init
    
    
    pargibbs.W = W
    pargibbs.Y = Y
    
    
    
    #==============================================================================
    # Paramètres à fixer
    #==============================================================================
    
    incert = True #
    
    parseg = param_otmf.ParamsSeg(nb_iter_sem=51,
                                  seuil_conv = 0.05,
                                  incert = incert
                                    )
    parseg.spec_snr=False #plus tard !
    parseg.multi = False # le multiclasse discret
    parseg.weights=np.ones(shape=(S0,S1))
    
    parseg.mpm = True
    #
    
    ###==============================================================================
    ### Segmentation OTMF
    #####==============================================================================
    parseg.tmf = True
    #parseg.seuil_conv = 0.05
    pargibbs = param_otmf.apply_parseg_pargibbs(parseg,pargibbs) # transfetrt a l'autre jeu de parametre

    
    
    Y_courant = np.copy(Y)
    
    #pargibbs.X_init = X_mpm_hmf
    pargibbs.Y = Y_courant
    
    
    X_otmf, V_otmf, Ux_otmf,Uv_otmf, parsem = sot.seg_otmf(parseg,pargibbs)
    
    ## 
   
    
    return X_otmf, V_otmf, Ux_otmf,Uv_otmf, parsem    

def _otmf(cube, center, size, lbda, nb_level_x, subcont,
           unit_center, unit_size, unit_wave):
# return SIG_SEM, RHO1, RHO2, sp_sem, sub_cube_tight, sub_cube_orig
    lbda_tight = lbda
    dlbda = lbda[1] - lbda[0]
    lbda_large = [lbda[0]-dlbda, lbda[1]+dlbda]
        
    sub_cube_tight = cube.subcube(center, size, lbda_tight,
                                  unit_center=unit_center, unit_size=unit_size,
                                  unit_wave=unit_wave)
    sub_cube_orig = cube.subcube(center, size, lbda_tight,
                                  unit_center=unit_center, unit_size=unit_size,
                                  unit_wave=unit_wave)
                                  
    # normalisation
    sub_cube_tight.data = sub_cube_tight.data/np.sqrt(sub_cube_tight.var)
                                      
    if (subcont):
        sub_cube_large = cube.subcube(center, size, lbda_large,
                                  unit_center=unit_center, unit_size=unit_size,
                                  unit_wave=unit_wave)
                                  
        # normalisation
        sub_cube_large.data = sub_cube_large.data/np.sqrt(sub_cube_large.var)                          
                                  
        im_med = sub_cube_large.median(axis=0)
        sub_cube_tight -= im_med._data[np.newaxis,:,:]
    
    
#    sub_cube_tight.data = sub_cube_tight.data/np.sqrt(sub_cube_tight.var)
    #datacube to proceed
    Y = np.swapaxes(sub_cube_tight._data, 2, 0)
    S0,S1,W = Y.shape
    
    X_init = apply_ht(S0,Y)
    
    
    X_otmf, V_otmf, Ux_otmf,Uv_otmf, parsem = apply_OTMF(Y,X_init=X_init)
    # Normalization
#    Y_moy = Y.max()/2.
#    Y = Y/Y_moy
#    Y = Y/Y.std()
    


    # User parameter
#    parseg = ParamsSeg()
#    # target SNR in dB 
#    parseg.facteur = psnr
#    # set if the target SNR a peak-snr [True] or not [False]
#    parseg.psnr = use_psnr
#    #  set if we use the MPM [True] or MAP [False] segmentation criterion.
#    parseg.mpm = mpm
#    # set an eventual weighting to use in parameter estimation.
#    parseg.weights=np.ones(shape=(S0,S1))  # a enlever
#
#    parseg.nb_iter_sem = 21
#    
#    if use_FSF:
#        FSF = Moffat(dim=3,FWHM = 0.66*1/0.2, beta=2.6) # a adapter a la longueur d'onde !!
#        
#        parseg.FSF = FSF
#        parseg.use_FSF = True
#
#    pargibbs = ParamsGibbs(S0 = S0, # x-dimension
#                           S1 = S1, # y-dimension
#                           W = W, # spectral dimension
#                           )
#    pargibbs = apply_parseg_pargibbs(parseg, pargibbs)
#    pargibbs.Y = Y
#    pargibbs.X_init = X_init
#    pargibbs.FSF = FSF
#    pargibbs.use_FSF = parseg.use_FSF
#    # total number of classes
#    pargibbs.set_xrange(nb_level_x)
#    print('PMF...')
#    start = time.time()
#    X_hmf, Ux_hmf, parsem = segment(parseg,pargibbs)
#    end = time.time() - start
#    print('Time: %.2f s'%end)
#    
    # creation d'une image dumb contenant les coordonnes WCS, etc
    im1 = Image(data=X_otmf.T.astype(float), wcs=sub_cube_tight.wcs) # 
    im2 = Image(data=V_otmf.T.astype(float), wcs=sub_cube_tight.wcs)
    #if mpm:
    im3 = Image(data=Ux_otmf.T, wcs=sub_cube_tight.wcs) # carte des incertitude
    im4 = Image(data=Uv_otmf.T, wcs=sub_cube_tight.wcs)
#    else:
#        im2 = None
#
    
    sp_sem = Spectrum(data=parsem.mu[0,:], wave=sub_cube_tight.wave)
    
    #psnr = parseg.facteur
    # recuperation valeurs variance, correlations
    SIG_SEM = parsem.sig[0]
    RHO1 = parsem.rho_1
    RHO2 = parsem.rho_2
#    # recuperation des spectres
    #sp_sem = Spectrum(data= parsem['mu'][1,:], wave=sub_cube_tight.wave)
#    
#    psnr = parseg.facteur
#    # recuperation valeurs variance, correlations
#    SIG_SEM = parsem_OTMF['sigma']
#    RHO1 = parsem_OTMF['rho_1']
#    RHO2 = parsem_OTMF['rho_2']
    
    return SIG_SEM, RHO1, RHO2, sp_sem,im1,im2,im3,im4, sub_cube_tight, sub_cube_orig


def seg(cube, center, lbda, src_id, size=12., nb_level_x=2, psnr=-9,use_psnr=True,
        mpm=False, subcont=True, unit_center=u.deg, unit_size=u.arcsec, unit_wave=u.angstrom):
    """
    Parameters
    ----------
    
    cube : mpdaf.obj.Cube
           Cube object.
    center : (float,float)             
             center (y, x) of the aperture
    size : float             
           The size to extract. It corresponds to the size along the delta            
           axis and the image is square.
    lbda : float, float
            tuple giving the wavelength range.
    nb_level_x : integer
                 Total number of classes.
    psnr       : float
                 Target SNR in dB 
    mpm        : bool
                 MPM [True] or MAP [False] segmentation criterion is used.
    subcont    : bool
                 if True, subtract compute the continuum as the median value
                 along a larger wavelength range and subtract it
    unit_center : `astropy.units.Unit`             
                  Type of the center coordinates (degrees by default) 
    unit_size : `astropy.units.Unit`             
                 Unit of the size value (arcseconds by default).
    unit_wave : `astropy.units.Unit`            
                 Wavelengths unit (angstrom by default)    
    If None, inputs are in pixels  
    """
    SIG_SEM, RHO1, RHO2, sp_sem, im1,im2,im3,im4, sub_cube_tight, sub_cube_orig =_otmf(cube, 
                                          center, size, lbda, nb_level_x, subcont,
                                          unit_center, unit_size, unit_wave)
                            
    if unit_center is None:
        center = cube.wcs.pix2sky(np.array([center]), unit=u.deg)
    elif unit_center!=u.deg:
        center[0] = center[0]*unit_center.to(u.deg).value
        center[1] = center[1]*unit_center.to(u.deg).value
        
    origin = ['OTMF', __version__, os.path.basename(cube.filename),
              cube.primary_header.get('CUBE_V', '')]
    src = Source.from_data(src_id, center[1], center[0], origin)
    
    src.cubes['OTMF'] = sub_cube_tight
    
    src.images['OTMF_SEG'] = im1
    src.images['OTMF_SEG_V'] = im2
    src.images['OTMF_USEG'] = im3
    src.images['OTMF_USEG_V'] = im4
#    if im2 is not None:
#        src.images['OTMF_USEG'] = im2

    src.spectra['OTMF']=sp_sem
    
    src.add_attr('OTMF', 'otmf v%s'%__version__)
#    src.add_attr('OTMF_PSNR', psnr)
    src.add_attr('OTMF_SIG_SEM', SIG_SEM)
    src.add_attr('OTMF_RHO1', RHO1)
    src.add_attr('OTMF_RHO2', RHO2)
    
    return src
 
 
          
def OTMF_src(src, cube, lbda, size=12., nb_level_x=2, psnr=-9,use_psnr=False, use_FSF = True, mpm=True,
           subcont=True, unit_size=u.arcsec, unit_wave=u.angstrom):
    """
    
    Parameters
    ----------
    
    src  : mpdaf.sdetect.Source
           Source object.
    cube : mpdaf.obj.Cube
           Cube object.
    lbda : float, float
            tuple giving the wavelength range.
    size : float             
           The size to extract. It corresponds to the size along the delta            
           axis and the image is square.
    nb_level_x : integer
                 Total number of classes.
    psnr       : float
                 Target SNR in dB 
    mpm        : bool
                 MPM [True] or MAP [False] segmentation criterion is used. 
    subcont    : bool
                 if True, subtract compute the continuum as the median value
                 along a larger wavelength range and subtract it
    unit_size : `astropy.units.Unit`             
                 Unit of the size value (arcseconds by default).
    unit_wave : `astropy.units.Unit`            
                 Wavelengths unit (angstrom by default)       
    If None, inputs are in pixels  
    """ 

        
    center = (src.dec, src.ra)
    unit_center=u.deg
    
    SIG_SEM, RHO1, RHO2, sp_sem, im1,im2,im3,im4, sub_cube_tight, sub_cube_orig =_otmf(cube, center, size, lbda, nb_level_x, subcont,
                                                                          unit_center, unit_size, unit_wave)
    
    
#    _hmf(cube,
#               center, size, lbda, nb_level_x, psnr,use_psnr,use_FSF, mpm, subcont, unit_center,
#               unit_size, unit_wave)
               
#            :
# return SIG_SEM, RHO1, RHO2, sp_sem, sub_cube_tight, sub_cube_orig
    
    src.cubes['OTMF'] = sub_cube_tight
    
    src.images['OTMF_SEG'] = im1
    src.images['OTMF_SEG_V'] = im2
    src.images['OTMF_USEG'] = im3
    src.images['OTMF_USEG_V'] = im4
#    if im2 is not None:
#        src.images['OTMF_USEG'] = im2

    src.spectra['OTMF']=sp_sem
    
    src.add_attr('OTMF', 'otmf v%s'%__version__)
#    src.add_attr('OTMF_PSNR', psnr)
    src.add_attr('OTMF_SIG_SEM', SIG_SEM)
    src.add_attr('OTMF_RHO1', RHO1)
    src.add_attr('OTMF_RHO2', RHO2)
    
    
#def Moffat(dim, FWHM,beta):
#    """
#    Non-normalized Moffat function windowed in a square. 
#    @param dim      Square size
#    @param FWHM     Full Width at Half Maximum of the function
#    @param beta     Parameter of the Moffat function.
#    
#    @return Moff   Moffat function values.
#    """
#    demidim = np.floor(dim/2)
#    X = np.tile(np.arange(-demidim,1+demidim)[:,np.newaxis],(1,dim)) ; Y = np.tile(np.arange(-demidim,1+demidim)[np.newaxis,:],(dim,1)) ; 
#    R2 = X.astype(float)**2+Y.astype(float)**2
#
#
#    alpha = FWHM/(2*np.sqrt(2**(1/beta)-1))
#    Moff = (1 + R2/alpha**2)**(-beta) 
#    
##    Moff = Moff/Moff.sum()
#    
#    return Moff  
    
    