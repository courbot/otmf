# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 17:23:09 2016

@author: courbot
"""
from otmf import fields_tools as ft
import numpy as np

class ParamsGibbs():

    def __init__( self,
                  S0 = 50,
                  S1 = 50, 
                  nb_iter = 50,
                  fuzzy = False,
                  nb_fuzzy=32.,
                  prop_fuzzy = 0.59,
                  anisotropic=False,
                  angle = 0,
                  v_range = 0,
                  x_range = 0,
                  v_help=0,
                  nb_nn_v_help = 9,
                  phi_theta_0 = 0.25,
                  alpha = 1.,
                  alpha_v = 1.,
                  beta = 1,
                  delta = 0,
                  phi_uni = 0.,
                  type_pot = 'potts',
                  init_method = 'std',
                  thr_conv = 0.01,
                  autoconv=True,
                  multi = False
                      
                ): 

        self.S0 = S0
        self.S1 = S1
        self.nb_iter = nb_iter
        self.fuzzy = fuzzy
        self.nb_fuzzy = nb_fuzzy
        self.prop_fuzzy = prop_fuzzy
        self.anisotropic=anisotropic
        self.angle = angle 
        self.v_range = v_range
        self.x_range = x_range
        self.v_help = v_help
        self.nb_nn_v_help = nb_nn_v_help 
        self.phi_theta_0 = phi_theta_0
        self.alpha = alpha
        self.alpha_v = alpha_v
        self.beta = beta
        self.delta = delta
        self.phi_uni = phi_uni
        if type_pot=='potts':
            phi_uni = 0.
            
        self.type_pot = type_pot
        self.init_method = init_method
        self.thr_conv = thr_conv
        self.autoconv=autoconv
        
        
        Vois = np.zeros(shape=(S0,S1,8))
        for i in xrange(S0):
            for j in xrange(S1):
                Vois[i,j,:] = ft.get_num_voisins(i,j,np.zeros(shape=(S0,S1)))
        self.Vois = Vois
        
        self.multi = multi
        
        

        
class ParamsChamps():

    def __init__( self,
                  alpha = 1.,
                  phi_theta_0 = 0,
                  mu = 0,
                  sig = 0.25,
                  rho = 0
                ): 

        self.alpha = alpha
        self.phi_theta_0 = phi_theta_0
        
        self.mu = mu
        self.sig = sig
        self.rho = rho
        
        # define a parameters stack here?

class ParamsSeg():

    def __init__(   self,
                    nb_iter_sem=40, # nb d'iter maximum pour SEM
                    nb_rea = 100, # nombre de realisations de Gibbs differentes pour le MPM
                    taille_fen = 5, # fenetre pour la convergence de SEM
                    seuil_conv = 0.05, # convergence de SEM
                    
                    nb_iter_mpm = 100, # longueur max. pour les Gibbs dans le MPM
                    pargibbs_nb_iter = 100, 
                    pargibbs_autoconv=True, # convergence automatique des estimateurs de Gibbs
                    pargibbs_thr_conv = 0.001, # seuil pour cette convergence, en relatif
                    incert = True, # Utilisation ou non de segmentation avec incertitude
                    pargibbs_Xi = 0. ,  # valeur de l'"incertitude" adoptee
                    tmf = True
                ): 
        
        self.nb_iter_sem=nb_iter_sem
        self.nb_rea = nb_rea
        self.taille_fen = taille_fen
        self.seuil_conv = seuil_conv
        
        self.nb_iter_mpm = nb_iter_mpm
        self.pargibbs_nb_iter = pargibbs_nb_iter 
        self.pargibbs_autoconv=pargibbs_autoconv
        self.pargibbs_thr_conv = pargibbs_thr_conv # seuil pour cette convergence, en relatif
        self.incert = incert
        self.pargibbs_Xi = pargibbs_Xi
        self.tmf = tmf
        
def apply_parseg_pargibbs(parseg,pargibbs):
    
    pargibbs.nb_iter = parseg.pargibbs_nb_iter
    pargibbs.autoconv = parseg.pargibbs_autoconv
    pargibbs.thr_conv = parseg.pargibbs_thr_conv
    pargibbs.Xi = parseg.pargibbs_Xi
    pargibbs.multi = parseg.multi
    
    return pargibbs