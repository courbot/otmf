# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:59:34 2016

@author: courbot
"""
import numpy as np
import scipy.stats as st

import image_tools as it
import fields_tools as ft
import gc
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt

from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter 
    
def init_champs(par):
    
    S0 = par.S0
    S1 = par.S1

    if par.init_method == 'astro':
        centre1 = np.array([S0/2,S1/2])
        r1 = 5
        y,x = np.ogrid[0:S0,0:S1]
        
        cercle1 = 1-( (x-centre1[0])**2+(y-centre1[1])**2 ).astype(float)/r1**2 
        dist_centre = np.sqrt((x-centre1[0])**2+(y-centre1[1])**2 )


    if par.fuzzy == False:
        X_init = np.random.choice(par.x_range,size=(par.S0,par.S1)) 
        
        if par.init_method=='astro':
            X_init[cercle1>0] = 1.# += 0.5* (cercle1>0)
            X_init[dist_centre > dist_centre[int(S0/2),0]]  = 0
        
    elif par.fuzzy == True:
        
        ran_fuzz = np.arange(0,par.nb_fuzzy+1)/par.nb_fuzzy
        if par.init_method=='astro':
                        
            X_init = np.random.choice(ran_fuzz, size=(par.S0,par.S1)) - .25/(par.nb_fuzzy+1) 
                
            X_init += 0.5* (cercle1>0)
            X_init[dist_centre > 0.9*dist_centre[int(S0/2),0]]  -= 0.5
                    
            X_init[X_init>1] = 1
            X_init[X_init<0] = 0 
        else:
            X_init = np.random.choice(ran_fuzz, size=(par.S0,par.S1)) #- .25/(par.nb_fuzzy+1) 


    
    return X_init

def cast_angles(V,v_range):
    
    V_new = np.zeros_like(V)
    
    
    decal = (v_range[1]-v_range[0])/2
    
    # il nous faut construire des intervalles sur 0,pi
    #vmin = 0
    #vmax = v_range[0]/2
    #V_new[(vc>=vmin)*(vc<vmax)] = v_range[-1]
    V_new = np.zeros_like(V)
    for i in range(v_range.size):
        
    
        vmin = (v_range[i] - decal)
        vmax = (v_range[i]+decal)
        Vcb = V%np.pi
        
#        print vmin, vmax
        if vmin < 0 :
            # intervalle supplementaire du cote de pi
            vmin_bis = vmin+np.pi
            vmax_bis = np.pi
            
            V_new[(Vcb>=vmin)*(Vcb<vmax)] = v_range[i]  
            V_new[(Vcb>=vmin_bis)*(Vcb<vmax_bis)] = v_range[i]  
#            print vmin_bis, vmax_bis
        elif vmax > np.pi:
            vmin_bis = 0
            vmax_bis = vmax-np.pi
            
            V_new[(Vcb>=vmin)*(Vcb<vmax)] = v_range[i]  
            V_new[(Vcb>=vmin_bis)*(Vcb<vmax_bis)] = v_range[i]  
#            print vmin_bis, vmax_bis
            
        else:
            V_new[(Vcb>=vmin)*(Vcb<vmax)] = v_range[i]      
    
#    decal = v_range[0]/2.
#    
#    V_new = ((V+decal)-(V+decal)%(v_range[0]))%np.pi
    
    
#    v_range_new = v_range[v_range!=0]
#    for i in range(V.shape[0]):
#        for j in range(V.shape[1]):
#            if np.isnan(V[i,j])==0:
#                ecart = np.abs(V[i,j] - v_range)%np.pi
#                ind_min = np.argmin(ecart)
#        
#                V_new[i,j] =v_range[ind_min]   
    return V_new

def get_dir(X,par):
    
    S0 = par.S0
    S1 = par.S1
    v_range = par.v_range

    if par.multi==False:
#        X_fil = gaussian_filter(X.astype(float), sigma=(0.5,0.5))
        X_fil = np.copy(X).astype(float)
    else:
#        if par.Y.shape[2]==1:
#            X_fil = np.copy(par.Y[:,:,0])
#        else:
        X_fil =gaussian_filter(X.astype(float), sigma=(1,1))
#        X_fil = np.copy(X).astype(float)
        
    dx,dy = np.gradient(X_fil.astype('float'))
    an = np.arctan2(dy,dx)
        
    
    an = (an+np.pi/2)%np.pi

    mask = (((dx==0)*(dy==0))) > 0

    ux, uy = np.ogrid[0:S0,0:S1]
    ux = np.tile(ux,(1,S1))
    uy = np.tile(uy,(S0,1))
    #
    
    
    image = np.copy(an)
    image[mask]+=np.nan
    
    ########## Actual interpolation
    valid_mask = ~np.isnan(image)
    coords = np.array(np.nonzero(valid_mask)).T
    values = image[valid_mask]
    
    it = interpolate.LinearNDInterpolator(coords, values)
    
    an_interp = it(list(np.ndindex(image.shape))).reshape(image.shape)

    
    # Interpolation fails outside of a convex hull. Misisng values are randomly filled.
    ani0 =an_interp[np.isnan(an_interp)]
    an_interp[np.isnan(an_interp)] = np.random.choice(v_range,size=ani0.size)
    
    ############## Recasting into the known range
    an_interp_fil = gaussian_filter(an_interp.astype(float), sigma=(2,2))
    
    
    an_interp2 = cast_angles(an_interp_fil, v_range)
    #
    return an_interp2   


def calc_likelihood(Y, x_range, multi,par):
    
        S0,S1,W = Y.shape
        num_x = x_range.size
        likelihood_y = np.ones(shape=(S0,S1,num_x))       
        parc = par.parchamp
        if multi: # multiclasse
            if W == 1:#monoband
                #ici
                for id_x in range(num_x):  

                    likelihood_y[:,:,id_x] = st.norm.pdf(Y[:,:,0], loc=parc.mu[id_x],scale=parc.sig[id_x])
            else:
                
                Sigma = np.eye(W) * parc.sig**2 + (np.eye(W,k=1) +  np.eye(W,k=-1)) * parc.rho_1 + (np.eye(W,k=2) +  np.eye(W,k=-2)) * parc.rho_2       
                
                for id_x in range(num_x):
                    likelihood_y[:,:,id_x] = st.multivariate_normal.pdf(Y, mean=par.mu[:,id_x],cov=Sigma)        
        else: # classes vs bruit
            if W == 1:
                for id_x in range(num_x):  
                    likelihood_y[:,:,id_x] = st.norm.pdf(Y[:,:,0], loc=x_range[id_x]*parc.mu,scale=parc.sig)
            else:
                Sigma = np.eye(W) * parc.sig**2 + (np.eye(W,k=1) +  np.eye(W,k=-1)) * parc.rho_1 + (np.eye(W,k=2) +  np.eye(W,k=-2)) * parc.rho_2 
           
                if parc.mu.ndim==2:
                    parc.mu = parc.mu[0,:]            
                
                for id_x in range(num_x):
                    likelihood_y[:,:,id_x] = st.multivariate_normal.pdf(Y, mean=x_range[id_x]*parc.mu,cov=Sigma)   
                    

            likelihood_y[:,:,1:] = likelihood_y[:,:,1:] * parc.weights[:,:,np.newaxis]
            
            
            likelihood_sum = likelihood_y.sum(axis=2)
            likelihood_y /= likelihood_sum[:,:,np.newaxis]
            
        return likelihood_y


 
def gen_champs_fast(par, generate_v, generate_x, use_y,normal=False,use_pi=True,icm=False):
    """ Generation d'un champs de Gibbs a partir d'une initialisation
    angle en radians    
    Situations prises en compte :
    X,V |Y
    X|Y V fixe
    V|Y X fixe
    ...
    
    icm specifie si l'on est en déterministe
    
     """
   
    np.random.seed() # this is important if used in parrallel !!
    
    S0 = par.S0
    S1 = par.S1    
    fuzzy = par.fuzzy
    nb_fuzzy = par.nb_fuzzy
#    ran_fuzz = np.arange(0,nb_fuzzy+1)/nb_fuzzy
    alpha = par.alpha 
    alpha_v = par.alpha_v
    beta = par.beta
    delta = par.delta
    phi_uni = par.phi_uni
    phi_theta_0 = par.phi_theta_0 
    multi=par.multi
    parc = par.parchamp
    
    # about convergence
    std_window_width = 30  
    
    
    ####################################
    # Label field X :
    if generate_x == True :
        v_range=par.v_range
        # Label field X
        x_range=par.x_range
        num_x = x_range.size
        if hasattr(par,'X_init')==0:
            X_init = init_champs(par)   
        else:
            X_init = par.X_init
            
        X_courant = np.copy(X_init)
        X_res = np.zeros(shape=(S0,S1,par.nb_iter+1))
        
        X_res[:,:,0] = X_init

    else:
        if hasattr(par,'X'):
            X = par.X
        else:
            X = np.zeros(shape=(S0,S1))

    ####################################
    # Auxiliarry field V :      
    if generate_v == True:
        v_range=par.v_range
        num_v = v_range.size  
        

        if hasattr(par,'V_init')==0: # Attention ici !!!
            
            if generate_x==False: #U|X        
                V_init = get_dir(X,par)
            else:
                V_init  = np.random.choice(v_range,size=(S0,S1))
            
        else:
            V_init = par.V_init
            
        V_courant = np.copy(V_init)
        V_res = np.zeros(shape=(S0,S1,par.nb_iter+1))
        V_res[:,:,0] = V_init
    else:
        V = par.V

    ################################### 
    if use_y == True:

        likelihood_y = calc_likelihood(par.Y, x_range, multi,par)
 

    
    # To speed up the gibbs sampler
    # defining quadrants
    if normal == False:
        # parcours sur un graphe "colore"
    
#        dq =  np.array([[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3],
#                        [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]  ])
        dq =  np.array([[0,1,1,0],
                        [0,0,1,1]])
                        
        pas_q = 2
        ran = xrange(pas_q**2)
        
    else:
        # parcours en ligne standard
        dq=np.zeros(shape=(2,S0*S1))
        q = 0
        for i in range(S0):
            for j in range(S1):
                dq[0,q] = i
                dq[1,q] = j
                q+=1
                
        pas_q = S0
                
        ran = xrange(S0*S1)

        
    # Retrieving locals neighborhoods:
    # pour le cas où on ne genere pas V
    Vois = par.Vois
    Beta = np.ones_like(Vois)
    beta_sum =2*np.abs(np.cos(v_range[v_range!=0])).sum()
    if generate_v == False:
        for i in xrange(S0):
            for j in xrange(S1):
                if np.isnan(V[i,j])==0:
                     Beta[i,j,:] =  ft.gen_beta(Vois[i,j],V[i,j],phi_theta_0,beta_sum) # il faudra en faire un qui gere les images

    for k in xrange(par.nb_iter):
        # random permutations of the quadrant order :
        i=np.random.permutation(np.arange(dq.shape[1]))
        dq = dq[:,i]
        if (generate_v==True) and (k <=std_window_width) and  (hasattr(par,'v_help')) :
            if (par.v_help==True) :
                if hasattr(par,'V_init'): 
                    V_courant = par.V_init
                else:
                    V_courant = get_dir(X_courant,par)


        # run over different disjoint neighborhhod sites
        for q in ran:
            
            vois_tr = Vois[dq[0,q]::pas_q,dq[1,q]::pas_q,:]            
            
            # retrieving appropriate X field (current, or fixed)
            if generate_x == True:
                vals_tout_x = it.get_vals_voisins_tout(X_courant)
                
            else:
                vals_tout_x = it.get_vals_voisins_tout(X)
            
            vals_tr_x = vals_tout_x[dq[0,q]::pas_q,dq[1,q]::pas_q,:] # this cannot be out of the 'for q' loop !
                
            # retrieving appropriate V field (current, or fixed)
            if generate_v == True:
                vals_tout_v = it.get_vals_voisins_tout(V_courant)
            else:
                vals_tout_v = it.get_vals_voisins_tout(V)
            vals_tr_v = vals_tout_v[dq[0,q]::pas_q,dq[1,q]::pas_q,:] # this cannot be out of the 'for q' loop !
            
            if use_y == True:
                likelihood_y_tr = likelihood_y[dq[0,q]::pas_q,dq[1,q]::pas_q,:]    
            else:
                likelihood_y_tr = np.ones_like(vals_tout_x)
            
            
                    #-----------------------------------#
                    
                    
            
            if (generate_x==True) and (generate_v == False): # we generate X |U, eventually X|V,Y
                
                beta_tr = Beta[dq[0,q]::pas_q,dq[1,q]::pas_q,:]
                
                probas = np.zeros(shape=(vois_tr.shape[0],vois_tr.shape[1],num_x))
                if use_pi==False:
                    for id_x in range(num_x):                
                        probas[:,:,id_x] = calc_proba_xyv(x_range[id_x],0, likelihood_y_tr, vals_tr_x,0, beta_tr, fuzzy,nb_fuzzy, alpha,alpha_v,beta,delta,phi_uni,vois_tr) # changer !
                else: 
                    for id_x in range(num_x):  
                        probas[:,:,id_x] = calc_proba_xyv_pi(x_range,v_range,id_x,0, likelihood_y_tr,use_y, vals_tr_x,0, beta_tr,fuzzy,nb_fuzzy, parc.pi, alpha,alpha_v,delta,phi_uni,vois_tr)                            


                probas_sum = probas.sum(axis=2)
                probas_norm = probas / probas_sum[:,:,np.newaxis]
                
                if icm==False:
                    r =  np.random.random(size=(vois_tr.shape[0],vois_tr.shape[1]))
    
                    choice = np.zeros_like(r)       
                    
                    cumsum = np.cumsum(probas_norm,axis=2)
    
                    for id_x in range(num_x):
                        if id_x == 0:
                            vmin = 0 # actually we should call that pmin and pmax
                        else:
                            vmin = cumsum[:,:,id_x-1]
                            
                        vmax = cumsum[:,:,id_x]
    
                        msk_choice = (r >=vmin)*(r<vmax)
                        choice[msk_choice] = x_range[id_x]
                else:
                    id_choice = np.argmax(probas_norm,axis=2)
                    choice = np.zeros(shape=(vois_tr.shape[0],vois_tr.shape[1]))
                    
                    for id_x in range(num_x):
                        choice[id_choice==id_x] = x_range[id_x]
                        
               
                X_courant[dq[0,q]::pas_q,dq[1,q]::pas_q] = choice
       
                    
                    #-----------------------------------#
                    
                    
            elif (generate_x==False) and (generate_v == True): # we generate U|X, eventually V,|X,Y
            
                probas =  np.zeros(shape=(vois_tr.shape[0],vois_tr.shape[1],num_v))

                # enumerating different cases
                for id_v in range(num_v):

                    Beta_tr = ft.gen_beta(Vois[dq[0,q]::pas_q,dq[1,q]::pas_q,:],v_range[id_v],phi_theta_0,beta_sum)

                    if use_pi==False:
                        probas[:,:,id_v] = calc_proba_xyv(0,v_range[id_v], likelihood_y_tr, 0,vals_tr_v, Beta_tr, fuzzy,nb_fuzzy, alpha,alpha_v,beta,delta,phi_uni,vois_tr) 
                    else:
                        probas[:,:,id_v] = calc_proba_xyv_pi(x_range,v_range,0,v_range[id_v], likelihood_y_tr,use_y, 0,vals_tr_v, Beta_tr, fuzzy,nb_fuzzy, parc.pi,delta,phi_uni,vois_tr)                     
                    
                    
                probas_sum = probas.sum(axis=2)
                probas_norm = probas / probas_sum[:,:,np.newaxis]
                
                if icm==False:
                    r = np.random.random(size=(vois_tr.shape[0],vois_tr.shape[1]))
    
                    choice = np.zeros_like(r)       
                    
                    cumsum = np.cumsum(probas_norm,axis=2)
    
                    for id_v in range(num_v):
                        if id_v == 0:
                            vmin = 0
                        else:
                            vmin = cumsum[:,:,id_v-1]
                            
                        vmax = cumsum[:,:,id_v]
    
                        msk_choice = (r >=vmin)*(r<vmax)
                        choice[msk_choice] = v_range[id_v]
                        
                else: #cas deterministe:
                    id_choice = np.argmax(probas_norm,axis=2)
                    choice = np.zeros(shape=(vois_tr.shape[0],vois_tr.shape[1]))
                    
                    for id_v in range(num_v):
                        choice[id_choice==id_v] = v_range[id_v]
                
                V_courant[dq[0,q]::pas_q,dq[1,q]::pas_q] = choice
        
                    #-----------------------------------#
                    
                    
            elif (generate_x==True) and (generate_v == True): # we generate U|X, eventually X,V, eventually X,V|Y
                
                probas =  np.zeros(shape=(vois_tr.shape[0],vois_tr.shape[1],num_v*num_x))

                # enumerating different cases
                for id_x in range(num_x):
                    
                    for id_v in range(num_v):
                        v = v_range[id_v]
                        
                        Beta_tr = ft.gen_beta(Vois[dq[0,q]::pas_q,dq[1,q]::pas_q,:],v,phi_theta_0,beta_sum)
                        
                        if use_pi==False:
                            probas[:,:,id_x*num_v+id_v] = calc_proba_xyv(x_range[id_x],v_range[id_v], likelihood_y_tr, vals_tr_x,vals_tr_v, Beta_tr, fuzzy,nb_fuzzy, alpha, alpha_v,beta,delta,phi_uni,vois_tr) 
                        else:
                            probas[:,:,id_x*num_v+id_v] = calc_proba_xyv_pi(x_range,v_range,id_x,v_range[id_v], likelihood_y_tr,use_y, vals_tr_x,vals_tr_v, Beta_tr, fuzzy,nb_fuzzy, parc.pi,alpha,alpha_v,delta,phi_uni,vois_tr) 
             
                probas_sum = probas.sum(axis=2)
                probas_norm = probas / probas_sum[:,:,np.newaxis]
                
                if icm==False:
                        r = np.random.random(size=(vois_tr.shape[0],vois_tr.shape[1]))
            
                        choice_x = np.zeros_like(r)    
                        choice_v = np.zeros_like(r)       
                        
                        cumsum = np.cumsum(probas_norm,axis=2)
            
                        for id_x in range(num_x):
                            for id_v in range(num_v):
                                indice = id_x*num_v + id_v
                                if (id_v == 0) and (id_x==0):
                                    vmin = 0
                                else:
                                    vmin = cumsum[:,:,indice-1]
                                    
                                vmax = cumsum[:,:,indice]
            
                                msk_choice = (r >= vmin) * ( r < vmax)
                                
                                choice_x[msk_choice] = x_range[id_x]
                                choice_v[msk_choice] = v_range[id_v]
                else:
                    id_choice = np.argmax(probas_norm,axis=2)
                    choice_x = np.zeros(shape=(vois_tr.shape[0],vois_tr.shape[1]))
                    choice_v = np.zeros_like(choice_x)
                    
                    for id_x in range(num_x):
                            for id_v in range(num_v):
                                indice = id_x*num_v + id_v
                                
                                choice_x[id_choice==indice] = x_range[id_x]
                                choice_v[id_choice==indice] = v_range[id_v]

                V_courant[dq[0,q]::pas_q,dq[1,q]::pas_q] = choice_v
                X_courant[dq[0,q]::pas_q,dq[1,q]::pas_q] = choice_x
                
         
       
                    #-----------------------------------#
                    
                    
        if (k >= std_window_width) and (par.autoconv==True) :

            nb_diff = 0
            
            if generate_x ==True:
                nb_diff += (X_courant!=X_res[:,:,k]).sum()
                
            if generate_v==True:
                nb_diff += (V_courant!=V_res[:,:,k]).sum()
                
            ecart_relatif = nb_diff / (S0*S1)    
        
            if generate_x==True and generate_v==True:
                ecart_relatif /=2.
         
            
            if ecart_relatif <= par.thr_conv:
                # retrieving results
                if generate_x==True:
                    
                    X_res[:,:,k+1] = X_courant    
                    X_res = X_res[:,:,:k+2]
                    
                    par.X_res = X_res
                    
                if generate_v==True:
                    V_res[:,:,k+1] = V_courant    
                    V_res = V_res[:,:,:k+2] 
                    
                    par.V_res = V_res

                par.nb_iter_conv = k

                break 

        if generate_x==True:       
            X_res[:,:,k+1] = X_courant   
            par.X_res = X_res
                        
        if generate_v==True:
            V_res[:,:,k+1] = V_courant  
            par.V_res = V_res

        par.nb_iter_conv = k
        
    gc.collect()  
    
    return par
    


   
def calc_proba_xyv(x_range,id_x,v,likelihood,vals,vals_v, Beta, fuzzy,nb_fuzzy, alpha,alpha_v,beta,delta,phi_uni,vois):
    
    
    energie_x = (Beta * ft.psi_ising( x_range[id_x], vals,fuzzy,alpha,beta,delta,phi_uni)    )#*(vois >=0) 
   
    energie_v = ( ft.psi_ising( v, vals_v,fuzzy,alpha_v,beta,delta,phi_uni)    )*np.ones_like(vois)#(vois >=0) 
    # /!\ si on ajoute un facteur Beta
    
    #print energie_v
    
    energie_y = np.log(likelihood[:,:,id_x]) 
    
    proba_courant = np.exp( - energie_x.sum(axis=2) - energie_v.sum(axis=2) +  energie_y )


    return proba_courant      


def calc_proba_xyv_pi(x_range,v_range,id_x,v,likelihood,use_y, vals,vals_v, Beta, fuzzy,nb_fuzzy, pi, alpha,alpha_v,delta,phi_uni,vois):
    
    # Dans l'utilisation : calc_proba_xyv_pi(x_range,v_range,id_x,v_range[id_v], likelihood_y_tr,use_y, vals_tr_x,vals_tr_v, Beta_tr, fuzzy,nb_fuzzy, parc.pi,delta,phi_uni,vois_tr) 
             
    # Ponderation par les "fractions d'abondance" pi
    ## pour x

    if isinstance(vals,(np.ndarray)):
        
        
        energie_x_sans_v =  ft.psi_ising( x_range[id_x], vals,fuzzy,1.,delta,phi_uni)
#        energie_x_sans_v =  ft.psi_ising( x_range[id_x], vals,fuzzy,1,delta,phi_uni)
        

        pi_x_tous = np.zeros(shape=(vois.shape[0],vois.shape[1]))
        config_x = (vals==x_range[id_x]).sum(axis=2)
        for conf in range(9):
            pi_x_tous[config_x==conf] = pi[0,conf]


        energie_x_sum =  np.log(pi_x_tous) - (Beta * energie_x_sans_v).sum(axis=2)

    else:
        energie_x_sum = 0
        
            
        
    ## pour v
    if isinstance(vals_v,(np.ndarray)):
        
        energie_v =  ft.psi_ising( v, vals_v,fuzzy,1.,delta,phi_uni)
        
        pi_v_tous = np.zeros(shape=(vois.shape[0],vois.shape[1]))
        config_v = (v==vals_v).sum(axis=2)
        for conf in range(9):
            pi_v_tous[config_v==conf] = pi[1,conf]

        
        energie_v_sum =  np.log(pi_v_tous) - energie_v.sum(axis=2)#*8./facteurs#* (facteurs==8)
        
    else:
        energie_v_sum = 0
   
    if use_y ==True :    
        
        energie_y = np.log(likelihood[:,:,id_x]) 
    else:
        energie_y = 0
    
    proba_courant = np.exp( energie_x_sum + energie_v_sum +  energie_y )
     

    return proba_courant      

