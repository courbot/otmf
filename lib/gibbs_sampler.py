# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:59:34 2016

@author: courbot
"""
import numpy as np
import scipy.stats as st
from otmf.fields_tools import gen_beta,psi_ising,get_vals_voisins_tout,init_champs
import gc

from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter 
    

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
        """ Compute the likelihood of Y given all possible classes and the noise parameters.
        
        **Note** To be simplified.
        
        This function **does not** acount for the FSF.
         
        :param ndarray Y: Hyperspectral observation
        :param ndarray x_range: set of possible x
        :param bool multi: deprecated
        :param parameter par: parameter set of the Gibbs sampling
        
        :returns: **likelihood_y** *(ndarray)* Likelihood values, aranged in (x-dim,y-dim, number of classes).  
        """
    
        S0,S1,W = Y.shape
        num_x = x_range.size
        likelihood_y = np.ones(shape=(S0,S1,num_x))       
        parc = par.parchamp
        
        for id_x in range(num_x):  
        
            likelihood_y[:,:,id_x] = st.norm.pdf(Y[:,:,0], loc=parc.mu[id_x],scale=parc.sig[id_x])
        
        likelihood_sum = likelihood_y.sum(axis=2)
        likelihood_y /= likelihood_sum[:,:,np.newaxis]
            
        return likelihood_y


 
def gen_champs_fast(par, generate_v, generate_x, use_y,normal=False,use_pi=True,icm=False):
    """Markov field simulation, either by Gibbs sampling or ICM.


    :param parameter par: parameter set of the Gibbs sampling    
    :param bool generate_v: set if we generate [True] or know [False] the V array (orientations).
    :param bool generate_x: set if we generate [True] or know [False] the X array (classes).
    :param bool use_y: set if we know [True] or ignore [False] an observation Y.
    :param bool normal: set if we use the standard [True] or chromatic [False] Gibbs sampler.
                        The second one is faster by several order of magnitude on large images.
    :param bool icm: set if the realization is deterministic (ICM) or not (Gibbs).

    :returns: **par** *(ParamsGibbs)* parameter containing the simulation output.
     """
   
    np.random.seed() # this is important if used in parrallel !!
    
    S0 = par.S0
    S1 = par.S1    


    alpha = par.alpha 
    alpha_v = par.alpha_v

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
    # Auxiliary field V :      
    if generate_v == True:
        v_range=par.v_range
        num_v = v_range.size  

        if hasattr(par,'V_init')==0: 
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
    if use_y == True: # if there is an observation, we compute the likelihood
        likelihood_y = calc_likelihood(par.Y, x_range, multi,par)
 

    
    # To speed up the gibbs sampler
    # defining quadrants
    if normal == False:
        # we are in the "chromatic" Gibbs sampler setting.
    
#        dq =  np.array([[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3],
#                        [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]  ])
        # above is an alternative choice for the grid (weak influence on the 
        # result)
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

    if generate_v == False:
        for i in xrange(S0):
            for j in xrange(S1):
                if np.isnan(V[i,j])==0:
                     Beta[i,j,:] =  gen_beta(Vois[i,j],V[i,j]) # il faudra en faire un qui gere les images

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
                vals_tout_x = get_vals_voisins_tout(X_courant)
                
            else:
                vals_tout_x = get_vals_voisins_tout(X)
            
            vals_tr_x = vals_tout_x[dq[0,q]::pas_q,dq[1,q]::pas_q,:] # this cannot be out of the 'for q' loop !
                
            # retrieving appropriate V field (current, or fixed)
            if generate_v == True:
                vals_tout_v = get_vals_voisins_tout(V_courant)
            else:
                vals_tout_v = get_vals_voisins_tout(V)
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
                        probas[:,:,id_x] = calc_proba_xyv(x_range[id_x],0, likelihood_y_tr, vals_tr_x,0, beta_tr, alpha,alpha_v,vois_tr) # changer !
                else: 
                    for id_x in range(num_x):  
                        probas[:,:,id_x] = calc_proba_xyv_pi(x_range,v_range,id_x,0, likelihood_y_tr,use_y, vals_tr_x,0, beta_tr, parc.pi, alpha,alpha_v,vois_tr)                            


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

                    Beta_tr = gen_beta(Vois[dq[0,q]::pas_q,dq[1,q]::pas_q,:],v_range[id_v])

                    if use_pi==False:
                        probas[:,:,id_v] = calc_proba_xyv(0,v_range[id_v], likelihood_y_tr, 0,vals_tr_v, Beta_tr, alpha,alpha_v,vois_tr) 
                    else:
                        probas[:,:,id_v] = calc_proba_xyv_pi(x_range,v_range,0,v_range[id_v], likelihood_y_tr,use_y, 0,vals_tr_v, Beta_tr, parc.pi,vois_tr)                     
                    
                    
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
                        
                        Beta_tr = gen_beta(Vois[dq[0,q]::pas_q,dq[1,q]::pas_q,:],v)
                        
                        if use_pi==False:
                            probas[:,:,id_x*num_v+id_v] = calc_proba_xyv(x_range[id_x],v_range[id_v], likelihood_y_tr, vals_tr_x,vals_tr_v, Beta_tr,  alpha, alpha_v,vois_tr) 
                        else:
                            probas[:,:,id_x*num_v+id_v] = calc_proba_xyv_pi(x_range,v_range,id_x,v_range[id_v], likelihood_y_tr,use_y, vals_tr_x,vals_tr_v, Beta_tr,  parc.pi,alpha,alpha_v,vois_tr) 
                                                           
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
    


   
def calc_proba_xyv(x_range,id_x,v,likelihood,vals,vals_v, Beta, alpha,alpha_v,vois):
    
    
    energie_x = Beta * psi_ising( x_range[id_x], vals,alpha)
   
    energie_v =psi_ising( v, vals_v,alpha_v)
    
    energie_y = np.log(likelihood[:,:,id_x]) 
    
    proba_courant = np.exp( - energie_x.sum(axis=2) - energie_v.sum(axis=2) +  energie_y )


    return proba_courant      


def calc_proba_xyv_pi(x_range,v_range,id_x,v,likelihood,use_y, vals,vals_v, Beta, pi, alpha,alpha_v,vois):
    """ Compute p(X, V|x_neighbor, v_neighbor, y)
    
    :param ndarray x_range: possible values for x
    :param ndarray v_range: possible values for v
    :param int id_x: indice of x classe.
    :param int id_x: value of v.
    :param ndarray likelihood: likelihood of x,v given y.
    :param bool use_y: set if we use an observation y [True] or not [False]
    :param ndarray vals: x neighbor values
    :param ndarray vals_v: v neighbor values
    :param ndarray pi: prior parameter.
    :param ndarray phi: precomputed value of phi.
    :param ndarray alpha: prior "granularity" parameter.
    
    
    :returns: **proba_courant** *(ndarray)* probability computation.
    """

    if isinstance(vals,(np.ndarray)):
        
        
        energie_x_sans_v =  psi_ising( x_range[id_x], vals,alpha)
        
        pi_x_tous = np.zeros(shape=(vois.shape[0],vois.shape[1]))
        config_x = (vals==x_range[id_x]).sum(axis=2)
        for conf in range(9):
            pi_x_tous[config_x==conf] = pi[0,conf]


        energie_x_sum =  np.log(pi_x_tous) - (Beta * energie_x_sans_v).sum(axis=2)

    else:
        energie_x_sum = 0
        
            
        
    ## pour v
    if isinstance(vals_v,(np.ndarray)):
        
        energie_v =  psi_ising( v, vals_v,alpha_v)
        
        pi_v_tous = np.zeros(shape=(vois.shape[0],vois.shape[1]))
        config_v = (v==vals_v).sum(axis=2)
        for conf in range(9):
            pi_v_tous[config_v==conf] = pi[1,conf]

        
        energie_v_sum =  np.log(pi_v_tous) - energie_v.sum(axis=2)
    else:
        energie_v_sum = 0
   
    if use_y ==True :    
        energie_y = np.log(likelihood[:,:,id_x]) 
    else:
        energie_y = 0
    
    proba_courant = np.exp( energie_x_sum + energie_v_sum +  energie_y )
     

    return proba_courant      

