# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 09:45:48 2015

@author: courbot
"""



import numpy as np 
#import numpy.ma as ma
import sys 
import matplotlib.pyplot as plt
import time
import scipy.stats as st
import scipy.ndimage.morphology as morph

from matplotlib import animation



def psi(x_1,x_2,nom='abs'):
    if nom=='abs':
        res = np.abs(x_1-x_2)
    
    elif nom=='sq':
        res = (x_1-x_2)**2
    
    elif nom=='po4':
        res = (x_1-x_2)**4
    
    return res

def phi(a_1,a_2):
    
    return (np.cos(a_1-a_2))**2

def gen_beta(vois, angle,type_psi):
    
    beta = np.zeros_like(vois) ; beta = beta.astype(float)
    pi = np.pi
    # This diminish the values on the perpendicular axis
#    angle_utile = (angle+np.pi/2)%(2*pi)
#   
#    beta[(vois==3)+(vois==7)] = psi(0,angle_utile, type_psi) 
#    beta[(vois==4)+(vois==0)] = psi(pi/4,angle_utile,type_psi)
#    beta[(vois==5)+(vois==1)] = psi(pi/2,angle_utile,type_psi)
#    beta[(vois==6)+(vois==2)] = psi(3*pi/4,angle_utile,type_psi)
#    

   
    beta[(vois==3)+(vois==7)] = phi(0,angle) 
    beta[(vois==4)+(vois==0)] = phi(pi/4,angle)
    beta[(vois==5)+(vois==1)] = phi(pi/2,angle)
    beta[(vois==6)+(vois==2)] = phi(3*pi/4,angle)
    
    beta_0 = 0
    beta = beta_0 + (1-beta_0)*beta
    #beta/=beta.max()
    #beta = np.sin(beta)
    return beta

def get_num_voisins(pos,image):
    """ Récupération des voisins d'un point dans une image, en tenant compte des bords
    
        The following numbering is used :
        
           --------------
        y+1 | 6 | 5 | 4 |
           --------------
          y | 7 |   | 3 |
           --------------
        y-1 | 0 | 1 | 2 |
           --------------
            x-1 | x | x+1
    
    """
    if pos.size==2:
        S0 = image.shape[0]-1
        S1 = image.shape[1]-1
        
        # convenience/brevity notation
        x = pos[0]
        y = pos[1]
        if x <  S0 and x > 0 and y <  S1 and y > 0 :
            voisins = np.array([0,1,2,3,4,5,6,7])            
            
        if x == 0:
            if y==0:
                voisins = np.array([3,4,5,-1,-1,-1,-1,-1])   
            elif y==S1:
                voisins = np.array([1,2,3,-1,-1,-1,-1,-1])
            else:
                voisins = np.array([1,2,3,4,5,-1,-1,-1])
        elif x == S0:
            if y == 0:
                voisins=np.array([5,6,7,-1,-1,-1,-1,-1])
            elif y == S1:
                voisins = np.array([7,0,1,-1,-1,-1,-1,-1])
            else:
                voisins = np.array([5,6,7,0,1,-1,-1,-1])
        elif y==0:
            voisins = np.array([3,4,5,6,7,-1,-1,-1])
        elif y == S1:
            voisins = np.array([7,0,1,2,3,-1,-1,-1])
        
        return voisins
        
def get_vals_voisins(pos,image):
    """ Récupération des voisins d'un point dans une image, en tenant compte des bords
    
        The following numbering is used :
        
           --------------
        y+1 | 6 | 5 | 4 |
           --------------
          y | 7 |   | 3 |
           --------------
        y-1 | 0 | 1 | 2 |
           --------------
            x-1 | x | x+1
    
    """
    if pos.size==2:
        S0 = image.shape[0]-1
        S1 = image.shape[1]-1
        
        # convenience/brevity notation
        x = pos[0]
        y = pos[1]
        if x <  S0 and x > 0 and y <  S1 and y > 0 :
            vals = np.array([image[x-1,y-1],image[x,y-1],image[x+1,y-1],image[x+1,y], image[x+1,y+1],image[x,y+1],image[x-1,y+1],image[x-1,y]])           
            
        if x == 0:
            if y==0:
                vals = np.array([image[x+1,y],image[x+1,y+1],image[x+1,y+1],0,0,0,0,0])  
            elif y==S1:
                vals = np.array([image[x,y-1],image[x+1,y-1],image[x+1,y],0,0,0,0,0])
            else:
                vals = np.array([image[x,y-1],image[x+1,y-1],image[x+1,y], image[x+1,y+1],image[x,y+1],0,0,0])
        elif x == S0:
            if y == 0:
                vals = np.array([image[x,y+1],image[x-1,y+1], image[x-1,y],0,0,0,0,0])
            elif y == S1:
                vals = np.array([image[x-1,y],image[x-1,y-1], image[x,y-1],0,0,0,0,0])
            else:
                vals = np.array([image[x,y+1],image[x-1,y+1], image[x-1,y],image[x-1,y-1], image[x,y-1] ,0,0,0])
        elif y==0:
            vals = np.array([image[x+1,y],image[x+1,y+1],image[x+1,y+1],image[x-1,y+1], image[x-1,y],0,0,0])
        elif y == S1:
            vals = np.array([image[x-1,y],image[x-1,y-1], image[x,y-1],image[x+1,y-1],image[x+1,y] ,0,0,0])
        
        return vals
        
def gen_champs(X_init,S0,S1,nb_iter, xmin,xmax,angle,type_psi='abs',isotropic=True):
    """ Generation d'un champs de Gibbs a partir d'une initialisation
    angle en radians    
    """
    X_courant = np.copy(X_init)
    proba_courant = np.zeros_like(X_init)
    num_ecart = np.zeros(shape=(nb_iter))
    X_res = np.zeros(shape=(S0,S1,nb_iter))
    
    # Retrieving locals neighborhoods:
    Vois = np.zeros(shape=(S0,S1,8))
    Beta = np.ones_like(Vois)
    for i in range(S0):
        for j in range(S1):
            Vois[i,j,:] = get_num_voisins(np.array([i,j]),X_init)
            
            if isotropic == False:
                    if np.isnan(angle[i,j])==0:
                         Beta[i,j,:] =  gen_beta(Vois[i,j],angle[i,j],type_psi)

    
    #energie_totale = np.sqrt((X_init**2).sum())
    
    # Generating all candidates:
    X_new_tout = np.random.uniform(low = xmin, high = xmax, size=(S0,S1,nb_iter))    
    
    for k in range(nb_iter):
        # Proposition de candidats
        X_new = X_new_tout[:,:,k]

        for i in range(S0):
            for j in range(S1):
                
                vals = get_vals_voisins(np.array([i,j]),X_courant)
                vois = Vois[i,j,:]

                if k == 0:
                    energie = Beta[i,j,:] * psi( X_courant[i,j], vals,type_psi) * (vois >=0)
                    #somme_courant = np.sum( Beta[i,j]*psi( X_courant[i,j], vals,type_psi))
                    proba_courant[i,j] = np.exp(-(energie.sum()))

                energie = Beta[i,j,:] * psi( X_new[i,j], vals,type_psi) * (vois >=0)
                #somme_new = np.sum( beta * psi( X_new[i,j], vals,type_psi))
                proba_new = np.exp(-(energie.sum()))

                q = proba_new / proba_courant[i,j]
                if q > 1:
                    X_courant[i,j] = X_new[i,j]
                    proba_courant[i,j] = proba_new
                    num_ecart[k] +=1
#                else:
#                    r = np.random.uniform()
#                    if r <= q :
#                        X_courant[i,j] = x_new
#                        num_ecart[k] +=1
                #if proba < 0.95:
                #   X_courant[i,j] =np.mean(vals)# np.random.uniform(low = xmin, high = xmax)
                #   num_ecart[k] +=1
                    
        # Etape de renormalisation - conservation de l'energie
        #energie_courante = np.sqrt((X_courant**2).sum())
        #X_courant = X_courant * energie_totale/energie_courante#/ (X_courant.max() - X_courant.min()) 
        X_res[:,:,k] = X_courant        
                                  
                    
    return X_courant, X_res

#
#def gen_champs_cond(X_init, X_cond,type_champs,S0,S1,nb_iter, xmin,xmax,type_psi='sq'):
#    """ Generation d'un champs de Gibbs conditionnellement à un autre champ a partir d'une initialisation"""
#    X_res = np.zeros(shape=(S0,S1,nb_iter))
#    X_courant = np.copy(X_init)
#    num_ecart = np.zeros(shape=(nb_iter))
#    for k in range(nb_iter):
#        #t = 1 / np.log(1+(k))
#        X_new = np.random.uniform(low = xmin, high = xmax, size=(S0,S1))
#        #X_new = st.norm.rvs(X_courant)
#        #
#        for i in range(S0):
#            for j in range(S1):
#                vals, vois  = get_voisins(np.array([i,j]),X_courant)
#                if type_champs=='cache':
#                    vals_cond = X_cond[i,j]
#                elif type_champs=='couple':
#                    vals_cond = np.append(get_voisins(np.array([i,j]),X_cond),X_cond[i,j]) # plus valide !
#                
#                somme = np.sum( psi( X_courant[i,j], vals,type_psi))
#                somme_cond = np.sum( psi( X_courant[i,j], vals_cond,type_psi))
#                
#                #proba_actuel = np.exp((-somme-1./2.*somme_cond))/np.exp(-1.5*(xmax-xmin))#/np.exp(-1.5*(xmax-xmin))#*vals.size)
#                proba_actuel = np.exp(-somme-somme_cond)
#                
#                x_new = X_new[i,j]
#                somme_new = np.sum( psi( x_new, vals,type_psi))
#                somme_new_cond = np.sum( psi( x_new, vals_cond,type_psi))
#                
#                proba_new = np.exp( -somme_new - somme_new_cond)#/np.exp(-2*(xmax-xmin))
#                
#                q = proba_new / proba_actuel
#                
#                
#                if q > 1:
#                    X_courant[i,j] = x_new
#                    num_ecart[k] +=1
##                else:
##                    r = np.random.uniform()
##                    if r <= q :
##                        X_courant[i,j] = x_new
##                        num_ecart[k] +=1
#        X_res[:,:,k] = X_courant        
#                    
#                    
#    return X_courant, X_res

def get_theta_isocontour(Im):
    dx, dy = np.gradient(Im)

    theta = np.zeros_like(Im)
    
    theta[dx!=0] = np.arctan(dy[dx!=0]/dx[dx!=0])
    theta[dx==0] = np.sign(dy[dx==0]) * np.pi/2
    theta = theta + np.pi * (dx< 0)
    #theta_iso = (theta + np.pi/2)%np.pi
#    
#    theta_iso = np.zeros_like(theta)
##    
#    theta_iso[(dy>0)+(dx>0)] = theta[(dy>0)+(dx>0)]
#    theta_iso[(dy<0)+(dx<0)] = theta[(dy<0)+(dx<0)]
#    theta_iso[(dy>0)+(dx<0)] = theta[(dy>0)+(dx<0)]+np.pi/2
#    theta_iso[(dy<0)+(dx>0)] = theta[(dy<0)+(dx>0)]+np.pi/2
#    theta_iso[(dy<0)] = -theta[dy<0]
    theta_iso = theta#%np.pi
    #theta_iso[np.isnan(theta_iso)]=0
    
    return theta_iso


# parametres utiles:
S0 = 150
S1 = 150

val_min = 0
val_inter = 0.1
val_max = 1



#%%
# Generation du processus auxiliaire
U = np.zeros(shape=(S0,S1))

centre1 = np.array([S0/2,S1/2])
r1 = 5
y,x = np.ogrid[0:S0,0:S1]
cercle1 = 1-( (x-centre1[0])**2+(y-centre1[1])**2 ).astype(float)/r1**2 #< r1**2
dist1 = np.sqrt((x-centre1[0])**2+(y-centre1[1])**2 )
centre2 = np.array([50,50])
r2 = 5
cercle2 = 1-( (x-centre2[0])**2+(y-centre2[1])**2 ).astype(float) /r2**2 #< r2**2
dist2 = np.sqrt((x-centre2[0])**2+(y-centre2[1])**2 )
cercles = ((cercle1>0)) + ((cercle2>0))

#largeur = 2.5
#coeff = (centre1[1].astype(float)-centre2[1].astype(float))/(centre1[0].astype(float)-centre2[0].astype(float))
#offset = centre1[1]-coeff*centre1[0]
#droite = (y-largeur <coeff*x + offset) * (y+largeur > coeff*x + offset)
#masque = np.zeros_like(droite)
#masque[centre1[0]:centre2[1], centre1[1]:centre2[0]] = 1 # suppose centre2 > centre1
#masque*= (cercles==0)
#droite *= masque

#%%
#Initialisation
#U = U + val_max*cercles + val_max*droite

# Champ isotrope
#V_0, V_0_tout = gen_champs(U, S0,S1,20,val_min,xmax=val_max,angle=0,type_psi='sq',isotropic=True)
# Il est preferable de placer une source brillante.
#V_0 = 0.5*( (cercle1>0) + (cercle2>0) )+np.random.uniform(low = 0, high = 1, size=(S0,S1))#
V_0 = 0.5*( (cercle1>0) )+ np.random.uniform(low = 0, high = 1, size=(S0,S1))#
V_0[V_0>1] = 1
V_0[V_0<0] = 0

#%%
nb_iter_gen = 1000

V_iso, V_iso_tout = gen_champs(V_0, S0,S1,nb_iter_gen,val_min,xmax=val_max,angle=0,type_psi='sq',isotropic=True)
print 'ok'

# Orientations des isocontours
# Ce bloc a l'air interessant. mis de cote pour plus tard.

#%%
#angle = dist1.max()/np.pi * (get_theta_isocontour(cercle1)/dist1 + get_theta_isocontour(cercle2)/dist2)
angle = get_theta_isocontour(cercle1) #* (dist1<dist2) + get_theta_isocontour(cercle2) * (dist1> dist2)
#angle[dist1==dist2] = np.NAN

#angle[:] = np.pi/4#np.NAN
#angle[(droite)>0]=np.pi/4
start = time.time()
V_ani, V_ani_tout = gen_champs(V_0, S0,S1,nb_iter_gen,val_min,xmax=val_max,angle=angle,type_psi='sq',isotropic=False)
temps = time.time()-start
print temps
#
#angle = np.pi/4*np.ones_like(U)
#angle[:int(S0/2),:] = 3*np.pi/4
#angle = get_theta_isocontour(V_0)
#for champ in range(2):
#    V_ani, V_ani_tout = gen_champs(V_0, S0,S1,20,val_min,xmax=val_max,angle=angle,type_psi='sq',isotropic=False)
#    angle = get_theta_isocontour(V_ani)

#
#tan1 = np.cos(theta_isocontour)
#tan2 = np.sin(theta_isocontour)

# Notes
# - Propager les intensités par un CM isotrope : OK
# - propager les angles : un CM isotrope ne marche pas ; ou en tout cas ne 
#   preserve pas les isocontours.
# - Initialiser avec du bruit peut être bien !
# - Pour le champ anisotrope, il faudrait le rendre "rayonnant" plutôt que uni-directionnel !

#%%
# Affichages
#plt.close('all')
nb_li = 2;
nb_col = 2;


def animate1(i):
    plt.cla()
    plt.imshow(V_iso_tout[:,:,i].T, interpolation='nearest', origin='lower', cmap=plt.cm.hot); 
    


fig=plt.figure(figsize=(6*nb_col,6*nb_li))

plt.subplot(nb_li,nb_col,1)
plt.imshow(V_0.T, interpolation='nearest', origin='lower', cmap=plt.cm.afmhot,vmin=0,vmax=1); #plt.colorbar()
plt.title('Initialisation')




plt.subplot(nb_li,nb_col,2)
plt.imshow(V_iso.T, interpolation='nearest', origin='lower', cmap=plt.cm.afmhot,vmin=0,vmax=1); #plt.colorbar()
plt.title('Isotropic')

#plt.subplot(nb_li,nb_col,3)
#plt.imshow(angle.T, interpolation='nearest', origin='lower', cmap=plt.cm.jet); #plt.colorbar()
##plt.title('Angles')
#plt.colorbar()
#plt.quiver(np.cos(angle.T),np.sin(angle.T))

plt.subplot(nb_li,nb_col,3)
plt.imshow((V_ani).T, interpolation='nearest', origin='lower', cmap=plt.cm.afmhot,vmin=0,vmax=1); #plt.colorbar()

plt.title('Anisotropic')


plt.subplot(nb_li,nb_col,4)
plt.imshow((V_ani).T, interpolation='nearest', origin='lower', cmap=plt.cm.gray,vmin=0,vmax=1); #plt.colorbar()
plt.contour(V_ani.T,np.arange(0.1,1,0.1))
#plt.imshow((V_ani>V_ani.mean()).T, interpolation='nearest', origin='lower', cmap=plt.cm.afmhot,vmin=0,vmax=1); #plt.colorbar()
plt.title('Anisotropic : contour')


#plt.subplot(nb_li,nb_col,8)
#anim2 = animation.FuncAnimation(fig, animate2, frames=V_ani_tout.shape[2])



#plt.subplot(nb_li,nb_col,3)
#plt.imshow(theta_isocontour,interpolation='nearest', origin='lower', cmap=plt.cm.jet)
#plt.title('Theta - init')
#
#plt.subplot(nb_li,nb_col, 4)
#plt.imshow(V, cmap=plt.cm.Blues,interpolation='nearest', origin='lower')
#plt.quiver(V*np.cos(theta_isocontour), V*np.sin(theta_isocontour))



plt.tight_layout()

plt.savefig('current.eps', format='eps',dpi=100)
#%%%

def animate2(i):
    plt.cla()
    plt.imshow(V_ani_tout[:,:,i].T, interpolation='nearest', origin='lower', cmap=plt.cm.gray,vmin=0,vmax=1); 
    plt.contour(V_ani_tout[:,:,i].T,np.arange(0.1,1,0.1))
    plt.title('Iteration %0.f'%i)


fig2=plt.figure(figsize=(10,10))
anim = animation.FuncAnimation(fig2, animate2, frames=V_ani_tout.shape[2])


