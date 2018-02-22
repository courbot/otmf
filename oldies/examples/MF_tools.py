# -*- coding: utf-8 -*-
"""
Quelques outils pour la gestion/simulation/segmentation d'un champ de Markov.
"""


import numpy as np 


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