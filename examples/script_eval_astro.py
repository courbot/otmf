# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 10:09:45 2016

@author: courbot
"""
import numpy as np 
import matplotlib.pyplot as plt
from os import listdir
import glob
import matplotlib

plt.close('all')
#range_psnr = np.array([-15,-12,-9,-6.,-3.,0])
range_psnr=np.arange(-15,1,1.5)
facteurs = np.array([1.])
nb_fa = facteurs.size
#range_psnr = np.arange(-10,2,2)
nb_psnr = range_psnr.size
nb_level=4
tp = np.zeros(shape=(nb_psnr,nb_level,nb_fa))-np.inf
fp,tn,fn, found,missed = np.copy(tp),np.copy(tp),np.copy(tp),np.copy(tp),np.copy(tp)


synth = False

if synth:
    dirname = './results/astro_hmf/synth/'
else:
    dirname = './results/astro_hmf/udf10/'

for id_fa in range(nb_fa):
    fa = facteurs[id_fa]   
    for id_psnr in range(nb_psnr):
        
        for nb_level_x in range(1,1+nb_level):
    #        print 'Nb level x : %.0f'%nb_level_x
        #snr = -5
#            try:
                psnr=range_psnr[id_psnr]
                
                nom_can=dirname+'nb'+str(nb_level_x)+'_psnr'+str(psnr)+'_fa'+str(fa)#+'.npz'
                
                listfile=glob.glob(nom_can+'*.npz')
                numel = np.size(listfile)
#                print numel
                i = 0
                if synth:
                    X_mpm_hmf = np.zeros(shape=(128,128,numel))
                else:
                    X_mpm_hmf = np.zeros(shape=(50,50,numel))
                msk_pres=np.copy(X_mpm_hmf)
                for filename in listfile:
                    d = np.load(filename)
                    X_mpm_hmf[:,:,i], msk_pres[:,:,i] = d['X_mpm_hmf'],d['msk_pres']
                    i+=1


                true = msk_pres>0#[msk_level]
                clas = X_mpm_hmf>0#[msk_level] > 0
                level = nb_level_x-1
                tp[id_psnr,level,id_fa] = np.mean((true==1)*(clas==1))/np.mean(true==1)
                fp[id_psnr,level,id_fa] = np.mean((true==0)*(clas==1))/np.mean(true==0)
                tn[id_psnr,level,id_fa] = np.mean((true==0)*(clas==0))/np.mean(true==0)
                fn[id_psnr,level,id_fa] = np.mean((true==1)*(clas==0))/np.mean(true==1)
                
                found[id_psnr,level,id_fa] = np.mean(true==clas)
                missed[id_psnr,level,id_fa] = np.mean(true!=clas)
                
#            except:
#                print '!'
#                pass
        
       
#%%


#%%
pfa = fp/(fp+tn)
pdet = tp/(tp+fn) 

prec = np.copy(pdet)
recall = tp/(tp+fn)

sensibilite = tp/(tp+fn)

specificite  = tn/(tn+fp)

F= 2*prec*recall /(prec+recall)

#%%
#best_int = np.zeros(shape=(nb_psnr,2))
#for id_psnr in range(nb_psnr):
#    fprov = pfa[id_psnr,:,:]
##    fprov[np.isnan(fprov)]=5
#    best_int[id_psnr,:] = np.where(fprov==fprov.min())
#%%   
#m=['o:','.:','.--','-',]     
#import matplotlib
#plt.figure(figsize=(15,4.5*nb_fa))
#cmap = matplotlib.cm.get_cmap('inferno')
#for id_fa in range(nb_fa):
#    plt.subplot(nb_fa,3,3*id_fa+2)
#    
#    for l in range(nb_level):
#        plt.plot(range_psnr,pdet[:,l,id_fa],'o:',linewidth=3,color=cmap(l/float(nb_level)),label='%.2f'%(1./(l+1)))
##        for id_psnr in range(nb_psnr):
##            if id_fa==best_int[id_psnr,1] and l==best_int[id_psnr,0]:
##                plt.plot(range_psnr[id_psnr],tp[id_psnr,l,id_fa],'o',color='r',markersize=10)
##                
#    
#    plt.legend(loc='best',title='Valeur min.')
#    plt.ylim((0.5,1.01))
#    plt.grid()
#    ##plt.xlim(snraxis[0],snraxis[-1])
#    plt.xlabel('peak-SNR (dB)')
#    plt.xlim((-15,0))
#    plt.title('PDET ($\\mu$ : %.2f) = bien detecte'%facteurs[id_fa])
#                
#    plt.subplot(nb_fa,3,3*id_fa+1)
#    
#    for l in range(nb_level):
#        plt.plot(range_psnr,pfa[:,l,id_fa],'o:',linewidth=3,color=cmap(l/float(nb_level)),label='%.2f'%(1./(l+1)))
##        for id_psnr in range(nb_psnr):
##            if id_fa==best_int[id_psnr,1] and l==best_int[id_psnr,0]:
##                plt.plot(range_psnr[id_psnr],fp[id_psnr,l,id_fa],'o',color='r',markersize=10)
#
#
##    plt.legend(loc='best',title='Valeur min.')
#    plt.ylim((0.,0.25))
#    plt.grid()
#    ##plt.xlim(snraxis[0],snraxis[-1])
#    plt.xlabel('peak-SNR (dB)')
#    plt.xlim((-15,0))
#    plt.title('PFA ($\\mu$ : %.2f) = mal detecte'%facteurs[id_fa])
#    
#    
#    plt.subplot(nb_fa,3,3*id_fa+3)
#    
#    for l in range(nb_level):
#        plt.plot(range_psnr,fn[:,l,id_fa],'o:',linewidth=3,color=cmap(l/float(nb_level)),label='%.0f'%(l+1))
##        for id_psnr in range(nb_psnr):
##            if id_fa==best_int[id_psnr,1] and l==best_int[id_psnr,0]:
##                plt.plot(range_psnr[id_psnr],fp[id_psnr,l,id_fa],'o',color='r',markersize=10)
#
#
##    plt.legend(loc='best',title='Nb niveaux')
#    plt.ylim((0.,0.5))
#    plt.grid()
#    ##plt.xlim(snraxis[0],snraxis[-1])
#    plt.xlabel('peak-SNR (dB)')
#    plt.xlim((-15,0))
#    plt.title('False negative ($\\mu$ : %.2f) = non detecte'%facteurs[id_fa])
#    
#    
##ax2 = ax.twin()  # ax2 is responsible for "top" axis and "right" axis
##ax2.set_xticks([0., .5*np.pi, np.pi, 1.5*np.pi, 2*np.pi])
##ax2.set_xticklabels(["$0$", r"$\frac{1}{2}\pi$",
##                     r"$\pi$", r"$\frac{3}{2}\pi$", r"$2\pi$"])
##
##ax2.axis["right"].major_ticklabels.set_visible(False)    
#    
#    
#plt.tight_layout()

#%%
#
#
#m=['.:','o:','.--','-',]     
#import matplotlib
#plt.figure(figsize=(12,5.*nb_fa))
#cmap = matplotlib.cm.get_cmap('inferno')
#for id_fa in range(nb_fa):
#    plt.subplot(nb_fa,2,2*id_fa+1)
#    
#    for l in range(nb_level):
#        plt.plot(pfa[:,l,id_fa],pdet[:,l,id_fa],m[id_fa],linewidth=3,color=cmap(0.1+l/float(nb_level)),label='%.0f'%(l+1))
##        for id_psnr in range(nb_psnr):
##            if id_fa==best_int[id_psnr,1] and l==best_int[id_psnr,0]:
##                plt.plot(range_psnr[id_psnr],tp[id_psnr,l,id_fa],'o',color='r',markersize=10)
##                
#    
#    plt.legend(loc='upper right',title='Nb niveaux')
#    plt.ylim((0.,1))
#    plt.grid()
#    ##plt.xlim(snraxis[0],snraxis[-1])
#    plt.xlabel('PFA')
##    plt.xlim((-15,0))
#    plt.title('Intensite max pour $\\mu$ : %.2f'%facteurs[id_fa])
#                
#    

#%%

plt.figure(figsize=(9,9))
cmap = matplotlib.cm.get_cmap('inferno')

plt.subplot(2,2,1)
id_fa = 0
for l in range(nb_level):
    plt.plot(range_psnr,fp[:,l,id_fa],'o:',linewidth=3,color=cmap(l/float(nb_level)),label='%.2f'%(1./(l+1)))

plt.legend(loc='best',title='Valeur min.')
plt.grid() ; plt.xlim((-15,0)) ; plt.xlabel('peak-SNR (dB)')
plt.ylim((0,1))
plt.title('False positive')
            
plt.subplot(2,2,2)
for l in range(nb_level):
    plt.plot(range_psnr,tp[:,l,id_fa],'o:',linewidth=3,color=cmap(l/float(nb_level)),label='%.2f'%(1./(l+1)))

plt.legend(loc='best',title='Valeur min.')
plt.grid() ; plt.xlim((-15,0)) ; plt.xlabel('peak-SNR (dB)')
plt.ylim((0,1))
plt.title('True positive')

plt.subplot(2,2,3)
for l in range(nb_level):
    plt.plot(range_psnr,tn[:,l,id_fa],'o:',linewidth=3,color=cmap(l/float(nb_level)),label='%.2f'%(1./(l+1)))

plt.legend(loc='best',title='Valeur min.')
plt.grid() ; plt.xlim((-15,0)) ; plt.xlabel('peak-SNR (dB)')
plt.ylim((0,1))
plt.title('True negative')


plt.subplot(2,2,4)
for l in range(nb_level):
    plt.plot(range_psnr,fn[:,l,id_fa],'o:',linewidth=3,color=cmap(l/float(nb_level)),label='%.2f'%(1./(l+1)))

plt.legend(loc='best',title='Valeur min.')
plt.grid() ; plt.xlim((-15,0)) ; plt.xlabel('peak-SNR (dB)')
plt.ylim((0,1))
plt.title('False negative')
    
plt.tight_layout()

#%%

#
#plt.figure(figsize=(13.5,4.5))
#cmap = matplotlib.cm.get_cmap('inferno')
#
#plt.subplot(1,3,1)
#id_fa = 0
#for l in range(nb_level):
#    plt.plot(range_psnr,sensibilite[:,l,id_fa],'o:',linewidth=3,color=cmap(l/float(nb_level)),label='%.2f'%(1./(l+1)))
#
#plt.grid() ; plt.xlim((-15,0)) ; plt.xlabel('peak-SNR (dB)')
#plt.title('Sensibilite')
#            
#plt.subplot(1,3,2)
#for l in range(nb_level):
#    plt.plot(range_psnr,specificite[:,l,id_fa],'o:',linewidth=3,color=cmap(l/float(nb_level)),label='%.2f'%(1./(l+1)))
#
#plt.legend(loc='best',title='Valeur min.')
#plt.grid() ; plt.xlim((-15,0)) ; plt.xlabel('peak-SNR (dB)')
#plt.title('Specificite')
#
#
#plt.subplot(1,3,3)
#for l in range(nb_level):
#    plt.plot(range_psnr,F[:,l,id_fa],'o:',linewidth=3,color=cmap(l/float(nb_level)),label='%.2f'%(1./(l+1)))
#
#plt.legend(loc='best',title='Valeur min.')
#plt.grid() ; plt.xlim((-15,0)) ; plt.xlabel('peak-SNR (dB)')
#plt.title('F')
#%%
plt.figure(figsize=(9,9))
cmap = matplotlib.cm.get_cmap('jet')

plt.subplot(1,1,1)
id_fa = 0

plot_line = []
for p in range(range_psnr.size):
    lin = plt.plot(pfa[p,:,id_fa],pdet[p,:,id_fa],'--',linewidth=2,color=cmap(p/(float(range_psnr.size))),label='%.2f'%(range_psnr[p]))
    plot_line.append(lin)

l1 = plt.legend([l[0] for l in plot_line],[str(range_psnr[r]) for r in range(range_psnr.size)],loc='lower right',title='SNR')

cmap = matplotlib.cm.get_cmap('inferno')
plot_dash = []
for l in range(nb_level):
    lin = plt.plot(pfa[:,l,id_fa],pdet[:,l,id_fa],'o-',linewidth=3,color=cmap(0.2+l/float(nb_level+1.)),label='%.2f'%(1./(l+1)))
    plot_dash.append(lin)
    
l2 = plt.legend([l[0] for l in plot_dash],[ '1.0','0.50','0.33','0.25'],title='Intensite min.',loc='center right')
plt.gca().add_artist(l1)
plt.gca().add_artist(l2)
plt.grid() ; 
plt.xlim(-0.001,0.5)
plt.ylim(0,1)
plt.title('ROC')

plt.xlabel('PFA')
plt.ylabel('PDET')
    
plt.tight_layout()

#%%

#
