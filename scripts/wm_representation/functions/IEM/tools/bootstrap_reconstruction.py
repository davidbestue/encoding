# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:05:33 2019

@author: David
"""

import sys, os
path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 
sys.path.insert(1, path_tools)
from tools import *


def bootstrap_reconstruction(testing_activity, targets, iterations, WM, WM_t, Inter, region, condition, subject, ref_angle=180):
    Reconstructions_boots=[]
    for n_rep in range(iterations):
        time_rec_boot_start=time.time()
        indexes_boots = np.random.randint(0,len(targets), len(targets))  #bootstraped indexes for reconstruction
        ### make the reconstryctions and append them
        targets_boot = targets[indexes_boots]
        signal_boots = testing_activity[indexes_boots, :, :] 
        signal_boots_paralel =[ signal_boots[:, i, :] for i in range(nscans_wm)]
        
        Reconstructions_boot = Parallel(n_jobs = numcores)(delayed(Representation)(signal, targets_boot, WM, WM_t, ref_angle=180, plot=False, intercept=Inter)  for signal in signal_boots_paralel)    #### reconstruction standard (paralel)
        Reconstructions_boot = pd.concat(Reconstructions_boot, axis=1) #mean of all the trials
        Reconstructions_boot.columns =  [str(i * TR) for i in range(nscans_wm)] #column names
        Reconstructions_boots.append(Reconstructions_boot) #append the reconstruction (of the current iteration)
        time_rec_boot_end = time.time() #time
        time_rec_boot = time_rec_boot_end - time_rec_boot_start
        print('boot_' + str(n_rep) + ': ' +str(time_rec_boot) ) #print time of the reconstruction shuffled
        
    ### Get just the supposed target location
    df_boots=[]
    for i in range(len(Reconstructions_boots)):
        n = Reconstructions_boots[i].iloc[ref_angle*2, :] #around the ref_angle (x2 beacuse now we have 720 instead of 360)
        n = n.reset_index()
        n.columns = ['times', 'decoding']
        n['decoding'] = [sum(Reconstructions_boots[i].iloc[:, ts] * f2(ref_angle)) for ts in range(len(n))] #population vector method (scalar product)
        n['times']=n['times'].astype(float)
        n['region'] = region
        n['subject'] = subject
        n['condition'] = condition
        df_boots.append(n) #save thhis
    
    ##
    df_boots = pd.concat(df_boots)    #same shape as the decosing of the signal
    return df_boots

