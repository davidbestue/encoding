# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:05:33 2019

@author: David
"""
import sys, os
path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 
sys.path.insert(1, path_tools)
from tools import *


def IEM_shuff(testing_activity, testing_behaviour, decode_item, WM, WM_t, Inter, tr_st, tr_end, 
    condition, subject, region, iterations, ref_angle=180):
    ####
    ####
    #### IEM: Inverted encoding model
    #### no cv: no cross validation. Different training and testing datasets
    ####  
    #### I use this function to run the IEM between condition
    #### Notice that here we need the WM and WM transposed (weights matrix) so we can not modify the training procedure
    #### 
    ##### decode_item (decide what you try to decode)
    ##### decode_item = 'T_alone', 'T'
    ##### decode_item = 'dist_alone' , 'Dist'  
    ##### You shuffle by getting the values of the references (maximizing signal)
    ####
    #### Get the Trs with shared information and the TRs without shared information
    list_wm_scans= range(nscans_wm)  
    ####
    #### Get the angles of the decoding item
    testing_angles = np.array(testing_behaviour[decode_item])   
    ####
    ### Respresentation
    Reconstructions_shuffled=[]
    for It in range(iterations):
        testing_angles_suhff = np.array([random.choice([0, 90, 180, 270]) for i in range(len(testing_angles))]) 
        signal_paralel =[ testing_activity[:, i, :] for i in list_wm_scans ]
        Reconstructions = Parallel(n_jobs = numcores)(delayed(Representation)(signal, testing_angles_suhff, WM, WM_t, ref_angle=180, plot=False, intercept=Inter)  for signal in signal_paralel)    #### reconstruction standard (paralel)
        Reconstruction= pd.concat(Reconstructions, axis=1) #mean of the reconstructions (all trials)
        Reconstruction.columns =  [str(i * TR) for i in list_wm_scans ]    ##column names
        Reconstructions_shuffled.append(Reconstruction)
        ##
    ######
    ###### Coger solo lo que te interesa
    ### Get just the supposed target location
    df_shuffle=[]
    for i in range(len(Reconstructions_shuffled)):
        n = Reconstructions_shuffled[i].iloc[ref_angle*2, :] #around the ref_angle (x2 beacuse now we have 720 instead of 360)
        n = n.reset_index()
        n.columns = ['times', 'decoding']
        n['decoding'] = [sum(Reconstructions_shuffled[i].iloc[:, ts] * f2(ref_angle)) for ts in range(len(n))] #population vector method (scalar product)
        n['times']=n['times'].astype(float)
        n['region'] = region
        n['subject'] = subject
        n['condition'] = condition
        df_shuffle.append(n) #save thhis
    
    ##
    df_shuffle = pd.concat(df_shuffle)    #same shape as the decosing of the signal
    ####
    return df_shuffle