# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:05:33 2019

@author: David
"""

import sys, os
path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 
sys.path.insert(1, path_tools)
from tools import *


def condition_wm( activity, behaviour, condition, distance, zscore_=True):
    if distance=='mix':
        if condition == '1_0.2': 
            Subset = activity[  np.array(behaviour['delay1']==0.2)  *  np.array(behaviour['order']==1) , :, :]
            beh_Subset = behaviour.loc[(behaviour['delay1']==0.2) & (behaviour['order']==1)  ] 
          
        elif condition == '1_7':
            Subset = activity[  np.array(behaviour['delay1']==7)  *  np.array(behaviour['order']==1) , :, :]
            beh_Subset = behaviour.loc[(behaviour['delay1']==7) & (behaviour['order']==1)  ] 
            
        elif condition == '2_0.2':
            Subset = activity[  np.array(behaviour['delay1']==0.2)  *  np.array(behaviour['order']==2)   , :, :]
            beh_Subset = behaviour.loc[(behaviour['delay1']==0.2) & (behaviour['order']==2) ] 
          
        elif condition == '2_7':
            Subset = activity[  np.array(behaviour['delay1']==7)  *  np.array(behaviour['order']==2)  , :, :]
            beh_Subset = behaviour.loc[(behaviour['delay1']==7) & (behaviour['order']==2)  ] 
    
    
    else: ### close or far
        if distance=='close':
            distance_t = 1
        elif distance == 'far':
            distance_t = 3
        ####
        if condition == '1_0.2':
            Subset = activity[  np.array(behaviour['delay1']==0.2)  *  np.array(behaviour['order']==1) *  np.array(behaviour['type']==distance_t)  , :, :]
            beh_Subset = behaviour.loc[(behaviour['delay1']==0.2) & (behaviour['order']==1) & (behaviour['type']==distance_t) ] 
          
        elif condition == '1_7':
            Subset = activity[  np.array(behaviour['delay1']==7)  *  np.array(behaviour['order']==1) * np.array(behaviour['type']==distance_t)  , :, :]
            beh_Subset = behaviour.loc[(behaviour['delay1']==7) & (behaviour['order']==1) & (behaviour['type']==distance_t)  ] 
            
        elif condition == '2_0.2':
            Subset = activity[  np.array(behaviour['delay1']==0.2)  *  np.array(behaviour['order']==2) * np.array(behaviour['type']==distance_t)  , :, :]
            beh_Subset = behaviour.loc[(behaviour['delay1']==0.2) & (behaviour['order']==2) & (behaviour['type']==distance_t)  ] 
          
        elif condition == '2_7':
            Subset = activity[  np.array(behaviour['delay1']==7)  *  np.array(behaviour['order']==2) *  np.array(behaviour['type']==distance_t) , :, :]
            beh_Subset = behaviour.loc[(behaviour['delay1']==7) & (behaviour['order']==2) & (behaviour['type']==distance_t)  ]
    
    
    #####zscore
    ### Per voxel (problem of mixing session, not in encoding)
    ### Get just the scans_wm that I will use
    if zscore_ == True:
        n_voxels = np.shape(Subset)[2]
        nscans_wm = np.shape(Subset)[1]
        for sc_time in range(nscans_wm):
            for voxel in range(0, n_voxels):
                Subset[:, sc_time, voxel] =  np.array( stats.zscore(Subset[:, sc_time, voxel]  ) ) ;
    
    
    ####
    return Subset, beh_Subset


