# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:05:33 2019

@author: David
"""

import sys, os
path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 
sys.path.insert(1, path_tools)
from tools import *


def preprocess_wm_data(wm_fmri_paths, masks, wm_beh_paths, condition, 
    distance='mix', sys_use='unix', nscans_wm=16, TR=2.335):
    #
    ### Mask and process the fmri data
    #
    start_process_wm = time.time()
    numcores = multiprocessing.cpu_count()
    wm_masked= Parallel(n_jobs = numcores)(delayed(mask_fmri_process)(fmri_path, masks, sys_use='unix')  for fmri_path in wm_fmri_paths)    ####
    scans_wm_runs = [len(wm_masked[r]) for r in range(len(wm_masked)) ]
    
    ### TRs of interest
    ### behaviour with all the extra columns (closest, isolated) --> process_beh_file
    activity_beh  = Parallel(n_jobs = numcores)(delayed(process_beh_file)(masked_data, beh_path, n_scans, condition, distance=distance, sys_use='unix', TR=TR, nscans_wm=nscans_wm) for masked_data, beh_path, n_scans in zip( wm_masked, wm_beh_paths, scans_wm_runs))    ####
    runs_signal = [activity_beh[i][0] for i in range(len(activity_beh))]
    runs_beh = [activity_beh[i][1] for i in range(len(activity_beh))]
    
    ## concatenate the runs
    testing_activity = np.vstack(runs_signal)
    testing_behaviour = pd.concat(runs_beh)
    ##
    end_process_wm = time.time()
    process_wm = end_process_wm - start_process_wm
    print( 'Time process wm: ' +str(process_wm))
    return testing_activity, testing_behaviour
