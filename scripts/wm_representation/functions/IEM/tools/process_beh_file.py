
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:05:33 2019

@author: David
"""

import sys, os
path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 
sys.path.insert(1, path_tools)
from tools import *


def process_beh_file(masked_data, beh_path, n_scans, condition,  distance, sys_use='unix', TR=2.335, nscans_wm=16):
    # Behaviour 
    beh_path = ub_wind_path(beh_path, system=sys_use) #change depending on windoxs/unix
    behaviour=np.genfromtxt(beh_path, skip_header=1) #open the file
    Beh = pd.DataFrame(behaviour)  #convert it to dataframe
    headers_col = ['type', 'delay1', 'delay2', 'T', 'NT1', 'NT2', 'Dist', 'Dist_NT1', 'Dist_NT2', 'distance_T_dist', 'cue', 'order',
                'orient', 'horiz_vertical', 'A_R', 'A_err', 'Abs_angle_error', 'Error_interference', 'A_DC', 'A_DC_dist', 'Q_DC', 
                'A_DF', 'A_DF_dist', 'Q_DF', 'A_DVF', 'Q_DVF', 'A_DVF_dist', 'Q_DVF_dist', 'presentation_att_cue_time', 'presentation_target_time',
                'presentation_dist_time', 'presentation_probe_time', 'R_T', 'trial_time', 'disp_time']
    Beh.columns=headers_col #add columns
    #take off the reference    
    ref_time = Beh.iloc[-1, 1] # get the reference(diff between tsat¡rt the display and start de recording)
    start_trial=Beh['presentation_att_cue_time'].iloc[0:-1]  - ref_time #take off the reference  
    Beh = Beh.iloc[0:-1, :] # behaviour is the same except the last line (reference time) 
    start_trial_hdf_scans = start_trial/TR#transform seconds to scans 
    timestamps = [  int(round(  start_trial_hdf_scans[n] ) ) for n in range(0, len(start_trial_hdf_scans) )]
    
    #adjust according to the number of scans you want (avoid having an incomplete trial)
    while timestamps[-1]> (n_scans-nscans_wm):
        timestamps=timestamps[:-1] #take off one trial form activity
        Beh = Beh.iloc[0:-1, :] #take off one trial from behaviour
    ###################
    ################### Add the columns of isolated target and distractor
    Beh = isolated_one(Beh)
    ################### Add the columns of close target and distractor
    Beh = close_one(Beh)
    ########################################################################   
    ################## Add columns with subject, session and run
    sub_ = beh_path.split('/')[-5]
    sess_long= beh_path.split('/')[-3]
    run_long= beh_path.split('/')[-2]
    sess_= int(re.split('(\d+)', sess_long)[1])
    run_= int(re.split('(\d+)', run_long)[1]  )
    Beh['subject'] = sub_
    Beh['session'] = sess_
    Beh['run'] = run_
    Beh['session_run'] = str(sess_) + '_' + str(run_)
    #append the timestands you want from this session
    n_trials = len(timestamps)
    n_voxels_wm = np.shape(masked_data)[1]
    ### Take the important TRs (from cue, the next 14 TRs)
    run_activity=np.zeros(( n_trials, nscans_wm,  n_voxels_wm   )) ## np.zeros matrix with the correct dimensions of the session
    for idx, t in enumerate(timestamps): #beginning of the trial
        for sc in range(0, nscans_wm): #each of the 14 TRs
            trial_activity = masked_data[t+sc, :]   
            run_activity[idx, sc, :] =trial_activity                    
    
    ### 
    
    Subset, beh_Subset = condition_wm( run_activity, Beh, condition, distance=distance, zscore_=False)
    
    return Subset, beh_Subset


