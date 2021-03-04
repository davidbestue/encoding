
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:24:32 2019

@author: David Bestue
"""

from model_functions import *
from fake_data_generator import *
from Weights_matrixs import *
from Representation import *
from process_encoding import *
from process_wm import *
from data_to_use import *
from bootstrap_functions import *
from joblib import Parallel, delayed
import multiprocessing
import time
import random
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut


numcores = multiprocessing.cpu_count() 


def get_quad(degree):
    if degree <= 90:
        angle = 1
    else:
        if degree <= 180:
            angle = 2
        else:
            if degree <= 270:
                angle = 3
            else:
                if degree < 360:
                    angle = 4
    ###
    return angle





def isolated_one(behaviour):
    targets_distractors = behaviour[['T', 'NT1', 'NT2', 'Dist', 'Dist_NT1', 'Dist_NT2']]
    ###
    targets_distractors['q_t1'] = [get_quad(targets_distractors.iloc[i]['T']) for i in range(len(targets_distractors))]
    targets_distractors['q_nt1'] = [get_quad(targets_distractors.iloc[i]['NT1']) for i in range(len(targets_distractors))]
    targets_distractors['q_nt2'] = [get_quad(targets_distractors.iloc[i]['NT2']) for i in range(len(targets_distractors))]
    targets_distractors['q_dist'] = [get_quad(targets_distractors.iloc[i]['Dist']) for i in range(len(targets_distractors))]
    targets_distractors['q_dist1'] = [get_quad(targets_distractors.iloc[i]['Dist_NT1']) for i in range(len(targets_distractors))]
    targets_distractors['q_dist2'] = [get_quad(targets_distractors.iloc[i]['Dist_NT2']) for i in range(len(targets_distractors))]
    ##
    targets_alone_quadrant=[]
    distractor_alone_quadrant=[]
    ###
    for i in range(len(targets_distractors)):
        targets_quadrants = [targets_distractors['q_t1'].iloc[i], targets_distractors['q_nt1'].iloc[i], targets_distractors['q_nt2'].iloc[i]]                                                                                 
        distractors_quadrants = [targets_distractors['q_dist'].iloc[i], targets_distractors['q_dist1'].iloc[i], targets_distractors['q_dist2'].iloc[i]] 
        ##################
        ################## get target alone
        ##################
        if targets_quadrants[0] not in distractors_quadrants:
            targets_alone_quadrant.append(targets_distractors.iloc[i]['T'])
        elif targets_quadrants[1] not in distractors_quadrants:
            targets_alone_quadrant.append(targets_distractors.iloc[i]['NT1'])
        elif targets_quadrants[2] not in distractors_quadrants:
            targets_alone_quadrant.append(targets_distractors.iloc[i]['NT2'])
        else:
            print('Error distribution stimuli')
        ##################
        ################## get distractor alone
        ##################
        if distractors_quadrants[0] not in targets_quadrants:
            distractor_alone_quadrant.append(targets_distractors.iloc[i]['Dist'])
        elif distractors_quadrants[1] not in targets_quadrants:
            distractor_alone_quadrant.append(targets_distractors.iloc[i]['Dist_NT1'])
        elif distractors_quadrants[2] not in targets_quadrants:
            distractor_alone_quadrant.append(targets_distractors.iloc[i]['Dist_NT2'])
        else:
            print('Error distribution stimuli')
    ###
    ###
    behaviour['T_alone'] = targets_alone_quadrant
    behaviour['dist_alone'] = distractor_alone_quadrant
    #
    return behaviour


###df = pd.read_excel('C:\\Users\\David\Desktop\\KI_Desktop\\data_reconstructions\\IEM\\example2_beh.xlsx') 
##df2 = isolated_one(df)


def err_deg(a1,ref):
    ### Calculate the error ref-a1 in an efficient way in the circular space
    ### it uses complex numbers!
    ### Input in degrees (0-360)
    a1=np.radians(a1)
    ref=np.radians(ref)
    err = np.angle(np.exp(1j*ref)/np.exp(1j*(a1) ), deg=True) 
    err=round(err, 2)
    return err



def close_one(behaviour):
    targets_distractors = behaviour[['T', 'NT1', 'NT2', 'Dist', 'Dist_NT1', 'Dist_NT2']]
    ###
    target_close_one=[]
    distractor_close_one=[]
    ###
    for i in range(len(targets_distractors)):
        err1 = abs(err_deg(targets_distractors['T'].iloc[i], targets_distractors['Dist'].iloc[i]))
        err2 =abs(err_deg(targets_distractors['NT1'].iloc[i], targets_distractors['Dist_NT1'].iloc[i]))
        err3 =abs(err_deg(targets_distractors['NT2'].iloc[i], targets_distractors['Dist_NT2'].iloc[i]))
        ############
        options_t = ['T', 'NT1', 'NT2']
        options_d = ['Dist', 'Dist_NT1', 'Dist_NT2']
        #
        erros_dist = [err1, err2, err3]
        pos_min_err = np.where(erros_dist==min(erros_dist))[0][0]
        #
        target_close_one.append( targets_distractors[options_t[pos_min_err]].iloc[i] )
        distractor_close_one.append( targets_distractors[options_d[pos_min_err]].iloc[i] )
        ###
    ###
    behaviour['T_close'] = target_close_one
    behaviour['dist_close'] = distractor_close_one
    #
    return behaviour














def all_process_condition_shuff_alone( Subject, Brain_Region, WM, WM_t, Inter, Condition, iterations, distance, decode_item, method='together', heatmap=False):
    enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, method, Brain_Region)
    ##### Process testing data
    testing_activity, testing_behaviour = preprocess_wm_files_alone(wm_fmri_paths, masks, wm_beh_paths, condition=Condition, distance=distance, sys_use='unix', nscans_wm=nscans_wm, TR=2.335)

    if decode_item == 'Target':
        dec_I = 'T_alone'
    elif decode_item == 'Response':
        dec_I = 'A_R'
    elif decode_item == 'Distractor':
        dec_I = 'dist_alone'
    else:
        'Error specifying the decode item'

    #
    testing_angles = np.array(testing_behaviour[dec_I])    # A_R # T # Dist
    ### Respresentation
    start_repres = time.time()    
    # TR separartion
    signal_paralel =[ testing_activity[:, i, :] for i in range(nscans_wm)]
    Reconstructions = Parallel(n_jobs = numcores)(delayed(Representation)(signal, testing_angles, WM, WM_t, ref_angle=180, plot=False, intercept=Inter)  for signal in signal_paralel)    #### reconstruction standard (paralel)
    Reconstruction = pd.concat(Reconstructions, axis=1) #mean of the reconstructions (all trials)
    Reconstruction.columns =  [str(i * TR) for i in range(nscans_wm)]    ##column names
    #Plot heatmap
    if heatmap==True:
        plt.figure()
        plt.title(Condition)
        ax = sns.heatmap(Reconstruction, yticklabels=list(Reconstruction.index), cmap="coolwarm") # cmap= viridis "jet",  "coolwarm" RdBu_r, gnuplot, YlOrRd, CMRmap  , center = midpoint
        ax.plot([0.25, np.shape(Reconstruction)[1]-0.25], [posch1_to_posch2(18),posch1_to_posch2(18)], 'k--')
        plt.yticks([posch1_to_posch2(4), posch1_to_posch2(13), posch1_to_posch2(22), posch1_to_posch2(31)] ,['45','135','225', '315'])
        plt.ylabel('Angle')
        plt.xlabel('time (s)')
        plt.tight_layout()
        plt.show(block=False)
    
    ######
    ######
    ######
    end_repres = time.time()
    process_recons = end_repres - start_repres
    print( 'Time process reconstruction: ' +str(process_recons)) #print time of the process
    
    #df_boots = bootstrap_reconstruction(testing_activity, testing_angles, iterations, WM, WM_t, Inter, Brain_Region, Condition, Subject, ref_angle=180)    
    ####### Shuff
    #### Compute the shuffleing
    shuffled_rec = shuffled_reconstruction(signal_paralel, testing_angles, iterations, WM, WM_t, Inter=Inter, region=Brain_Region, condition=Condition, subject=Subject, ref_angle=180)
    
    return Reconstruction, shuffled_rec






def wm_condition2(masked_data, beh_path, n_scans, condition,  distance, sys_use='unix', TR=2.335, nscans_wm=16):
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
    ########################################################################    
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





def preprocess_wm_files_alone(wm_fmri_paths, masks, wm_beh_paths, condition, distance='mix', sys_use='unix', nscans_wm=16, TR=2.335):
    ### Mask and process the fmri data
    start_process_wm = time.time()
    numcores = multiprocessing.cpu_count()
    wm_masked= Parallel(n_jobs = numcores)(delayed(mask_fmri_process)(fmri_path, masks, sys_use='unix')  for fmri_path in wm_fmri_paths)    ####
    scans_wm_runs = [len(wm_masked[r]) for r in range(len(wm_masked)) ]
    
    ### TRs of interest
    activity_beh  = Parallel(n_jobs = numcores)(delayed(wm_condition2)(masked_data, beh_path, n_scans, condition, distance=distance, sys_use='unix', TR=TR, nscans_wm=nscans_wm) for masked_data, beh_path, n_scans in zip( wm_masked, wm_beh_paths, scans_wm_runs))    ####
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

###
