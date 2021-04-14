
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
        pos_min_err = np.where( np.array(erros_dist)==min(erros_dist))[0][0]
        #
        target_close_one.append( targets_distractors[options_t[pos_min_err]].iloc[i] )
        distractor_close_one.append( targets_distractors[options_d[pos_min_err]].iloc[i] )
        ###
    ###
    behaviour['T_close'] = target_close_one
    behaviour['dist_close'] = distractor_close_one
    #
    return behaviour






#############################################################################################
############################################################################################# alone
#############################################################################################


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




#############################################################################################
############################################################################################# close
#############################################################################################




def all_process_condition_shuff_close( Subject, Brain_Region, WM, WM_t, Inter, Condition, iterations, distance, decode_item, method='together', heatmap=False):
    enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, method, Brain_Region)
    ##### Process testing data
    testing_activity, testing_behaviour = preprocess_wm_files_close(wm_fmri_paths, masks, wm_beh_paths, condition=Condition, distance=distance, sys_use='unix', nscans_wm=nscans_wm, TR=2.335)

    if decode_item == 'Target':
        dec_I = 'T_close'
    elif decode_item == 'Response':
        dec_I = 'A_R'
    elif decode_item == 'Distractor':
        dec_I = 'dist_close'
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






def wm_condition3(masked_data, beh_path, n_scans, condition,  distance, sys_use='unix', TR=2.335, nscans_wm=16):
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
    Beh = close_one(Beh)
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




def preprocess_wm_files_close(wm_fmri_paths, masks, wm_beh_paths, condition, distance='mix', sys_use='unix', nscans_wm=16, TR=2.335):
    ### Mask and process the fmri data
    start_process_wm = time.time()
    numcores = multiprocessing.cpu_count()
    wm_masked= Parallel(n_jobs = numcores)(delayed(mask_fmri_process)(fmri_path, masks, sys_use='unix')  for fmri_path in wm_fmri_paths)    ####
    scans_wm_runs = [len(wm_masked[r]) for r in range(len(wm_masked)) ]
    
    ### TRs of interest
    activity_beh  = Parallel(n_jobs = numcores)(delayed(wm_condition3)(masked_data, beh_path, n_scans, condition, distance=distance, sys_use='unix', TR=TR, nscans_wm=nscans_wm) for masked_data, beh_path, n_scans in zip( wm_masked, wm_beh_paths, scans_wm_runs))    ####
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





#######################################
####################################### cross validation TRs
#######################################
#######################################







def IEM_cross_condition_kfold_allTRs_alone(testing_activity, testing_behaviour, decode_item, WM, WM_t, Inter, 
    tr_st, tr_end, n_slpits=10):
    ####
    ####
    #### IEM usando data de WM test
    #### IEM de aquellos TRs donde se use tambien training data (condiciones 1_7 y 2_7)
    #### En vez de hacer leave one out, que tarda mucho, o usar el mismo data (overfitting), hago k_fold, con 10 splits. 
    ####
    ####
    if decode_item == 'Target':
        dec_I = 'T_alone'
    elif decode_item == 'Response':
        dec_I = 'A_R'
    elif decode_item == 'Distractor':
        dec_I = 'dist_alone'
    else:
        'Error specifying the decode item'
    ####
    #### Get the Trs with shared information and the TRs without shared information
    list_wm_scans= range(nscans_wm)  
    trs_shared = range(tr_st, tr_end)
    nope=[list_wm_scans.remove(tr_s) for tr_s in trs_shared]
    list_wm_scans2 = list_wm_scans
    ####
    #### Run the ones without shared information the same way
    testing_angles = np.array(testing_behaviour[dec_I])    # A_R # T # Dist
    #####
    #####
    #####
    Recons_dfs_not_shared=[]
    for not_shared in list_wm_scans2:
        training_data =   np.mean(testing_activity[:, tr_st:tr_end, :], axis=1) ## son los mismos siempre, pero puede haber time dependence!
        testing_data= testing_activity[:, not_shared, :]   
        reconstrction_sh=[]
        kf = KFold(n_splits=n_slpits);
        kf.get_n_splits(testing_data);
        for train_index, test_index in kf.split(testing_data):
            X_train, X_test = training_data[train_index], testing_data[test_index]
            y_train, y_test = testing_angles[train_index], testing_angles[test_index]
            ## train
            WM2, Inter2 = Weights_matrix_LM(X_train, y_train)
            WM_t2 = WM2.transpose()
            ## test
            rep_x = Representation(testing_data=X_test, testing_angles=y_test, Weights=WM2, Weights_t=WM_t2, ref_angle=180, plot=False, intercept=Inter2)
            reconstrction_sh.append(rep_x)
        ###
        reconstrction_sh = pd.concat(reconstrction_sh, axis=1) ##una al lado de la otra, de lo mismo, ahora un mean manteniendo indice
        reconstrction_sh_mean = reconstrction_sh.mean(axis = 1) #solo queda una columna con el mean de cada channel 
        Recons_dfs_not_shared.append(reconstrction_sh_mean)
    ####
    Reconstruction_not_shared = pd.concat(Recons_dfs_not_shared, axis=1)
    Reconstruction_not_shared.columns =  [str(i * TR) for i in list_wm_scans2 ] 
    ####
    #### Run the ones with shared information: k fold
    Recons_dfs_shared=[]
    for shared_TR in trs_shared:
        testing_data= testing_activity[:, shared_TR, :]            
        reconstrction_sh=[]
        kf = KFold(n_splits=n_slpits);
        kf.get_n_splits(testing_data);
        for train_index, test_index in kf.split(testing_data):
            X_train, X_test = testing_data[train_index], testing_data[test_index]
            y_train, y_test = testing_angles[train_index], testing_angles[test_index]
            ## train
            WM2, Inter2 = Weights_matrix_LM(X_train, y_train)
            WM_t2 = WM2.transpose()
            ## test
            rep_x = Representation(testing_data=X_test, testing_angles=y_test, Weights=WM2, Weights_t=WM_t2, ref_angle=180, plot=False, intercept=Inter2)
            reconstrction_sh.append(rep_x)
        ###
        reconstrction_sh = pd.concat(reconstrction_sh, axis=1) ##una al lado de la otra, de lo mismo, ahora un mean manteniendo indice
        reconstrction_sh_mean = reconstrction_sh.mean(axis = 1) #solo queda una columna con el mean de cada channel 
        Recons_dfs_shared.append(reconstrction_sh_mean)
    ####
    Reconstruction_shared = pd.concat(Recons_dfs_shared, axis=1)
    Reconstruction_shared.columns =  [str(i * TR) for i in trs_shared ]  
    #### 
    #### Merge both recosntructions dfs to get a single one
    Reconstruction = pd.concat([Reconstruction_not_shared, Reconstruction_shared], axis=1)
    ### sort the columns so the indep does not get at the end
    sorted_col = np.sort([float(Reconstruction.columns[i]) for i in range(len(Reconstruction.columns))])           
    sorted_col = [str(sorted_col[i]) for i in range(len(sorted_col))]
    Reconstruction = Reconstruction.reindex( sorted_col, axis=1)  
    #
    return Reconstruction








def IEM_cross_condition_kfold_shuff_allTRs_alone(testing_activity, testing_behaviour, decode_item, WM, WM_t, Inter, condition, subject, region,
    iterations, tr_st, tr_end, ref_angle=180, n_slpits=10):
    ####
    ####
    #### IEM usando data de WM test
    #### IEM de aquellos TRs donde se use tambien training data (condiciones 1_7 y 2_7)
    #### En vez de hacer leave one out, que tarda mucho, o usar el mismo data (overfitting), hago k_fold, con 10 splits. 
    #### Pongo el shuffle al principio segun el numero de iterations
    ####
    ####
    if decode_item == 'Target':
        dec_I = 'T_alone'
    elif decode_item == 'Response':
        dec_I = 'A_R'
    elif decode_item == 'Distractor':
        dec_I = 'dist_alone'
    else:
        'Error specifying the decode item'
    ####
    #### Get the Trs with shared information and the TRs without shared information
    list_wm_scans= range(nscans_wm)  
    trs_shared = range(tr_st, tr_end)
    nope=[list_wm_scans.remove(tr_s) for tr_s in trs_shared]
    list_wm_scans2 = list_wm_scans
    ####
    #### Run the ones without shared information the same way
    testing_angles = np.array(testing_behaviour[dec_I])    # A_R # T # Dist
    Reconstructions_shuffled=[]
    for It in range(iterations):
        testing_angles_suhff = np.array([random.choice([0, 90, 180, 270]) for i in range(len(testing_angles))]) 
        Recons_dfs_not_shared=[]
        for not_shared in list_wm_scans2:
            training_data =   np.mean(testing_activity[:, tr_st:tr_end, :], axis=1) ## son los mismos siempre, pero puede haber time dependence!
            testing_data= testing_activity[:, not_shared, :]   
            reconstrction_sh=[]
            kf = KFold(n_splits=n_slpits);
            kf.get_n_splits(testing_data);
            for train_index, test_index in kf.split(testing_data):
                X_train, X_test = training_data[train_index], testing_data[test_index]
                y_train, y_test = testing_angles[train_index], testing_angles[test_index]
                ## train
                WM2, Inter2 = Weights_matrix_LM(X_train, y_train)
                WM_t2 = WM2.transpose()
                ## test
                ## do the suffle here!
                y_test = np.array([random.choice([0, 90, 180, 270]) for i in range(len(y_test))]) 
                rep_x = Representation(testing_data=X_test, testing_angles=y_test, Weights=WM2, Weights_t=WM_t2, ref_angle=180, plot=False, intercept=Inter2)
                reconstrction_sh.append(rep_x)
            ###
            reconstrction_sh = pd.concat(reconstrction_sh, axis=1) ##una al lado de la otra, de lo mismo, ahora un mean manteniendo indice
            reconstrction_sh_mean = reconstrction_sh.mean(axis = 1) #solo queda una columna con el mean de cada channel 
            Recons_dfs_not_shared.append(reconstrction_sh_mean)
        ####
        Reconstruction_not_shared = pd.concat(Recons_dfs_not_shared, axis=1)
        Reconstruction_not_shared.columns =  [str(i * TR) for i in list_wm_scans2 ] 
        ###
        #### Run the ones with shared information: k fold
        Recons_dfs_shared=[]
        for shared_TR in trs_shared:
            testing_data= testing_activity[:, shared_TR, :] 
            reconstrction_sh=[]
            kf = KFold(n_splits=n_slpits, shuffle=True)
            kf.get_n_splits(testing_data)
            for train_index, test_index in kf.split(testing_data):
                X_train, X_test = testing_data[train_index], testing_data[test_index]
                y_train, y_test = testing_angles[train_index], testing_angles[test_index] ##aqui no mezclas, ya que antes WM t WM_t no estanba trained en shuffled data
                ## train
                WM2, Inter2 = Weights_matrix_LM(X_train, y_train);
                WM_t2 = WM2.transpose();
                ## do the suffle here!
                y_test = np.array([random.choice([0, 90, 180, 270]) for i in range(len(y_test))]) 
                ## test
                rep_x = Representation(testing_data=X_test, testing_angles=y_test, Weights=WM2, Weights_t=WM_t2, ref_angle=180, plot=False, intercept=Inter2)
                reconstrction_sh.append(rep_x)
            ###
            reconstrction_sh = pd.concat(reconstrction_sh, axis=1) ##una al lado de la otra, de lo mismo, ahora un mean manteniendo indice
            reconstrction_sh_mean = reconstrction_sh.mean(axis = 1) #solo queda una columna con el mean de cada channel 
            Recons_dfs_shared.append(reconstrction_sh_mean)
        ####
        Reconstruction_shared = pd.concat(Recons_dfs_shared, axis=1)
        Reconstruction_shared.columns =  [str(i * TR) for i in trs_shared ]   
        #### 
        #### Merge both recosntructions dfs to get a single one
        Reconstruction = pd.concat([Reconstruction_not_shared, Reconstruction_shared], axis=1)
        ### sort the columns so the indep does not get at the end
        sorted_col = np.sort([float(Reconstruction.columns[i]) for i in range(len(Reconstruction.columns))])           
        sorted_col = [str(sorted_col[i]) for i in range(len(sorted_col))]
        Reconstruction = Reconstruction.reindex( sorted_col, axis=1)  
        #      
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
    return df_shuffle














# targets_distractors = behaviour[['T', 'NT1', 'NT2', 'Dist', 'Dist_NT1', 'Dist_NT2']]
# ###
# target_close_one=[]
# distractor_close_one=[]
#     ###
# for i in range(len(targets_distractors)):
#     err1 = abs(err_deg(targets_distractors['T'].iloc[i], targets_distractors['Dist'].iloc[i]))
#     err2 =abs(err_deg(targets_distractors['NT1'].iloc[i], targets_distractors['Dist_NT1'].iloc[i]))
#     err3 =abs(err_deg(targets_distractors['NT2'].iloc[i], targets_distractors['Dist_NT2'].iloc[i]))
#     ############
#     options_t = ['T', 'NT1', 'NT2']
#     options_d = ['Dist', 'Dist_NT1', 'Dist_NT2']
#     #
#     erros_dist = [err1, err2, err3]
#     pos_min_err = np.where(np.array(erros_dist)==min(erros_dist))[0][0]
#     #
#     target_close_one.append( targets_distractors[options_t[pos_min_err]].iloc[i] )
#     distractor_close_one.append( targets_distractors[options_d[pos_min_err]].iloc[i] )
#     ###

# ###
# behaviour['T_close'] = target_close_one
# behaviour['dist_close'] = distractor_close_one
# #
# return behaviour





# behaviour=np.genfromtxt(beh_path, skip_header=1) #open the file
# Beh = pd.DataFrame(behaviour)  #convert it to dataframe
# headers_col = ['type', 'delay1', 'delay2', 'T', 'NT1', 'NT2', 'Dist', 'Dist_NT1', 'Dist_NT2', 'distance_T_dist', 'cue', 'order',
#             'orient', 'horiz_vertical', 'A_R', 'A_err', 'Abs_angle_error', 'Error_interference', 'A_DC', 'A_DC_dist', 'Q_DC', 
#             'A_DF', 'A_DF_dist', 'Q_DF', 'A_DVF', 'Q_DVF', 'A_DVF_dist', 'Q_DVF_dist', 'presentation_att_cue_time', 'presentation_target_time',
#             'presentation_dist_time', 'presentation_probe_time', 'R_T', 'trial_time', 'disp_time']
# Beh.columns=headers_col #add columns
# #take off the reference    
# ref_time = Beh.iloc[-1, 1] # get the reference(diff between tsat¡rt the display and start de recording)
# start_trial=Beh['presentation_att_cue_time'].iloc[0:-1]  - ref_time #take off the reference  
# Beh = Beh.iloc[0:-1, :] # behaviour is the same except the last line (reference time) 
# start_trial_hdf_scans = start_trial/TR#transform seconds to scans 
# timestamps = [  int(round(  start_trial_hdf_scans[n] ) ) for n in range(0, len(start_trial_hdf_scans) )]

