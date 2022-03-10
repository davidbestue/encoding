# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:24:32 2019

@author: David Bestue
"""


#######
####### In this analysis:
####### I am doing the reconstruction training in the delay period and testing in each trial. No CV and No Shuffles
#######



############# Add to sys path the path where the tools folder is
import sys, os
#path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) ### same directory or one back options
path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)) ### same directory or one back options
sys.path.insert(1, path_tools)
from tools import *

############# Namefiles for the savings. 
path_save_reconst_shuffs ='/home/david/Desktop/Reconstructions/IEM/recs_shuffs_quadrantrandom_IEM_trainT_testT_wm3.npy' 

############# Testing options
decoding_thing = 'T_alone'  #'dist_alone'  'T_alone'  

############# Training options
training_item = 'T_alone'  #'dist_alone'  'T_alone' 
cond_t = '1_7'             #'1_7'  '2_7'

Distance_to_use = 'mix' #'close' 'far'
training_time= 'delay'  #'stim_p'  'delay' 'respo'
tr_st=4
tr_end=6




############# Elements for the loop
Conditions=['1_0.2', '1_7', '2_0.2', '2_7'] 
Subjects=['d001', 'n001', 'b001', 'r001', 's001', 'l001']
brain_regions = ['visual','ips', 'pfc', 'broca']
ref_angle=180


Reconstructions_ = [] ## subjects x brain regiond --> ntrials x 16 x 720 matrix

############# Analysis
#############
for Subject in Subjects:
    for Brain_region in brain_regions:
        enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, 'together', Brain_region)
        activity, behaviour = process_wm_task(wm_fmri_paths, masks, wm_beh_paths, nscans_wm=nscans_wm) 
        behaviour['Condition'] = behaviour['Condition'].replace(['1.0_0.2', '1.0_7.0', '2.0_0.2','2.0_7.0' ], ['1_0.2', '1_7', '2_0.2', '2_7'])
        behaviour['brain_region'] = Brain_region
        behaviour = get_free_quadrant(behaviour)
        ###
        ###
        print(Subject, Brain_region)
        Reconstructed_trials=[]  ## ntrials x 16 x 720 matrix
        ###
        ###
        for trial in range(len(behaviour)):
            activity_trial = activity[trial,:,:]
            beh_trial = behaviour.iloc[trial,:]
            session_trial = beh_trial.session_run 
            ###
            ### Training
            ###
            if cond_t == '1_7':
                boolean_trials_training = np.array(behaviour['delay1']==7)  *  np.array(behaviour['order']==1) *  np.array(behaviour['session_run']!=session_trial)
            elif cond_t == '2_7':
                boolean_trials_training = np.array(behaviour['delay1']==7)  *  np.array(behaviour['order']==2) *  np.array(behaviour['session_run']!=session_trial)
            #
            activity_train_model = activity[boolean_trials_training, :, :]
            activity_train_model_TRs = np.mean(activity_train_model[:, tr_st:tr_end, :], axis=1)
            behavior_train_model = behaviour[boolean_trials_training]
            training_angles = behavior_train_model[['T', 'NT1', 'NT2']].values
            #
            Weights_matrix, Interc = Weights_matrix_LM_3items(activity_train_model_TRs, training_angles)
            Weights_matrix_t = Weights_matrix.transpose()
            ###
            ### Testing
            ###
            Reconstructed_TR = [] ## 16 x 720 matrix
            #
            for TR_ in range(nscans_wm):
                activity_TR = activity_trial[TR_, :]
                free_quad = beh_trial.quadrant_free
                if free_quad==1:
                    angle_trial = random.randint(0,89)
                elif free_quad==2:
                    angle_trial = random.randint(90,179)
                elif free_quad==3:
                    angle_trial = random.randint(180,269)
                elif free_quad==4:
                    angle_trial = random.randint(270,359)
                ###
                Inverted_encoding_model = np.dot( np.dot ( np.linalg.pinv( np.dot(Weights_matrix_t, Weights_matrix ) ),  Weights_matrix_t),  activity_TR) 
                #Inverted_encoding_model_pos = Pos_IEM2(Inverted_encoding_model)
                IEM_hd = ch2vrep3(Inverted_encoding_model_pos) #36 to 720
                to_roll = int( (ref_angle - angle_trial)*(len(IEM_hd)/360) ) ## degrees to roll
                IEM_hd_aligned=np.roll(IEM_hd, to_roll) ## roll this degree   ##vector of 720
                Reconstructed_TR.append(IEM_hd_aligned)
            ##
            resconstr_trial = np.array(Reconstructed_TR)
            Reconstructed_trials.append(resconstr_trial)
        ##
        ##
        Reconstructions_.append(Reconstructed_trials)



########

final_rec = np.array(Reconstructions_)

np.save(path_save_reconst_shuffs, final_rec)




############# Options de training times, the TRs used for the training will be different 

# training_time=='delay':
# tr_st=4
# tr_end=6

# training_time=='stim_p':
# tr_st=3
# tr_end=4

# training_time=='delay':
# tr_st=4
# tr_end=6

# training_time=='respo':
#     if decoding_thing=='Target':
#         tr_st=8
#         tr_end=9
#     elif decoding_thing=='Distractor':
#         tr_st=11
#         tr_end=12