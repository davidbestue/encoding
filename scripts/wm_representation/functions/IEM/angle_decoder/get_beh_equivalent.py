# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:24:32 2019

@author: David Bestue
"""

############# Add to sys path the path where the tools folder is
import sys, os
#path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)) ### same directory or one back options
path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) ### same directory or one back options
sys.path.insert(1, path_tools)
from tools import *

############# Namefiles for the savings. 
path_save_ ='/home/david/Desktop/Reconstructions/IEM/IEM_trainT_decoded_angle_related_behaviour.xlsx' 

############# Training options
training_item = 'T_alone'  #'dist_alone'  'T_alone' 
cond_t = '1_7'             #'1_7'  '2_7'

Distance_to_use = 'mix' #'close' 'far'
training_time= 'delay'  #'stim_p'  'delay' 'respo'
tr_st=4
tr_end=6


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


############# Dictionary and List to save the files.
Reconstruction_angles=[]

############# Elements for the loop
Conditions=['1_0.2', '1_7', '2_0.2', '2_7'] 
Subjects=['d001', 'n001', 'b001', 'r001', 's001', 'l001']
brain_regions = ['visual', 'ips', 'pfc']
ref_angle=180


BEHAVIOUR = []

############# Analysis
#############
for Subject in Subjects:
    for Brain_region in brain_regions:
        for idx_c, Condition in enumerate(Conditions):
            ####
            print(Subject, Brain_region, Condition )
            ####
            if Condition == cond_t:  ### Cross-validate if training and testing condition are the same (1_7 when training on target and 2_7 when training on distractor)
                #############
                ############# Get the data
                enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, 'together', Brain_region)
                #############
                ###### Process wm files (I call them activity instead of training_ or testing_ as they come from the same condition)
                activity, behaviour = preprocess_wm_data(wm_fmri_paths, masks, wm_beh_paths, 
                    condition=Condition, distance=Distance_to_use, nscans_wm=nscans_wm)
                #############
                behaviour['new_index'] = np.arange(0, len(behaviour),1) 
                behaviour['subject']=Subject
                behaviour['brain_region']=Brain_region
                behaviour['condition']=Condition
                ############# IEM shuffle
                BEHAVIOUR.append(behaviour)
            
            else:
                #############
                ############# Get the data
                enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, 'together', Brain_region)
                ##################
                ###### Process training data
                ##################
                ###### Process testing data 
                testing_activity, testing_behaviour = preprocess_wm_data(wm_fmri_paths, masks, wm_beh_paths, 
                    condition=Condition, distance=Distance_to_use, nscans_wm=nscans_wm)
                ##################
                testing_behaviour['new_index'] = np.arange(0, len(testing_behaviour),1) 
                testing_behaviour['subject']=Subject
                testing_behaviour['brain_region']=Brain_region
                testing_behaviour['condition']=Condition
                ###### 
                BEHAVIOUR.append(testing_behaviour)


                

#####
#####

BEHAVIOUR_file = pd.concat(BEHAVIOUR)

BEHAVIOUR_file.to_excel(path_save_)

