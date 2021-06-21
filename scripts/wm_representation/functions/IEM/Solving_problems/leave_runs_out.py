# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:24:32 2019

@author: David Bestue
"""

############# Add to sys path the path where the tools folder is
import sys, os
path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) ### same directory or one back options
sys.path.insert(1, path_tools)
from tools import *

# ############# Namefiles for the savings. 
# path_save_signal ='/home/david/Desktop/Reconstructions/IEM/IEM_trainD_testT.xlsx' 
# path_save_reconstructions = '/home/david/Desktop/Reconstructions/IEM/IEM_heatmap_trainD_testT.xlsx'

# path_save_shuffle = '/home/david/Desktop/Reconstructions/IEM/shuff_IEM_trainD_testT.xlsx'


############# Testing options
decoding_thing = 'dist_alone'  #'dist_alone'  'T_alone'  


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
Reconstructions={}
Reconstructions_shuff=[]

############# Elements for the loop
Conditions=['1_7'] #, '1_7', '2_0.2', '2_7'] 
Subjects=['d001'] #, 'n001', 'b001', 'r001', 's001', 'l001']
brain_regions = ['visual'] #, 'ips', 'pfc']
ref_angle=180



############# Analysis
#############
for Subject in Subjects:
    for Brain_region in brain_regions:
        for idx_c, Condition in enumerate(Conditions):
            print(Subject, Brain_region, Condition )
            #                    
            if Condition == cond_t:  ### Cross-validate if training and testing condition are the same (1_7 when training on target and 2_7 when training on distractor)
                #############
                ############# Get the data
                enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, 'together', Brain_region)
                #############
                ###### Process wm files (I call them activity instead of training_ or testing_ as they come from the same condition)
                activity, behaviour = preprocess_wm_data(wm_fmri_paths, masks, wm_beh_paths, 
                    condition=Condition, distance=Distance_to_use, nscans_wm=nscans_wm)
                #############
                ####### IEM cross-validating all the TRs
                #L1out=int(len(behaviour)-1) ##instead of the default 10, do the leave one out!
                Reconstruction2 = IEM_cv_all_runsout(testing_activity=activity, testing_behaviour=behaviour,
                 decode_item=decoding_thing, training_item=training_item, tr_st=tr_st, tr_end=tr_end)

                #Reconstruction = IEM_cv_all(testing_activity=activity, testing_behaviour=behaviour,
                # decode_item=decoding_thing, training_item=training_item, tr_st=tr_st, tr_end=tr_end, n_slpits=10)





# AAA = np.array(behaviour['T_alone']  )


# kf = KFold(shuffle=False, n_splits=8);
# kf.get_n_splits(AAA);
# for train_index, test_index in kf.split(AAA):
# 	train_index
    # X_train, X_test = AAA[train_index], AAA[test_index]
    # y_train, y_test = AAA[train_index], AAA[test_index]



# behs=[]

# ############# Analysis
# #############
# for Subject in Subjects:
#     for Brain_region in brain_regions:
#         for idx_c, Condition in enumerate(Conditions):
#             print(Subject, Brain_region, Condition )
#             #                    
#             if Condition == cond_t:  ### Cross-validate if training and testing condition are the same (1_7 when training on target and 2_7 when training on distractor)
#                 #############
#                 ############# Get the data
#                 enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, 'together', Brain_region)
#                 #############
#                 for p in wm_beh_paths:
#                 	behs.append(p)


# import re


# BEH = []


# for i in range(len(behs)):
# 	sub = behs[i].split('/')[-5]
# 	sess= behs[i].split('/')[-3]
# 	run = behs[i].split('/')[-2]
# 	sess2 = int(re.split('(\d+)', sess)[1])
# 	run2 = int(re.split('(\d+)', run)[1]  )
# 	BEH.append([sub, sess, sess2, run, run2])


# BEH = pd.DataFrame(BEH)
# BEH.columns=['subject', 'sess', 'sess2', 'run', 'run2']
