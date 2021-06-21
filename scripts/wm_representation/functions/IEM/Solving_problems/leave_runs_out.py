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
Conditions=['1_0.2', '1_7', '2_0.2', '2_7'] 
Subjects=['d001', 'n001', 'b001', 'r001', 's001', 'l001']
brain_regions = ['visual', 'ips', 'pfc']
ref_angle=180


behs=[]

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
                behs.append(wm_beh_paths)
 