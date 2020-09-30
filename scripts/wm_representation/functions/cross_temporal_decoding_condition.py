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
#from leave_one_out import *
#from support_vector_machine import *
#from support_vector_machine_octaves import *
from cross_temporal_decoding import * 
from joblib import Parallel, delayed
import multiprocessing
import time
import random
#
numcores = multiprocessing.cpu_count() - 10

##paths to save the 3 files 
decoding_thing = 'Target' #'Distractor' #'Target'
Distance_to_use = 'mix'

path_save_signal ='/home/david/Desktop/Reconstructions/SVM/cross_b001_target_mix_octave_1_7.xlsx'
path_save_shuffle = '/home/david/Desktop/Reconstructions/SVM/shuff_cross_b001_target_mix_octave_1_7.xlsx'

matrixs={}
matrixs_shuff=[]

Conditions=['1_0.2', '1_7', '2_0.2', '2_7'] #
Subjects=['b001'] #, 'n001', 'd001', 'r001', 's001', 'l001'] #, 'r001', 'd001', 'b001', 's001', 'l001'
brain_regions = ['visual', 'ips', 'pfc']# 'frontinf'] #, 'ips', 'frontsup', 'frontmid', 'frontinf'

for Subject in Subjects:
    for Brain_region in brain_regions:
        for idx_c, Condition in enumerate(Conditions):
            print(Subject + ', ' + Brain_region +', ' + Condition)
            ## octaves
            enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, method, Brain_Region)
            training_activity, training_behaviour = preprocess_wm_files(wm_fmri_paths, masks, wm_beh_paths, condition='1_7', distance=distance, sys_use='unix', nscans_wm=nscans_wm, TR=2.335)
            ##
            signal_cross_temp, shuff_cross_temp = cross_tempo_SVM_shuff_condition( Subject=Subject, Brain_Region=Brain_region, Condition=Condition, 
                iterations=100, distance=Distance_to_use, decode_item=decoding_thing, training_activity=training_activity, 
                training_behaviour=training_behaviour, method='together', heatmap=False) #100
            ## quadrants
            #Reconstruction, shuff = leave1out_SVM_shuff( Subject=Subject, Brain_Region=Brain_region, Condition=Condition, iterations=100, 
            #    distance=Distance_to_use, decode_item=decoding_thing, method='together', heatmap=False) #100
            ##
            #matrixs.append(signal_cross_temp)
            matrixs[Subject + '_' + Brain_region + '_' + Condition]=signal_cross_temp
            #Reconstructions_boots.append(boots)
            matrixs_shuff.append(shuff_cross_temp)


###
# a=0
# for i, Subject in enumerate(Subjects):
#     for j, Brain_region in enumerate(brain_regions):
#         for idx_c, Condition in enumerate(Conditions):
#             matrixs[Subject + '_' + Brain_region + '_' + Condition]=matrixs[a]
#             print(a)
#             a+=1

### Save signal from the reconstructions and shuffles
writer = pd.ExcelWriter(path_save_signal)
for i in range(len(matrixs.keys())):
    matrixs[matrixs.keys()[i]].to_excel(writer, sheet_name=matrixs.keys()[i]) #each dataframe in a excel sheet

writer.save()   #save reconstructions (heatmaps)

### Save signal from the reconstructions and shuffles
# Decoding_df = pd.concat(Reconstructions, axis=0) 
# Decoding_df['label']='signal'
# Decoding_df.to_excel( path_save_signal )

# Shuffle_df = pd.concat(Reconstructions_shuff, axis=0) 
# Shuffle_df['label']='shuffle'
# Shuffle_df.to_excel( path_save_shuffle )