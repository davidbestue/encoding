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
numcores = multiprocessing.cpu_count() - 8

##paths to save the 3 files 
decoding_thing = 'Target' #'Distractor' #'Target'
Distance_to_use = 'far'

path_save_signal ='/home/david/Desktop/Reconstructions/SVM/cross_target_far_delay_1_7.xlsx' #cross_b001_target_mix_octave_1_7_far.xlsx'
path_save_shuffle = '/home/david/Desktop/Reconstructions/SVM/shuff_cross_target_far_delay_1_7.xlsx'

matrixs={}
matrixs_shuff=[]

Conditions=['1_0.2', '1_7', '2_0.2', '2_7'] #
Subjects=['b001', 'n001', 'd001', 'r001', 's001', 'l001'] #, 'r001', 'd001', 'b001', 's001', 'l001'
brain_regions = ['visual', 'ips', 'pfc']# 'frontinf'] #, 'ips', 'frontsup', 'frontmid', 'frontinf'

sh_reps = 2

for Subject in Subjects:
    for Brain_region in brain_regions:
        for idx_c, Condition in enumerate(Conditions):
            print(Subject + ', ' + Brain_region +', ' + Condition)
            ## octaves, get the specific trianing before!
            enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject_analysis=Subject, Method_analysis='together', brain_region=Brain_region)
            training_activity, training_behaviour = preprocess_wm_files(wm_fmri_paths, masks, wm_beh_paths, condition='1_7', 
                distance=Distance_to_use, sys_use='unix', nscans_wm=nscans_wm, TR=2.335)
            #
            #delay_TR_cond = np.mean(training_activity[:, 2, :], axis=1) ## training_activity[:, 8, :]
            delay_TR_cond = np.mean(training_activity[:, 3:5, :], axis=1)## training_activity[:, 8, :]
            training_activity_paralel = signal_paralel_testing =[ delay_TR_cond for i in range(nscans_wm)] 
            ##
            signal_cross_temp, shuff_cross_temp = cross_tempo_SVM_shuff_condition( Subject=Subject, Brain_Region=Brain_region, Condition=Condition, 
                iterations=sh_reps, distance=Distance_to_use, decode_item=decoding_thing, signal_paralel_training=training_activity_paralel, 
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
### Save signal from the reconstructions and shuffles
writer = pd.ExcelWriter(path_save_signal)
for i in range(len(matrixs.keys())):
    matrixs[matrixs.keys()[i]].to_excel(writer, sheet_name=matrixs.keys()[i]) #each dataframe in a excel sheet

writer.save()   #save reconstructions (heatmaps)

### Save signal from the  shuffles
matrixs_shuffle={}
a=0
for i, Subject in enumerate(Subjects):
    for j, Brain_region in enumerate(brain_regions):
        for idx_c, Condition in enumerate(Conditions):
            matrixs_shuff_100 = matrixs_shuff[a]
            for rep in range(sh_reps):
                matrixs_shuffle[Subject + '_' + Brain_region + '_' + Condition + '_shuff_' + str(rep)]=matrixs_shuff_100[rep]
            #print(a)
            a+=1


sorted_keys = sorted(matrixs_shuffle.keys()) 

writer_s = pd.ExcelWriter(path_save_shuffle)
for i in range(len(sorted_keys)):
    matrixs_shuffle[sorted_keys[i]].to_excel(writer_s, sheet_name=sorted_keys[i]) #each dataframe in a excel sheet

writer_s.save()   #save reconstructions (heatmaps)

