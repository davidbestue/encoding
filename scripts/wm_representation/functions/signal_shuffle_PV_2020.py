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
from leave_one_out import *
from joblib import Parallel, delayed
import multiprocessing
import time
import random
#

numcores = multiprocessing.cpu_count() #- 10

##paths to save the 3 files 
decoding_thing = 'Target' #'Distractor' #'Target'
Distance_to_use = 'close'
#path_save_reconstructions = '/home/david/Desktop/all_target_mix.xlsx' 
Reconstructions={}
#path_save_signal ='/home/david/Desktop/target_close/signal_all_target_mix.xlsx'
#path_save_shuff = '/home/david/Desktop/target_close/shuff_all_target_mix.xlsx'

Reconstructions_shuff=[]

Conditions=['1_0.2']#, '1_7', '2_0.2', '2_7'] #
Subjects=['d001'] #, 'n001', 'b001', 'r001', 's001', 'l001'] #, 'r001', 'd001', 'b001', 's001', 'l001'
brain_regions = ['visual'] #, 'ips', 'pfc']# 'frontinf'] #, 'ips', 'frontsup', 'frontmid', 'frontinf'
#ref_angle=180


for Subject in Subjects:
    for Brain_region in brain_regions:
        #plt.figure()
        ### Data to use
        #enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, 'together', Brain_region)
        ##### Process training data
        #training_dataset, training_targets = process_encoding_files(enc_fmri_paths, masks, enc_beh_paths, sys_use='unix', hd=6, TR=2.335) #4
        ##### Train your weigths
        #WM, Inter = Weights_matrix_LM( training_dataset, training_targets )
        #WM_t = WM.transpose()
        for idx_c, Condition in enumerate(Conditions):
            print(Condition)
            Reconstruction, shuff = leave_one_out_shuff( Subject=Subject, Brain_Region=Brain_region, Condition=Condition, iterations=2, 
                distance=Distance_to_use, decode_item=decoding_thing, method='together', heatmap=False)

            # Reconstruction, shuff = all_process_condition_shuff( Subject=Subject, Brain_Region=Brain_region, WM=WM, WM_t=WM_t, 
            # distance=Distance_to_use, decode_item= decoding_thing, iterations=100, Inter=Inter, Condition=Condition, method='together',  heatmap=False) #100

            # Reconstructions[Subject + '_' + Brain_region + '_' + Condition]=Reconstruction
            # Reconstructions_shuff.append(shuff)




#Reconstruction, shuff = leave_one_out_shuff( Subject=Subject, 
#    Brain_Region=Brain_region, Condition=Condition, iterations=2, distance=Distance_to_use, decode_item=decoding_thing, method='together', heatmap=False)


Subject='d001'
method='together' 
Brain_Region='visual'
enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, 'together', Brain_region)
Condition='1_0.2'
Subject='d001'
distance='close'
testing_activity, testing_behaviour = preprocess_wm_files(wm_fmri_paths, masks, wm_beh_paths, condition=Condition, distance=distance, sys_use='unix', nscans_wm=nscans_wm, TR=2.335)
decode_item = 'Target'
dec_I = 'T'
testing_angles_beh = np.array(testing_behaviour[dec_I])    # A_R # T # Dist
angles_paralel= [testing_angles_beh for i in range(nscans_wm)]
signal_paralel =[ testing_activity[:, i, :] for i in range(nscans_wm)]



testing_data=signal_paralel[0]
testing_angles=angles_paralel[0]

loo = LeaveOneOut()
errors_=[]
for train_index, test_index in loo.split(testing_data):
    X_train, X_test = testing_data[train_index], testing_data[test_index]
    y_train, y_test = testing_angles[train_index], testing_angles[test_index]
    ##
    ## correr el modelo en cada uno de los sets y guardar el error en cada uno de los trials
    ## la std no la hare con estos errores, sinó con el shuffle. No necesito guardar el error en cada repetición.
    model_trained_err = model_PV(X_train, X_test, y_train, y_test)
    errors_.append(model_trained_err)