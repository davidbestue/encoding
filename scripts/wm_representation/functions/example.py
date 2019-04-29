# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:32:56 2019

@author: David Bestue
"""

from model_functions import *
from fake_data_generator import *
from Weigths_matrix import *
from Representation import *
from process_encoding import *
import random

numcores = multiprocessing.cpu_count()

n_trials_train=900
training_angles = np.array([ random.randint(0,359) for i in range(n_trials_train)])
training_data = fake_data(training_angles) +5

##
WM = Weights_matrix( training_data, training_angles )
WM_t = WM.transpose()
##

n_trials_test=20000
testing_angles = np.array([ random.randint(0,359) for i in range(n_trials_test)])
testing_data = fake_data(testing_angles) +10

Representation(testing_data, testing_angles, WM, WM_t, ref_angle=180)



###############################################
###############################################
###############################################
###############################################
###############################################


### Data to use
root= '/home/david/Desktop/IEM_data/'

masks = [ root+'temp_masks/n001/visual_fsign_rh.nii.gz', root+ 'temp_masks/n001/visual_fsign_lh.nii.gz']

enc_fmri_paths= [root +'n001/encoding/s01/r01/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s01/r02/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s01/r03/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s01/r04/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s01/r05/nocfmri3_Encoding_Ax.nii',
             root +'n001/encoding/s02/r01/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s02/r02/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s02/r03/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s02/r04/nocfmri3_Encoding_Ax.nii',
             root +'n001/encoding/s03/r01/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s03/r02/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s03/r03/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s03/r04/nocfmri3_Encoding_Ax.nii']

enc_beh_paths =[root +'n001/encoding/s01/r01/enc_beh.txt', root +'n001/encoding/s01/r02/enc_beh.txt', root +'n001/encoding/s01/r03/enc_beh.txt', root +'n001/encoding/s01/r04/enc_beh.txt', root +'n001/encoding/s01/r05/enc_beh.txt',
            root +'n001/encoding/s02/r01/enc_beh.txt', root +'n001/encoding/s02/r02/enc_beh.txt', root +'n001/encoding/s02/r03/enc_beh.txt', root +'n001/encoding/s02/r04/enc_beh.txt',
            root +'n001/encoding/s03/r01/enc_beh.txt', root +'n001/encoding/s03/r02/enc_beh.txt', root +'n001/encoding/s03/r03/enc_beh.txt', root +'n001/encoding/s03/r04/enc_beh.txt']


wm_fmri_paths = [root +'n001/WMtask/s01/r01/nocfmri5_task_Ax.nii', root +'n001/WMtask/s01/r02/nocfmri5_task_Ax.nii', root +'n001/WMtask/s01/r03/nocfmri5_task_Ax.nii', root +'n001/WMtask/s01/r04/nocfmri5_task_Ax.nii', root +'n001/WMtask/s01/r05/nocfmri5_task_Ax.nii',
                 root +'n001/WMtask/s02/r01/nocfmri5_task_Ax.nii', root +'n001/WMtask/s02/r02/nocfmri5_task_Ax.nii', root +'n001/WMtask/s02/r03/nocfmri5_task_Ax.nii', root +'n001/WMtask/s02/r04/nocfmri5_task_Ax.nii',
                 root +'n001/WMtask/s03/r01/nocfmri5_task_Ax.nii', root +'n001/WMtask/s03/r02/nocfmri5_task_Ax.nii', root +'n001/WMtask/s03/r03/nocfmri5_task_Ax.nii', root +'n001/WMtask/s03/r04/nocfmri5_task_Ax.nii', root +'n001/WMtask/s03/r05/nocfmri5_task_Ax.nii']


wm_beh_paths=[root +'n001/WMtask/s01/r01/wm_beh.txt', root +'n001/WMtask/s01/r02/wm_beh.txt', root +'n001/WMtask/s01/r03/wm_beh.txt', root +'n001/WMtask/s01/r04/wm_beh.txt', root +'n001/WMtask/s01/r05/wm_beh.txt',
              root +'n001/WMtask/s02/r01/wm_beh.txt', root +'n001/WMtask/s02/r02/wm_beh.txt', root +'n001/WMtask/s02/r03/wm_beh.txt', root +'n001/WMtask/s02/r04/wm_beh.txt',
              root +'n001/WMtask/s03/r01/wm_beh.txt', root +'n001/WMtask/s03/r02/wm_beh.txt', root +'n001/WMtask/s03/r03/wm_beh.txt', root +'n001/WMtask/s03/r04/wm_beh.txt', root +'n001/WMtask/s03/r05/wm_beh.txt']



##### Process training data
training_dataset, training_targets = process_encoding_files(enc_fmri_paths, masks, enc_beh_paths, sys_use='unix', hd=6, TR=2.335)

### Train your weigths
WM = Weights_matrix( training_dataset, training_targets )
WM_t = WM.transpose()


### Process testing data
testing_activity, testing_behaviour = process_encoding_files(wm_fmri_paths, masks, wm_beh_paths, condition='2_7', distance='mix', sys_use='unix', nscans_wm=16, TR=2.335)
testing_angles = np.array(testing_behaviour['T'])


### Respresentation
# TR separartion

signal = testing_activity[:, 0, :]
Representation(signal, testing_angles, WM, WM_t, ref_angle=180)









