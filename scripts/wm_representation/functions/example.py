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
from process_wm import *
from data_to_use import *
import random
from joblib import Parallel, delayed
import multiprocessing
import time

#
numcores = multiprocessing.cpu_count()
#
#n_trials_train=900
#training_angles = np.array([ random.randint(0,359) for i in range(n_trials_train)])
#training_data = fake_data(training_angles)
#
###
#WM = Weights_matrix( training_data, training_angles )
#WM_t = WM.transpose()
###
#
#n_trials_test=20000
#testing_angles = np.array([ random.randint(0,359) for i in range(n_trials_test)])
#testing_data = fake_data(testing_angles)
#
#Representation(testing_data, testing_angles, WM, WM_t, ref_angle=180, plot=True)



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

##### Train your weigths
WM = Weights_matrix( training_dataset, training_targets )
WM_t = WM.transpose()

##### Process testing data
testing_activity, testing_behaviour = process_wm_files(wm_fmri_paths, masks, wm_beh_paths, condition='2_7', distance='mix', sys_use='unix', nscans_wm=nscans_wm, TR=2.335)
testing_angles = np.array(testing_behaviour['T'])


### Respresentation
start_repres = time.time()
numcores = multiprocessing.cpu_count()

# TR separartion
signal_paralel =[ testing_activity[:, i, :] for i in range(nscans_wm)]
Reconstructions = Parallel(n_jobs = numcores)(delayed(Representation)(signal, testing_angles, WM, WM_t, ref_angle=180, plot=False)  for signal in signal_paralel)    ####
Reconstruction = pd.concat(Reconstructions, axis=1) 
Reconstruction.columns =  [str(i * TR) for i in range(nscans_wm)]

#Plot heatmap
plt.figure()
plt.title('Heatmap decoding')
######midpoint = df.values.mean() # (df.values.max() - df.values.min()) / 2
ax = sns.heatmap(Reconstruction, yticklabels=list(Reconstruction.index), cmap="coolwarm") # cmap= viridis "jet",  "coolwarm" RdBu_r, gnuplot, YlOrRd, CMRmap  , center = midpoint
ax.plot([0.25, np.shape(Reconstruction)[1]-0.25], [posch1_to_posch2(18),posch1_to_posch2(18)], 'k--')
plt.yticks([posch1_to_posch2(4), posch1_to_posch2(13), posch1_to_posch2(22), posch1_to_posch2(31)] ,['45','135','225', '315'])
plt.ylabel('Angle')
plt.xlabel('time (s)')
plt.show(block=False)

######
######
######

end_repres = time.time()
process_recons = end_repres - start_repres
print( 'Time process reconstruction: ' +str(process_recons))








