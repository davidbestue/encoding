# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:54:20 2019

@author: David Bestue
"""

from model_functions import *
from fake_data_generator import *
from Weigths_matrix import *
from Representation import *
from process_encoding import *
from process_wm import *
from data_to_use import *
from joblib import Parallel, delayed
import multiprocessing
import time

###############################################
###############################################
############################################### Fake data
###############################################
###############################################

import random

n_trials_train=900
training_angles = np.array([ random.randint(0,359) for i in range(n_trials_train)])
training_data = fake_data(training_angles)

##
WM = Weights_matrix( training_data, training_angles )
WM_t = WM.transpose()
##

n_trials_test=2000
testing_angles = np.array([ random.randint(0,359) for i in range(n_trials_test)])
testing_data = fake_data(testing_angles)

Representation(testing_data, testing_angles, WM, WM_t, ref_angle=180, plot=True)



###############################################
###############################################
###############################################
###############################################
###############################################

#if you want to use the ones of behaviour
#enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( 's001', 'together', 'visual')
#testing_activity, testing_behaviour = process_wm_files(wm_fmri_paths, masks, wm_beh_paths, condition='1_0.2', distance='mix', sys_use='unix', nscans_wm=nscans_wm, TR=2.335)
#t, nt1, nt2 = list(testing_behaviour['T']), list(testing_behaviour['NT1']), list(testing_behaviour['NT2'])
#testing_angles=t
#testing_data = fake_data_3(t, nt1, nt2)