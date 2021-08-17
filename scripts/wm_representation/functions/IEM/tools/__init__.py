# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:05:33 2019

@author: David
"""

import sys
import os
import numpy as np
import pandas as pd
from nilearn.masking import apply_mask
from nitime.timeseries import TimeSeries
from nitime.analysis import FilterAnalyzer
from scipy import stats
import time
from joblib import Parallel, delayed
import multiprocessing
import random
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import Lasso
import statsmodels.api as sm
from scipy.stats import zscore
import matplotlib.pyplot as plt
import re


numcores = multiprocessing.cpu_count() 
if numcores>20:
    numcores=numcores-10
if numcores<10:
    numcores=numcores-3


TR=2.335
nscans_wm=16

###Model_functions
#Generate the positions of the channels (there will be 14)
sep_channels=10
adjusted_size_contant = 48.519

pos_channels = np.arange(sep_channels/2,360,sep_channels)
pos_channels = [round(pos_channels[i],3) for i in range(0, len(pos_channels))]

pos_channels2 = np.arange(0,360,0.5)
pos_channels2 = [round(pos_channels2[i],3) for i in range(0, len(pos_channels2))]

next_path = os.path.abspath(os.path.join(os.getcwd(), 'tools')) 
sys.path.insert(1, next_path)


###### model functions
from circ_dist import *
from f import *
from f2 import *
from ch2vrep3 import *
from ch2vrep3_int import *
from posch1_to_posch2 import *

###### data to use
from data_to_use import *


###### Weights_matrix
from Weights_matrix_LM import *


##### Representation
from trial_rep import *
from trial_rep_decode_trial_by_trial import *
from Representation import *
from Representation_heatmap import *


###### process wm files
from ub_wind_path import *
from mask_fmri_process import *
from condition_wm import *
from wm_condition import *
from preprocess_wm_files import *
from subset_training import *


###### bootsrtap functions
from shuffled_reconstruction import *
from bootstrap_reconstruction import *
from all_process_condition_shuff_boot import *
from all_process_condition_shuff import *
from IEM_cross_condition_kfold import *
from IEM_cross_condition_kfold_shuff import *
from IEM_cross_condition_kfold_allTRs import *
from IEM_cross_condition_kfold_shuff_allTRs import *
from IEM_cross_condition_l1out import *
from IEM_cross_condition_l1out_shuff import *
from IEM_cv_all import * 
from IEM_cv import * 
from IEM import * 
from IEM_shuff import * 
from IEM_cv_all_shuff import *
from IEM_cv_all_runsout import *
from IEM_cv_all_runsout_shuff import *
from IEM_all_runsout import *
from IEM_all_runsout_shuff import *
from IEM_cv_all_runsout_performance import *
from IEM_cv_all_runsout_performance_shuff import *
from IEM_all_runsout_beh_shuff import *
from IEM_all_runsout_shuff import *
from IEM_cv_all_runsout_performance_responded import *
from IEM_cv_all_runsout_performance_responded_shuff import *


###### Isolated stimulus
from get_quad import *
from isolated_one import *
from err_deg import *
from close_one import *
from all_process_condition_shuff_alone import *
from wm_condition2 import *
from preprocess_wm_files_alone import *
from all_process_condition_shuff_close import *
from preprocess_wm_files_close import *
from IEM_cross_condition_kfold_allTRs_alone import *
from IEM_cross_condition_kfold_shuff_allTRs_alone import *

from process_beh_file import *
from preprocess_wm_data import *