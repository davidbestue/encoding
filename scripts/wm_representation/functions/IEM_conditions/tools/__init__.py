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
from model_functions import f, pos_channels
import statsmodels.api as sm
from scipy.stats import zscore
import matplotlib.pyplot as plt


numcores = multiprocessing.cpu_count() 


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

###### process wm files
from ub_wind_path import *
from mask_fmri_process import *
from condition_wm import *
from wm_condition import *
from preprocess_wm_files import *

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




###### Isolated stimulus
from get_quad import *
from isolated_ones import *
from err_deg import *
from close_one import *
from all_process_condition_shuff_alone import *
from wm_condition2 import *
from preprocess_wm_files_alone import *
from all_process_condition_shuff_close import *
from preprocess_wm_files_close import *
from IEM_cross_condition_kfold_allTRs_alone import *
from IEM_cross_condition_kfold_shuff_allTRs_alone import *
















next_path = os.path.abspath(os.path.join(os.getcwd(), 'funciones')) 
sys.path.insert(1, next_path)


from a import *
from b import *
from c import *

