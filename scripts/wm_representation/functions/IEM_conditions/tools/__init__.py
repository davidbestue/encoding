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
from scipy import stats
import time
from joblib import Parallel, delayed
import multiprocessing
import random
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut



###Model_functions
#Generate the positions of the channels (there will be 14)
sep_channels=10
adjusted_size_contant = 48.519

pos_channels = np.arange(sep_channels/2,360,sep_channels)
pos_channels = [round(pos_channels[i],3) for i in range(0, len(pos_channels))]

pos_channels2 = np.arange(0,360,0.5)
pos_channels2 = [round(pos_channels2[i],3) for i in range(0, len(pos_channels2))]







nscans_wm=16



next_path = os.path.abspath(os.path.join(os.getcwd(), 'funciones')) 
sys.path.insert(1, next_path)


from a import *
from b import *
from c import *

