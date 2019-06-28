# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:36:19 2019

@author: David
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from model_functions import *

#path_save_signal ='/home/david/Desktop/signal_LM.xlsx'
#path_save_shuffle = '/home/david/Desktop/shuff_LM.xlsx'
path_save_signal ='/home/david/Desktop/signal_LM_dist.xlsx'
path_save_shuffle = '/home/david/Desktop/shuff_LM_dist.xlsx'

Df = pd.read_excel(path_save_signal) #convert them to pd.dataframes
Df_shuff = pd.read_excel(path_save_shuffle)


df = pd.concat([Df, Df_shuff]) #concatenate the files

presentation_period= 0.35 #stim presnetation time
presentation_period_cue=  0.50 #presentation of attentional cue time
pre_stim_period= 0.5 #time between cue and stim
resp_time = 4  #time the response is active

