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
numcores = multiprocessing.cpu_count() - 10

##paths to save the 3 files 
decoding_thing = 'Target' #'Distractor' #'Target'
Distance_to_use = 'mix'

path_save_signal ='/home/david/Desktop/leave1out/signal_all_target_mix_l1o_prueba2.xlsx'
path_save_shuffle = '/home/david/Desktop/leave1out/shuff_all_target_mix_l1o_prueba2.xlsx'

Reconstructions=[]
Reconstructions_shuff=[]

Conditions=['1_0.2']#, '1_7', '2_0.2', '2_7'] #
Subjects=['d001'] #, 'n001', 'b001', 'r001', 's001', 'l001'] #, 'r001', 'd001', 'b001', 's001', 'l001'
brain_regions = ['visual']#, 'ips', 'pfc']# 'frontinf'] #, 'ips', 'frontsup', 'frontmid', 'frontinf'

for Subject in Subjects:
    for Brain_region in brain_regions:
        for idx_c, Condition in enumerate(Conditions):
            print(Subject + ', ' + Brain_region +', ' + Condition)
            Reconstruction, shuff = leave_one_out_shuff( Subject=Subject, Brain_Region=Brain_region, Condition=Condition, iterations=2, 
                distance=Distance_to_use, decode_item=decoding_thing, method='together', heatmap=False) #100
            ##
            Reconstructions.append(Reconstruction)
            #Reconstructions_boots.append(boots)
            Reconstructions_shuff.append(shuff)


###
### Save signal from the reconstructions and shuffles
Decoding_df = pd.concat(Reconstructions, axis=0) 
Decoding_df['label']='signal'
Decoding_df.to_excel( path_save_signal )

Shuffle_df = pd.concat(Reconstructions_shuff, axis=0) 
Shuffle_df['label']='shuffle'
Shuffle_df.to_excel( path_save_shuffle )