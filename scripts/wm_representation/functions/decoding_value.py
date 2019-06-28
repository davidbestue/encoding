# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:36:19 2019

@author: David
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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




##### Measure of difference to shuffle
subj_decoding=[]
for brain_region in ['visual', 'ips', 'frontsup', 'frontmid', 'frontinf']: #['visual', 'ips', 'pfc']: ['front_sup', 'front_mid', 'front_inf']
    for condition in ['1_0.2', '1_7', '2_0.2', '2_7']:        
        for subject in df.subject.unique():
            #decode_timepoint = []
            for times in df.times.unique():                
                values = df.loc[(df['label']=='shuffle') & (df['condition']==condition) & (df['region'] ==brain_region)  & (df['subject'] ==subject) & (df['times']==times), 'decoding'] ## all shuffled reconstructions
                value_decoding = df.loc[(df['label']=='signal') & (df['condition']==condition) & (df['region'] ==brain_region)  & (df['subject'] ==subject) & (df['times']==times), 'decoding'].values[0] #the real reconstruction
                #### zscore method
                for n_rep in range(len(values)):
                    prediction = value_decoding - values.iloc[n_rep]
                    subj_decoding.append([prediction, times, subject, brain_region, condition])

#

dfsn = pd.DataFrame(subj_decoding) 
dfsn.columns=['decoding', 'times', 'subject', 'region', 'condition' ] #decode compared to shuffle

dfsn
