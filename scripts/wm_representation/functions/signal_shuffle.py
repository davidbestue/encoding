# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:45:04 2019

@author: David Bestue
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from model_functions import *

#path_save_signal ='/home/david/Desktop/signal_LMaxpv_n001.xlsx'
#path_save_shuffle = '/home/david/Desktop/shuff_LMaxpv_n001.xlsx'

path_save_signal ='/home/david/Desktop/signal_LM.xlsx'
path_save_shuffle = '/home/david/Desktop/shuff_LM.xlsx'

Df = pd.read_excel(path_save_signal)
Df_shuff = pd.read_excel(path_save_shuffle)


df = pd.concat([Df, Df_shuff])


presentation_period= 0.35 
presentation_period_cue=  0.50
inter_trial_period= 0.1 
pre_cue_period= 0.5 
pre_stim_period= 0.5 
limit_time=5 
ref_angle=45


##### Measure of difference to shuffle
subj_decoding=[]
for brain_region in ['visual', 'ips', 'frontsup', 'frontmid', 'frontinf']: #['visual', 'ips', 'pfc']: ['front_sup', 'front_mid', 'front_inf']
    for condition in ['1_0.2', '1_7', '2_0.2', '2_7']:        
        for subject in df.subject.unique():
            decode_timepoint = []
            for times in df.times.unique():                
                values = df.loc[(df['label']=='shuffle') & (df['condition']==condition) & (df['region'] ==brain_region)  & (df['subject'] ==subject) & (df['times']==times), 'decoding']
                value_decoding = df.loc[(df['label']=='signal') & (df['condition']==condition) & (df['region'] ==brain_region)  & (df['subject'] ==subject) & (df['times']==times), 'decoding'].values[0]
                #### zscore method
                prediction = ( value_decoding - np.mean(values) )/ np.std(values)
                #### non parametric kernel method
                #predict = statsmodels.nonparametric.kernel_density.KDEMultivariate(values, var_type='c', bw='cv_ls')
                #prediction = 1 / predict.pdf([ value_decoding])
                decode_timepoint.append(prediction)
            ####
            decode_timepoint = pd.DataFrame(decode_timepoint)
            decode_timepoint.columns=['decoding']
            decode_timepoint['times'] = df.times.unique()
            decode_timepoint['subject'] = subject
            decode_timepoint['region'] = brain_region
            decode_timepoint['condition'] = condition
            subj_decoding.append( decode_timepoint)



dfsn = pd.concat(subj_decoding)

fig = plt.figure(figsize=(10,8))
for indx_c, condition in enumerate(['1_0.2', '1_7', '2_0.2', '2_7']):
    
    if condition == '1_0.2':
        delay1 = 0.2
        delay2 = 11.8
        cue=0
        t_p = cue + presentation_period_cue + pre_stim_period 
        d_p = t_p + presentation_period +delay1 
        r_t = d_p + presentation_period + delay2
        xlim = [0, 27]
        
    elif condition == '1_7':
        delay1 = 7
        delay2 = 5
        cue=0
        t_p = cue + presentation_period_cue + pre_stim_period 
        d_p = t_p + presentation_period +delay1 
        r_t = d_p + presentation_period + delay2
        xlim = [0, 27]
        
    elif condition == '2_0.2':
        delay1 = 0.2
        delay2 = 12
        cue=0
        d_p = cue + presentation_period_cue + pre_stim_period 
        t_p = d_p + presentation_period +delay1 
        r_t = t_p + presentation_period + delay2   
        xlim = [0, 27]
        
    elif condition == '2_7':
        delay1 = 7
        delay2 = 12
        cue=0
        d_p = cue + presentation_period_cue + pre_stim_period 
        t_p = d_p + presentation_period +delay1 
        r_t = t_p + presentation_period + delay2
        xlim = [0, 35]
        
    
    start_hrf = 4
    sec_hdrf = 3
    resp_time = 4    
    d_p1 = (start_hrf + d_p) 
    t_p1 = (start_hrf +t_p)
    r_t1=  (start_hrf + r_t)
    #
    d_p2 = d_p1 + sec_hdrf 
    t_p2 = t_p1 + sec_hdrf
    r_t2=  r_t1 + sec_hdrf + resp_time
    
    y_vl_min = -5 #df_all_by_subj.Decoding.min()
    y_vl_max = 5 #◙df_all_by_subj.Decoding.max()
    
    #fig = plt.figure()
    ax = fig.add_subplot(2,2, indx_c+1) 
    sns.lineplot( ax=ax, x="times", y="decoding", hue='region', hue_order =  ['visual', 'ips', 'frontsup', 'frontmid', 'frontinf'],  ci=69,  data=dfsn.loc[ (dfsn['condition']==condition)]) #, 'frontmid', 'frontinf'
    
    plt.plot([0, 35], [0,0], 'k--')   
    plt.fill_between(  [ t_p1, t_p2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='b', alpha=0.3) #, label='target'  )
    plt.fill_between(  [ d_p1, d_p2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='g', alpha=0.3) #, label='distractor'  )
    plt.fill_between(  [ r_t1, r_t2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='y', alpha=0.3) #, label='response'  )     
    TITLE_BR = condition 
    plt.title(TITLE_BR)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().get_xaxis().tick_bottom()
    plt.gca().get_yaxis().tick_left()
    plt.xticks([5,15,25])
    plt.yticks([-4, 0 , 4])
    plt.xlim(xlim)
    if indx_c==3:        
        plt.gca().legend(loc= 2, frameon=False)
        plt.xticks([10, 20 ,30])
        
    else:
        plt.gca().legend(loc= 1, frameon=False).remove()
    


##♥    
plt.suptitle( 'LM', fontsize=18)
plt.tight_layout(w_pad=5, h_pad=5, rect=[0, 0.03, 1, 0.95])
plt.show(block=False)




plt.figure()
sns.lineplot(x='times', y='decoding', data=Df_shuff)
plt.show(block=False)

