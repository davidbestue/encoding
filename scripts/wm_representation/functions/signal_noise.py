# -*- coding: utf-8 -*-
"""
Created on Wed May 22 17:23:51 2019

@author: David Bestue 
""" 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from model_functions import *


##### LM
#
##### signal visual, ipc
#### Load reconstruction and take the interesting part
#xls = pd.ExcelFile('/home/david/Desktop/Reconstructions_LM.xlsx')
#sheets = xls.sheet_names
###
#R={}
#for sh in sheets:
#    R[sh]  = pd.read_excel(xls, sheet_name=sh)
#
#Decoding_df =[]
#
#for dataframes in R.keys():
#    df = R[dataframes]
#    a = pd.DataFrame(df.iloc[360,:])
#    a = a.reset_index()
#    a.columns = ['times', 'decoding']
#    a['times']=a['times'].astype(float)
#    a['region'] = dataframes.split('_')[1]
#    a['subject'] = dataframes.split('_')[0]
#    a['condition'] = dataframes.split('_')[-2] + '_' + dataframes.split('_')[-1] 
#    Decoding_df.append(a)
#
#
#
#Df = pd.concat(Decoding_df)
#Df['label'] = 'signal'
#
#
##### signal pfc
#### Load reconstruction and take the interesting part
#xls = pd.ExcelFile('/home/david/Desktop/Reconstructions_LM_pfc.xlsx')
#sheets = xls.sheet_names
###
#R={}
#for sh in sheets:
#    R[sh]  = pd.read_excel(xls, sheet_name=sh)
#
#Decoding_df =[]
#
#for dataframes in R.keys():
#    df = R[dataframes]
#    a = pd.DataFrame(df.iloc[360,:])
#    a = a.reset_index()
#    a.columns = ['times', 'decoding']
#    a['times']=a['times'].astype(float)
#    a['region'] = dataframes.split('_')[1]
#    a['subject'] = dataframes.split('_')[0]
#    a['condition'] = dataframes.split('_')[-2] + '_' + dataframes.split('_')[-1] 
#    Decoding_df.append(a)
#
#
#
#Df_pfc = pd.concat(Decoding_df)
#Df_pfc['label'] = 'signal'
#
#
#
#
### Load the shuffle (it already has the interesting part)
#Df_shuff = pd.read_excel('/home/david/Desktop/Reconstructions_LM_shuff.xlsx')
#Df_shuff['label'] = 'shuffle'
#Df_shuff_pfc = pd.read_excel('/home/david/Desktop/Reconstructions_LM_pfc_shuff.xlsx')
#Df_shuff_pfc['label'] = 'shuffle'
#
#
###combine them
#df = pd.concat([Df, Df_pfc, Df_shuff, Df_shuff_pfc])
#
#





################## Lasso
#xls = pd.ExcelFile('/home/david/Desktop/Reconstructions_Lasso.xlsx')
#xls = pd.ExcelFile('/home/david/Desktop/Reconstructions_Lasso_frontal.xlsx')
#xls = pd.ExcelFile('/home/david/Desktop/Reconstructions_Lasso_6.xlsx')
#xls = pd.ExcelFile('/home/david/Desktop/Reconstructions_LM_6.xlsx')
xls = pd.ExcelFile('/home/david/Desktop/Reconstructions_LM_n001_2_7_visual.xlsx')


sheets = xls.sheet_names
##
R={}
for sh in sheets:
    R[sh]  = pd.read_excel(xls, sheet_name=sh)

Decoding_df =[]

for dataframes in R.keys():
    df = R[dataframes]
    a = pd.DataFrame(df.iloc[360,:])
    a = a.reset_index()
    a.columns = ['times', 'decoding']
    a['decoding'] = [sum(R[dataframes].iloc[:,i] * f2(180)) for i in range(len(a))]
    a['times']=a['times'].astype(float)
    a['region'] = dataframes.split('_')[1]
    #a['region'] = dataframes.split('_')[1] + '_' + dataframes.split('_')[2]
    a['subject'] = dataframes.split('_')[0]
    a['condition'] = dataframes.split('_')[-2] + '_' + dataframes.split('_')[-1] 
    Decoding_df.append(a)



Df = pd.concat(Decoding_df)
Df['label'] = 'signal'


## Load the shuffle (it already has the interesting part)
#Df_shuff = pd.read_excel('/home/david/Desktop/Reconstructions_Lasso_shuff.xlsx')
#Df_shuff = pd.read_excel('/home/david/Desktop/Reconstructions_Lasso_frontal_shuff.xlsx')
#Df_shuff = pd.read_excel('/home/david/Desktop/Reconstructions_Lasso_6_shuff.xlsx')
#Df_shuff = pd.read_excel('/home/david/Desktop/Reconstructions_LM_6_shuff.xlsx')
Df_shuff = pd.read_excel('/home/david/Desktop/Reconstructions_LM_n001_2_7_visual_suff.xlsx')


Df_shuff['label'] = 'shuffle'


df = pd.concat([Df, Df_shuff])


presentation_period= 0.35 
presentation_period_cue=  0.50
inter_trial_period= 0.1 
pre_cue_period= 0.5 
pre_stim_period= 0.5 
limit_time=5 
ref_angle=45

#
#for condition in ['1_0.2', '1_7', '2_0.2', '2_7']:
#    
#    if condition == '1_0.2':
#        delay1 = 0.2
#        delay2 = 11.8
#        cue=0
#        t_p = cue + presentation_period_cue + pre_stim_period 
#        d_p = t_p + presentation_period +delay1 
#        r_t = d_p + presentation_period + delay2
#    elif condition == '1_7':
#        delay1 = 7
#        delay2 = 5
#        cue=0
#        t_p = cue + presentation_period_cue + pre_stim_period 
#        d_p = t_p + presentation_period +delay1 
#        r_t = d_p + presentation_period + delay2
#    elif condition == '2_0.2':
#        delay1 = 0.2
#        delay2 = 12
#        cue=0
#        d_p = cue + presentation_period_cue + pre_stim_period 
#        t_p = d_p + presentation_period +delay1 
#        r_t = t_p + presentation_period + delay2    
#    elif condition == '2_7':
#        delay1 = 7
#        delay2 = 12
#        cue=0
#        d_p = cue + presentation_period_cue + pre_stim_period 
#        t_p = d_p + presentation_period +delay1 
#        r_t = t_p + presentation_period + delay2
#        
#    
#    start_hrf = 3
#    sec_hdrf = 4
#    
#    d_p1 = (start_hrf + d_p) 
#    t_p1 = (start_hrf +t_p)
#    r_t1=  (start_hrf + r_t)
#    #
#    d_p2 = d_p1 + sec_hdrf 
#    t_p2 = t_p1 + sec_hdrf
#    r_t2=  r_t1 + sec_hdrf 
#    
#    y_vl_min = -0.2 #df_all_by_subj.Decoding.min()
#    y_vl_max = 0.2 #◙df_all_by_subj.Decoding.max()
#    
#    fig = plt.figure()
#    fig.set_size_inches(13, 4)
#    fig.tight_layout()
#    fig.suptitle(condition)
#    ax1 = fig.add_subplot(131)
#    ax2 = fig.add_subplot(132)
#    ax3 = fig.add_subplot(133)
#    sns.lineplot(ax= ax1, x="times", y="decoding", hue='label', hue_order = ['signal', 'shuffle'],  data=df.loc[ (df['condition']==condition)  & (df['region'] =='visual')]) 
#    ax1.fill_between(  [ t_p1, t_p2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='b', alpha=0.3, label='target'  )
#    ax1.fill_between(  [ d_p1, d_p2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='g', alpha=0.3, label='distractor'  )
#    ax1.fill_between(  [ r_t1, r_t2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='y', alpha=0.3, label='response'  )    
#    
#    sns.lineplot(ax= ax2, x="times", y="decoding", hue='label', hue_order = ['signal', 'shuffle'],  data=df.loc[ (df['condition']==condition)  & (df['region'] =='ips')]) 
#    ax2.fill_between(  [ t_p1, t_p2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='b', alpha=0.3, label='target'  )
#    ax2.fill_between(  [ d_p1, d_p2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='g', alpha=0.3, label='distractor'  )
#    ax2.fill_between(  [ r_t1, r_t2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='y', alpha=0.3, label='response'  )    
#    
#    sns.lineplot(ax= ax3, x="times", y="decoding", hue='label', hue_order = ['signal', 'shuffle'],  data=df.loc[ (df['condition']==condition)  & (df['region'] =='pfc')]) 
#    ax3.fill_between(  [ t_p1, t_p2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='b', alpha=0.3, label='target'  )
#    ax3.fill_between(  [ d_p1, d_p2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='g', alpha=0.3, label='distractor'  )
#    ax3.fill_between(  [ r_t1, r_t2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='y', alpha=0.3, label='response'  )  
#    
#    axes=[ax1, ax2, ax3]
#    Titles=['visual', 'ips', 'pfc']
#    
#    for i, Ax in enumerate(axes):
#        Ax.spines['right'].set_visible(False)
#        Ax.spines['top'].set_visible(False)
#        Ax.title.set_text(Titles[i])
#        #Ax.legend_.remove()
#        #Ax.set_xticklabels(['in','out'])
#        #Ax.set_xlabel('Distance T-Dist')
#        Ax.set_ylabel('decoding value')
#        Ax.set_xlabel('time')
#        #Ax.set_ylim(-8,8)
#    
#    
#    plt.show(block=False)
#
#

#plt.figure()
#ax = sns.lineplot(x="times", y="decoding", hue='label', hue_order = ['signal', 'shuffle'],  data=df.loc[ (df['condition']=='1_7') & (df['subject'] =='n001')  & (df['region'] =='visual')]) 
#plt.show(block=False)
#
#plt.figure()
#ax = sns.lineplot(x="times", y="decoding", hue='label', hue_order = ['signal', 'shuffle'],  data=df.loc[ (df['condition']=='1_7') & (df['subject'] =='n001')  & (df['region'] =='ips')]) 
#plt.show(block=False)


#### by subject


#
#    
#for Subject in ['n001', 'b001', 'd001', 'r001', 'l001', 's001']:    
#    for condition in ['1_0.2', '1_7', '2_0.2', '2_7']:
#        if condition == '1_0.2':
#            delay1 = 0.2
#            delay2 = 11.8
#            cue=0
#            t_p = cue + presentation_period_cue + pre_stim_period 
#            d_p = t_p + presentation_period +delay1 
#            r_t = d_p + presentation_period + delay2
#        elif condition == '1_7':
#            delay1 = 7
#            delay2 = 5
#            cue=0
#            t_p = cue + presentation_period_cue + pre_stim_period 
#            d_p = t_p + presentation_period +delay1 
#            r_t = d_p + presentation_period + delay2
#        elif condition == '2_0.2':
#            delay1 = 0.2
#            delay2 = 12
#            cue=0
#            d_p = cue + presentation_period_cue + pre_stim_period 
#            t_p = d_p + presentation_period +delay1 
#            r_t = t_p + presentation_period + delay2    
#        elif condition == '2_7':
#            delay1 = 7
#            delay2 = 12
#            cue=0
#            d_p = cue + presentation_period_cue + pre_stim_period 
#            t_p = d_p + presentation_period +delay1 
#            r_t = t_p + presentation_period + delay2
#            
#        
#        start_hrf = 3
#        sec_hdrf = 4
#        
#        d_p1 = (start_hrf + d_p) 
#        t_p1 = (start_hrf +t_p)
#        r_t1=  (start_hrf + r_t)
#        #
#        d_p2 = d_p1 + sec_hdrf 
#        t_p2 = t_p1 + sec_hdrf
#        r_t2=  r_t1 + sec_hdrf 
#        
#        y_vl_min = -0.2 #df_all_by_subj.Decoding.min()
#        y_vl_max = 0.2 #◙df_all_by_subj.Decoding.max()
#        
#        fig = plt.figure()
#        fig.set_size_inches(13, 4)
#        fig.tight_layout()
#        fig.suptitle(Subject + ': ' +condition)
#        ax1 = fig.add_subplot(131)
#        ax2 = fig.add_subplot(132)
#        ax3 = fig.add_subplot(133)
#        sns.lineplot(ax= ax1, x="times", y="decoding", hue='label', hue_order = ['signal', 'shuffle'],  data=df.loc[ (df['condition']==condition) & (df['subject']==Subject)  & (df['region'] =='visual')]) 
#        ax1.fill_between(  [ t_p1, t_p2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='b', alpha=0.3, label='target'  )
#        ax1.fill_between(  [ d_p1, d_p2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='g', alpha=0.3, label='distractor'  )
#        ax1.fill_between(  [ r_t1, r_t2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='y', alpha=0.3, label='response'  )    
#        
#        sns.lineplot(ax= ax2, x="times", y="decoding", hue='label', hue_order = ['signal', 'shuffle'],  data=df.loc[ (df['condition']==condition) & (df['subject']==Subject)   & (df['region'] =='ips')]) 
#        ax2.fill_between(  [ t_p1, t_p2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='b', alpha=0.3, label='target'  )
#        ax2.fill_between(  [ d_p1, d_p2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='g', alpha=0.3, label='distractor'  )
#        ax2.fill_between(  [ r_t1, r_t2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='y', alpha=0.3, label='response'  )    
#        
#        sns.lineplot(ax= ax3, x="times", y="decoding", hue='label', hue_order = ['signal', 'shuffle'],  data=df.loc[ (df['condition']==condition) & (df['subject']==Subject)   & (df['region'] =='pfc')]) 
#        ax3.fill_between(  [ t_p1, t_p2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='b', alpha=0.3, label='target'  )
#        ax3.fill_between(  [ d_p1, d_p2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='g', alpha=0.3, label='distractor'  )
#        ax3.fill_between(  [ r_t1, r_t2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='y', alpha=0.3, label='response'  )  
#        
#        axes=[ax1, ax2, ax3]
#        Titles=['visual', 'ips', 'pfc']
#        
#        for i, Ax in enumerate(axes):
#            Ax.spines['right'].set_visible(False)
#            Ax.spines['top'].set_visible(False)
#            Ax.title.set_text(Titles[i])
#            #Ax.legend_.remove()
#            #Ax.set_xticklabels(['in','out'])
#            #Ax.set_xlabel('Distance T-Dist')
#            Ax.set_ylabel('decoding value')
#            Ax.set_xlabel('time')
#            #Ax.set_ylim(-8,8)
#        
#        
#        plt.show(block=False)
#
#
#


##### Measure of difference to shuffle
subj_decoding=[]
for brain_region in ['visual', 'ips', 'frontsup', 'frontmid', 'frontinf']: 
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
    ax = fig.add_subplot(1,1, indx_c+1) 
    sns.lineplot( ax=ax, x="times", y="decoding", hue='region', hue_order =  ['visual', 'ips', 'frontsup', 'frontmid', 'frontinf'],  ci=69,  data=dfsn.loc[ (dfsn['condition']==condition)]) 
    
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
plt.suptitle( 'Lasso', fontsize=18)
plt.tight_layout(w_pad=5, h_pad=5, rect=[0, 0.03, 1, 0.95])
plt.show(block=False)





###### by subject
    
for Subject in ['n001', 'b001', 'd001', 'r001', 'l001', 's001']:
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
        sns.lineplot( ax=ax, x="times", y="decoding", hue='region', hue_order =  ['visual', 'ips', 'frontsup', 'frontmid', 'frontinf'],  ci=69,  data=dfsn.loc[ (dfsn['condition']==condition) & (dfsn['subject']== Subject )]) 
        
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
    plt.suptitle( Subject + ' lasso', fontsize=18)
    plt.tight_layout(w_pad=5, h_pad=5, rect=[0, 0.03, 1, 0.95])
    plt.show(block=False)





