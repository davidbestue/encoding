# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 11:56:47 2019

@author: David
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions_encoding_loop import *


root = '/home/david/Desktop/KAROLINSKA/together_mix_2TR/Conditions/'

dfs_visual = {}
dfs_ips = {}

#Parameters
presentation_period= 0.35 
presentation_period_cue=  0.50
inter_trial_period= 0.1 
pre_cue_period= 0.5 
pre_stim_period= 0.5 
limit_time=5 


for CONDITION in ['1_0.2', '1_7', '2_0.2', '2_7']:
    for SUBJECT_USE_ANALYSIS in ['d001', 'n001', 'r001', 'b001', 'l001', 's001']:
        for algorithm in ["visual", "ips"]:  
            Method_analysis = 'together'
            distance='mix'
            #CONDITION = '1_0.2' #'1_0.2', '1_7', '2_0.2', '2_7'
            
            ## Load Results
            Matrix_results_name = root +  CONDITION + '/' + SUBJECT_USE_ANALYSIS + '_' + algorithm + '_'  + CONDITION + '_'  + distance + '_' + Method_analysis + '.xlsx'
            xls = pd.ExcelFile(Matrix_results_name)
            sheets = xls.sheet_names
            ##
            if algorithm == 'visual':
                for sh in sheets:
                    Matrix_results = pd.read_excel(Matrix_results_name, sheet_name=sh)            
                    dfs_visual[ SUBJECT_USE_ANALYSIS + '_' + sh] = Matrix_results
            
            if algorithm == 'ips':
                for sh in sheets:
                    Matrix_results = pd.read_excel(Matrix_results_name, sheet_name=sh)            
                    dfs_ips[ SUBJECT_USE_ANALYSIS + '_' + sh] = Matrix_results
    
    
    
    
    #####
    #####
    
    panel_v=pd.Panel(dfs_visual)
    df_visual=panel_v.mean(axis=0)
    df_visual.columns = [float(df_visual.columns[i])*2 for i in range(0, len(df_visual.columns))]
    
    panel_i=pd.Panel(dfs_ips)
    df_ips=panel_i.mean(axis=0)
    df_ips.columns = [float(df_ips.columns[i])*2 for i in range(0, len(df_ips.columns))]
    
    
    df_heatmaps = {}
    df_heatmaps['ips'] = df_ips
    df_heatmaps['visual'] = df_visual
    
    #####
    #####
    
    
    b_reg = []
    b_reg360=[]
    
    for algorithm in ['visual', 'ips']:
        plt.figure()
        TITLE_HEATMAP =  algorithm + '_' + CONDITION + '_' +distance + '_' + Method_analysis + ' heatmap'
        plt.title(TITLE_HEATMAP)
        #midpoint = df.values.mean() # (df.values.max() - df.values.min()) / 2
        ax = sns.heatmap(df_heatmaps[algorithm], yticklabels=list(df_heatmaps[algorithm].index), cmap="coolwarm", vmin=-0.1, vmax=0.1) # cmap= viridis "jet",  "coolwarm" RdBu_r, gnuplot, YlOrRd, CMRmap  , center = midpoint
        #ax.invert_yaxis()
        ax.plot([0.25, shape(df_heatmaps[algorithm])[1]-0.25], [posch1_to_posch2(4),posch1_to_posch2(4)], 'k--')
        plt.yticks([posch1_to_posch2(4), posch1_to_posch2(13), posch1_to_posch2(22), posch1_to_posch2(31)] ,['45','135','225', '315'])
        plt.ylabel('Angle')
        plt.xlabel('time (s)')
        plt.show(block=False)
        
        #### TSplot preferred
        ref_angle=45
        Angle_ch = ref_angle * (len(df_heatmaps[algorithm]) / 360)
        df_45 = df_heatmaps[algorithm].iloc[int(Angle_ch)-20 : int(Angle_ch)+20]
        df_together = df_45.melt()
        df_together['ROI'] = [algorithm for i in range(0, len(df_together))]
        df_together['voxel'] = [i+1 for i in range(0, len(df_45))]*np.shape(df_45)[1]
        df_together.columns = ['timepoint', 'Decoding', 'ROI', 'voxel']
        df_together['timepoint'] = [float(df_together['timepoint'].iloc[i]) for i in range(0, len(df_together))]
        b_reg.append(df_together)
        
        ##for whole area
        Angle_ch = ref_angle * (len(df_heatmaps[algorithm]) / 360)
        df_all360 = df_heatmaps[algorithm]
        df_together = df_all360.melt()
        df_together['ROI'] = [algorithm for i in range(0, len(df_together))]
        df_together['voxel'] = [i+1 for i in range(0, len(df_all360))]*np.shape(df_all360)[1]
        df_together.columns = ['timepoint', 'Decoding', 'ROI', 'voxel']
        df_together['timepoint'] = [float(df_together['timepoint'].iloc[i]) for i in range(0, len(df_together))]
        b_reg360.append(df_together)
        
        #plt.figure()   
        #### FactorPlot preferred (save)
        #a=sns.factorplot(x='timepoint', y='Decoding',  data=df_together, size=5, aspect=1.5)        
        #plt.ylabel('Decoding value')
        #plt.xlabel('time (s)')
        #TITLE_PREFERRED =  algorithm + '_' + CONDITION + '_' +distance + '_' + Method_analysis + ' preferred'
        #plt.title(TITLE_PREFERRED)
        #plt.show(block=False)
        
    
    ### FactorPlot all brain region
    #12.35 in 1 and 12 in 2 ( :S :S aghhhhhhh should not affect both in beh and imaging )
    ### depending on condition
    if CONDITION == '1_0.2':
        delay1 = 0.2
        delay2 = 11.8
        cue=0
        t_p = cue + presentation_period_cue + pre_stim_period 
        d_p = t_p + presentation_period +delay1 
        r_t = d_p + presentation_period + delay2
    elif CONDITION == '1_7':
        delay1 = 7
        delay2 = 5
        cue=0
        t_p = cue + presentation_period_cue + pre_stim_period 
        d_p = t_p + presentation_period +delay1 
        r_t = d_p + presentation_period + delay2
    elif CONDITION == '2_0.2':
        delay1 = 0.2
        delay2 = 12
        cue=0
        d_p = cue + presentation_period_cue + pre_stim_period 
        t_p = d_p + presentation_period +delay1 
        r_t = t_p + presentation_period + delay2    
    elif CONDITION == '2_7':
        delay1 = 7
        delay2 = 12
        cue=0
        d_p = cue + presentation_period_cue + pre_stim_period 
        t_p = d_p + presentation_period +delay1 
        r_t = t_p + presentation_period + delay2
    
    
    ## position in axes
    
    plt.figure()
    df_all = pd.concat(b_reg)    
    x_bins = len(df_all.timepoint.unique())
    
    d_pp = d_p/x_bins
    t_pp = t_p/x_bins
    r_tp= r_t/x_bins
    
    y_vl_min = df_all.Decoding.min()
    y_vl_max = df_all.Decoding.max()
    
    range_hrf = [float(5)/x_bins, float(6)/x_bins] #  
    sns.pointplot(x='timepoint', y='Decoding', hue='ROI', data=df_all, size=5, aspect=1.5)
    plt.fill_between(  [ range_hrf[0] + t_pp, range_hrf[1] + t_pp ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='b', alpha=0.3, label='target'  )
    plt.fill_between(  [ range_hrf[0] + d_pp, range_hrf[1] + d_pp ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='g', alpha=0.3, label='distractor'  )
    plt.fill_between(  [ range_hrf[0] + r_tp, range_hrf[1] + r_tp ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='y', alpha=0.3, label='response'  )
    plt.ylabel('Decoding value')
    plt.xlabel('time (s)')
    TITLE_BR = CONDITION + '_' +distance + '_' + Method_analysis + ' preferred b_r'
    plt.legend(frameon=False)
    plt.title(TITLE_BR)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().get_xaxis().tick_bottom()
    plt.gca().get_yaxis().tick_left()
    plt.tight_layout()
    plt.show(block=False)
    
    
    #### plot all 360 in area
    plt.figure()
    df_all = pd.concat(b_reg360)    
    x_bins = len(df_all.timepoint.unique())
    
    y_vl_min = -0.1
    y_vl_max = 0.1
    
    range_hrf = [float(5)/x_bins, float(6)/x_bins] #  
    sns.pointplot(x='timepoint', y='Decoding', hue='ROI', data=df_all, size=5, aspect=1.5)
    plt.fill_between(  [ range_hrf[0] + t_p, range_hrf[1] + t_p ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='b', alpha=0.3, label='target'  )
    plt.fill_between(  [ range_hrf[0] + d_p, range_hrf[1] + d_p ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='g', alpha=0.3, label='distractor'  )
    plt.fill_between(  [ range_hrf[0] + r_t, range_hrf[1] + r_t ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='y', alpha=0.3, label='response'  )
    plt.ylabel('Decoding value')
    plt.xlabel('time (s)')
    TITLE_BR = CONDITION + '_' +distance + '_' + Method_analysis + ' all 360'
    plt.legend(frameon=False)
    plt.title(TITLE_BR)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().get_xaxis().tick_bottom()
    plt.gca().get_yaxis().tick_left()
    plt.tight_layout()
    plt.show(block=False)











