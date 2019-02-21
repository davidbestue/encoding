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


root = '/home/david/Desktop/KAROLINSKA/together_mix_2TR_shuff/Conditions/'

dfs_visual = {}
dfs_ips = {}

#Parameters
presentation_period= 0.35 
presentation_period_cue=  0.50
inter_trial_period= 0.1 
pre_cue_period= 0.5 
pre_stim_period= 0.5 
limit_time=5 
ref_angle=45


def decode(RE):
    N=len(RE)
    R = []
    angles = np.arange(0,N)*2*np.pi/N
    R=np.dot(RE,np.exp(1j*angles)) / N
    angle = np.angle(R)
    if angle < 0:
        angle +=2*np.pi 
    return np.degrees(angle)


###

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
                    df_rolled=np.roll(Matrix_results, -2*ref_angle, 0)
                    df_rolled=pd.DataFrame(df_rolled)
                    dfs_visual[ SUBJECT_USE_ANALYSIS + '_' + sh] = df_rolled
            
            if algorithm == 'ips':
                for sh in sheets:
                    Matrix_results = pd.read_excel(Matrix_results_name, sheet_name=sh)   
                    df_rolled=np.roll(Matrix_results, -2*ref_angle, 0)
                    df_rolled=pd.DataFrame(df_rolled)
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
    
    df_heatmaps_by_subj = {}
    df_heatmaps_by_subj['ips'] = dfs_ips
    df_heatmaps_by_subj['visual'] = dfs_visual
    
    
    #####
    #####
    
    
    b_reg = []
    b_reg_by_subj = []
    b_reg360=[]
    
    for algorithm in ['visual', 'ips']:
#        plt.figure()
#        TITLE_HEATMAP =  algorithm + '_' + CONDITION + '_' +distance + '_' + Method_analysis + ' heatmap'
#        plt.title(TITLE_HEATMAP)
#        #midpoint = df.values.mean() # (df.values.max() - df.values.min()) / 2
#        ax = sns.heatmap(df_heatmaps[algorithm], yticklabels=list(df_heatmaps[algorithm].index), cmap="coolwarm", vmin=-0.1, vmax=0.1) # cmap= viridis "jet",  "coolwarm" RdBu_r, gnuplot, YlOrRd, CMRmap  , center = midpoint
#        #ax.invert_yaxis()
#        ax.plot([0.25, shape(df_heatmaps[algorithm])[1]-0.25], [posch1_to_posch2(4),posch1_to_posch2(4)], 'k--')
#        plt.yticks([posch1_to_posch2(4), posch1_to_posch2(13), posch1_to_posch2(22), posch1_to_posch2(31)] ,['45','135','225', '315'])
#        plt.ylabel('Angle')
#        plt.xlabel('time (s)')
#        plt.show(block=False)
        
        #### TSplot preferred
        ## mean
        ref_angle=45
        Angle_ch = ref_angle * (len(df_heatmaps[algorithm]) / 360)
        values= [ decode(df_heatmaps[algorithm].iloc[:, TR]) for TR in range(0, np.shape(df_heatmaps)[1])]
        times= list(df_heatmaps.columns)
        df_together = pd.DataFrame({'Decoding':values, 'timepoint':times})
        df_together['ROI'] = [algorithm for i in range(0, len(df_together))]
        b_reg.append(df_together)
        
#        ## by_subj
#        ref_angle=45
#        for Subj in df_heatmaps_by_subj[algorithm].keys():
#            Angle_ch = ref_angle * (len(     df_heatmaps_by_subj[algorithm][Subj]    ) / 360)
#            df_45 = df_heatmaps_by_subj[algorithm][Subj].iloc[int(Angle_ch)-20 : int(Angle_ch)+20]
#            df_together = df_45.melt()
#            df_together['ROI'] = [algorithm for i in range(0, len(df_together))]
#            df_together['voxel'] = [i+1 for i in range(0, len(df_45))]*np.shape(df_45)[1]
#            df_together.columns = ['timepoint', 'Decoding', 'ROI', 'voxel']
#            df_together['timepoint'] = [float(df_together['timepoint'].iloc[i]) for i in range(0, len(df_together))]
#            df_together['subj'] = Subj.split('_')[0]
#            b_reg_by_subj.append(df_together)
#        
#        
#        #####
#        #####
#        ## for whole area
##        Angle_ch = ref_angle * (len(df_heatmaps[algorithm]) / 360)
##        df_all360 = df_heatmaps[algorithm]
##        df_together = df_all360.melt()
##        df_together['ROI'] = [algorithm for i in range(0, len(df_together))]
##        df_together['voxel'] = [i+1 for i in range(0, len(df_all360))]*np.shape(df_all360)[1]
##        df_together.columns = ['timepoint', 'Decoding', 'ROI', 'voxel']
##        df_together['timepoint'] = [float(df_together['timepoint'].iloc[i]) for i in range(0, len(df_together))]
##        b_reg360.append(df_together)
#    
#    
#    ### FactorPlot all brain region
#    #12.35 in 1 and 12 in 2 ( :S :S aghhhhhhh should not affect both in beh and imaging )
#    ### depending on condition
#    if CONDITION == '1_0.2':
#        delay1 = 0.2
#        delay2 = 11.8
#        cue=0
#        t_p = cue + presentation_period_cue + pre_stim_period 
#        d_p = t_p + presentation_period +delay1 
#        r_t = d_p + presentation_period + delay2
#    elif CONDITION == '1_7':
#        delay1 = 7
#        delay2 = 5
#        cue=0
#        t_p = cue + presentation_period_cue + pre_stim_period 
#        d_p = t_p + presentation_period +delay1 
#        r_t = d_p + presentation_period + delay2
#    elif CONDITION == '2_0.2':
#        delay1 = 0.2
#        delay2 = 12
#        cue=0
#        d_p = cue + presentation_period_cue + pre_stim_period 
#        t_p = d_p + presentation_period +delay1 
#        r_t = t_p + presentation_period + delay2    
#    elif CONDITION == '2_7':
#        delay1 = 7
#        delay2 = 12
#        cue=0
#        d_p = cue + presentation_period_cue + pre_stim_period 
#        t_p = d_p + presentation_period +delay1 
#        r_t = t_p + presentation_period + delay2
#    
#    
#    ## position in axes
#    
#    plt.figure()
#    df_all = pd.concat(b_reg)   
#    df_all_by_subj = pd.concat(b_reg_by_subj)
#    x_bins = len(df_all.timepoint.unique()) -1 
#    max_val_x = df_all.timepoint.max()
#    
#    start_hrf = 4
#    sec_hdrf = 2
#    
#    d_p1 = (start_hrf + d_p) * x_bins/ max_val_x
#    t_p1 = (start_hrf +t_p)* x_bins/ max_val_x
#    r_t1=  (start_hrf + r_t)* x_bins/ max_val_x
#    #
#    d_p2 = d_p1 + sec_hdrf * x_bins/ max_val_x
#    t_p2 = t_p1 + sec_hdrf * x_bins/ max_val_x
#    r_t2=  r_t1 + sec_hdrf * x_bins/ max_val_x
#    
#    y_vl_min = df_all.Decoding.min()
#    y_vl_max = df_all.Decoding.max()
#    
#    range_hrf = [float(5)/x_bins, float(6)/x_bins] #  
#    paper_rc = {'lines.linewidth': 2, 'lines.markersize': 2}  
#    sns.set_context("paper", rc = paper_rc) 
#    sns.pointplot(x='timepoint', y='Decoding', hue='ROI', data=df_all, size=5, aspect=1.5)
#    ##all subj visual
#    paper_rc = {'lines.linewidth': 0.25, 'lines.markersize': 0.5}                  
#    sns.set_context("paper", rc = paper_rc)
#    for a in ['visual', 'ips']: 
#        if a=='visual':
#            c='b'
#        elif a =='ips':
#            c='darkorange'
#        for s in df_all_by_subj.subj.unique():
#            sns.pointplot(x='timepoint', y='Decoding',
#                          data=df_all_by_subj.loc[ (df_all_by_subj['ROI']==a) & (df_all_by_subj['subj']==s) ],
#                          linestyles='--', color=c, legend=False, size=5, aspect=1.5)   
#    
#    
#    #   
#    plt.fill_between(  [ t_p1, t_p2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='b', alpha=0.3, label='target'  )
#    plt.fill_between(  [ d_p1, d_p2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='g', alpha=0.3, label='distractor'  )
#    plt.fill_between(  [ r_t1, r_t2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='y', alpha=0.3, label='response'  )
#    plt.ylabel('Decoding value')
#    plt.xlabel('time (s)')
#    TITLE_BR = CONDITION + '_' +distance + '_' + Method_analysis + ' preferred b_r'
#    plt.legend(frameon=False)
#    plt.title(TITLE_BR)
#    plt.gca().spines['right'].set_visible(False)
#    plt.gca().spines['top'].set_visible(False)
#    plt.gca().get_xaxis().tick_bottom()
#    plt.gca().get_yaxis().tick_left()
#    plt.tight_layout()
#    plt.show(block=False)
    







