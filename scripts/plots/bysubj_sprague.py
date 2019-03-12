# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 11:56:47 2019

@author: David Bestue
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from functions_encoding_loop import *


root = '/mnt/c/Users/David/Desktop/n001_bysess_mix_2TR_target/Conditions/'



#Parameters
presentation_period= 0.35 
presentation_period_cue=  0.50
inter_trial_period= 0.1 
pre_cue_period= 0.5 
pre_stim_period= 0.5 
limit_time=5 

ref_angle=45
dec_thing = 'response'


def decode_sprague(RE):
    #  to implement:   mean (r (theta) cos( theta))
    N=len(RE)
    R = []
    angles = np.arange(0,N)*2*np.pi/N    
    R = [   round(np.cos(angles[i]) * RE[i], 3) for i in range(N)]
    angle_dec = np.mean(R)
    return np.degrees(angle_dec)



def ub_wind_path(PATH, system):
    if system=='wind':
        A = PATH                                                                                                    
        B = A.replace('/', os.path.sep)                                                                                               
        C= B.replace('\\mnt\\c\\', 'C:\\') 
    
    if system=='unix':
        C=PATH
    
    ###
    return C




for i_c, CONDITION in enumerate(['1_0.2', '1_7', '2_0.2', '2_7']): #
    dfs_visual = []
    dfs_ips = []
    plt.subplot(2,2,i_c+1)
    for SUBJECT_USE_ANALYSIS in ['n001']:  #'d001', 'n001', 'r001', 'b001', 'l001', 's001'
        for brain_region in ["visual", "ips"]:  
            Method_analysis = 'bysess'
            distance='mix'
            #CONDITION = '1_0.2' #'1_0.2', '1_7', '2_0.2', '2_7'
            
            ## Load Results
            Matrix_results_name = root +  CONDITION + '/' + SUBJECT_USE_ANALYSIS + '_' + brain_region + '_'  + CONDITION + '_'  + distance + '_' + Method_analysis + '.xlsx'
            Matrix_results_name= ub_wind_path(Matrix_results_name, system='wind') 
            xls = pd.ExcelFile(Matrix_results_name)
            sheets = xls.sheet_names
            ##
            if brain_region == 'visual':
                for sh in sheets:
                    Matrix_results = pd.read_excel(Matrix_results_name, sheet_name=sh)  
                    df_rolled = Matrix_results.iloc[:180, :] ### just the quadrant
                    df_rolled=np.roll(df_rolled, -2*ref_angle, 0) #roll a 0 sie l prefreed es el 45
                    df_rolled=pd.DataFrame(df_rolled)
                    #df_rolled[df_rolled<0]=0
                    df_rolled['session'] = sh[-1]
                    dfs_visual.append(df_rolled)
            
            if brain_region == 'ips':
                for sh in sheets:
                    Matrix_results = pd.read_excel(Matrix_results_name, sheet_name=sh)  
                    df_rolled = Matrix_results.iloc[:180, :] 
                    df_rolled=np.roll(df_rolled, -2*ref_angle, 0) #roll a 0 sie l prefreed es el 45
                    df_rolled=pd.DataFrame(df_rolled)
                    #df_rolled[df_rolled<0]=0
                    df_rolled['session'] = sh[-1]
                    dfs_ips.append(df_rolled) 
    
    
    #####
    ##### 
    df_visual = pd.concat(dfs_visual)
    df_ips = pd.concat(dfs_ips)
    
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
    #   
    TIMES = list(np.array([float(Matrix_results.columns.values[i]) for i in range(len(Matrix_results.columns.values))]) * 2 )
    
    for brain_region in ['visual', 'ips']:
#        plt.figure()
#        TITLE_HEATMAP =  brain_region + '_' + CONDITION + '_' +distance + '_' + Method_analysis + ' heatmap'
#        plt.title(TITLE_HEATMAP)
#        #midpoint = df.values.mean() # (df.values.max() - df.values.min()) / 2
#        ax = sns.heatmap(df_heatmaps[brain_region], yticklabels=list(df_heatmaps[brain_region].index), cmap="coolwarm", vmin=-0.1, vmax=0.1) # cmap= viridis "jet",  "coolwarm" RdBu_r, gnuplot, YlOrRd, CMRmap  , center = midpoint
#        #ax.invert_yaxis()
#        ax.plot([0.25, shape(df_heatmaps[brain_region])[1]-0.25], [posch1_to_posch2(4),posch1_to_posch2(4)], 'k--')
#        plt.yticks([posch1_to_posch2(4), posch1_to_posch2(13), posch1_to_posch2(22), posch1_to_posch2(31)] ,['45','135','225', '315'])
#        plt.ylabel('Angle')
#        plt.xlabel('time (s)')
#        plt.show(block=False)
        
        #### TSplot preferred
        # by_subj
        for Sess in range(len(df_heatmaps_by_subj[brain_region])):
            values= [ round(decode_sprague(df_heatmaps_by_subj[brain_region][Sess].iloc[:, TR]), 3) for TR in range(0, np.shape(df_heatmaps_by_subj[brain_region][Sess])[1] -1)]
            #times= list(df_heatmaps[brain_region].columns)
            df_together_s = pd.DataFrame({'Decoding':values, 'timepoint':TIMES})
            df_together_s['ROI'] = [brain_region +  '_'+ dec_thing  for i in range(0, len(df_together_s))]
            df_together_s['subj'] = SUBJECT_USE_ANALYSIS
            df_together_s['session']=Sess
            b_reg_by_subj.append(df_together_s)
        
        
        #####
        #####
        ## for whole area
#        Angle_ch = ref_angle * (len(df_heatmaps[brain_region]) / 360)
#        df_all360 = df_heatmaps[brain_region]
#        df_together = df_all360.melt()
#        df_together['ROI'] = [brain_region for i in range(0, len(df_together))]
#        df_together['voxel'] = [i+1 for i in range(0, len(df_all360))]*np.shape(df_all360)[1]
#        df_together.columns = ['timepoint', 'Decoding', 'ROI', 'voxel']
#        df_together['timepoint'] = [float(df_together['timepoint'].iloc[i]) for i in range(0, len(df_together))]
#        b_reg360.append(df_together)
    
    
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
    
    df_all_by_subj = pd.concat(b_reg_by_subj)
    x_bins = len(df_all_by_subj.timepoint.unique()) -1 
    max_val_x = df_all_by_subj.timepoint.max()
    
    start_hrf = 4
    sec_hdrf = 2
    
    d_p1 = (start_hrf + d_p) * x_bins/ max_val_x
    t_p1 = (start_hrf +t_p)* x_bins/ max_val_x
    r_t1=  (start_hrf + r_t)* x_bins/ max_val_x
    #
    d_p2 = d_p1 + sec_hdrf * x_bins/ max_val_x
    t_p2 = t_p1 + sec_hdrf * x_bins/ max_val_x
    r_t2=  r_t1 + sec_hdrf * x_bins/ max_val_x
    
    y_vl_min = df_all_by_subj.Decoding.min()
    y_vl_max = df_all_by_subj.Decoding.max()
    
    range_hrf = [float(5)/x_bins, float(6)/x_bins] #  
    #paper_rc = {'lines.linewidth': 2, 'lines.markersize': 2}  
    #sns.set_context("paper", rc = paper_rc) 
    #sns.pointplot(x='timepoint', y='Decoding', hue='ROI', data=df_all, size=5, aspect=1.5)
    ##all subj visual
    paper_rc = {'lines.linewidth': 0.45, 'lines.markersize': 0.5}                  
    sns.set_context("paper", rc = paper_rc)
    for a in ['visual_response', 'ips_response']: 
        if a=='visual_response':
            c='b'
        elif a =='ips_response':
            c='darkorange'
        for s in df_all_by_subj.session.unique():
            sns.pointplot(x='timepoint', y='Decoding',
                          data=df_all_by_subj.loc[ (df_all_by_subj['ROI']==a) & (df_all_by_subj['session']==s) ],
                          linestyles='--', color=c, legend=False, size=5, aspect=1.5)   
    
    
    #   
    ##all subj visual   
    plt.fill_between(  [ t_p1, t_p2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='b', alpha=0.3, label='target'  )
    plt.fill_between(  [ d_p1, d_p2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='g', alpha=0.3, label='distractor'  )
    plt.fill_between(  [ r_t1, r_t2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='y', alpha=0.3, label='response'  )
    plt.ylabel('Decoding value')
    plt.xlabel('time (s)')
    TITLE_BR = CONDITION 
    plt.legend(frameon=False)
    plt.title(TITLE_BR)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().get_xaxis().tick_bottom()
    plt.gca().get_yaxis().tick_left()
    plt.gca().legend(loc= 0, frameon=False)
    

plt.tight_layout()
plt.suptitle( 'Sprague by session n001, ' +distance + '_' + Method_analysis, fontsize=12)
plt.show(block=False)
    
    #### plot all 360 in area
#    plt.figure()
#    df_all = pd.concat(b_reg360)    
#    x_bins = len(df_all.timepoint.unique()) -1 
#    
#    start_hrf = 4
#    sec_hdrf = 2
#    max_val_x = df_all.timepoint.max()
#    
#    d_p1 = (start_hrf + d_p) * x_bins/ max_val_x
#    t_p1 = (start_hrf +t_p)* x_bins/ max_val_x
#    r_t1=  (start_hrf + r_t)* x_bins/ max_val_x
#    #
#    d_p2 = d_p1 + sec_hdrf * x_bins/ max_val_x
#    t_p2 = t_p1 + sec_hdrf * x_bins/ max_val_x
#    r_t2=  r_t1 + sec_hdrf * x_bins/ max_val_x
#    
#    y_vl_min = -0.1
#    y_vl_max = 0.1
#    
#    range_hrf = [float(5)/x_bins, float(6)/x_bins] #  
#    sns.pointplot(x='timepoint', y='Decoding', hue='ROI', data=df_all, size=5, aspect=1.5)
#    plt.fill_between(  [ t_p1, t_p2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='b', alpha=0.3, label='target'  )
#    plt.fill_between(  [ d_p1, d_p2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='g', alpha=0.3, label='distractor'  )
#    plt.fill_between(  [ r_t1, r_t2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='y', alpha=0.3, label='response'  )
#    plt.ylabel('Decoding value')
#    plt.xlabel('time (s)')
#    TITLE_BR = CONDITION + '_' +distance + '_' + Method_analysis + ' all 360'
#    plt.legend(frameon=False)
#    plt.title(TITLE_BR)
#    plt.gca().spines['right'].set_visible(False)
#    plt.gca().spines['top'].set_visible(False)
#    plt.gca().get_xaxis().tick_bottom()
#    plt.gca().get_yaxis().tick_left()
#    plt.tight_layout()
#    plt.show(block=False)






