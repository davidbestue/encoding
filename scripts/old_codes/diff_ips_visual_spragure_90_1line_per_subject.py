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
#from functions_encoding_loop import *

root = '/mnt/c/Users/David/Desktop/together_mix_2TR/Conditions/'
root = '/mnt/c/Users/David/Desktop/together_mix_2TR_response_zs5/Conditions/'

#root = '/home/david/Desktop/KAROLINSKA/together_mix_2TR_distractor/Conditions/'




def ub_wind_path(PATH, system):
    if system=='wind':
        A = PATH                                                                                                    
        B = A.replace('/', os.path.sep)                                                                                               
        C= B.replace('\\mnt\\c\\', 'C:\\') 
    
    if system=='unix':
        C=PATH
    
    ###
    return C

##
##

def decode_sprague(RE):
    #  to implement:   mean (r (theta) cos( theta))
    N=len(RE)
    R = []
    angles = np.arange(0,N)*2*np.pi/N    
    R = [   round(np.cos(angles[i]) * RE[i], 3) for i in range(N)]
    angle_dec = np.mean(R)
    return np.degrees(angle_dec)



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


subjects = ['d001', 'n001', 'r001', 'b001', 'l001', 's001'] #   

 


for i_c, CONDITION in enumerate(['1_0.2', '1_7', '2_0.2', '2_7']): #
    ax = plt.subplot(2, 2, i_c+1)
    ###   
    for SUBJECT_USE_ANALYSIS in subjects: #   , 'r001', 'b001', 'l001', 's001'
        for brain_region in ["visual", "ips"]:  
            Method_analysis = 'together'
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
                    #df_rolled = Matrix_results
                    df_rolled = Matrix_results.iloc[:180, :] ### just the quadrant
                    df_rolled=np.roll(df_rolled, -2*ref_angle, 0) #roll a 0 sie l prefreed es el 45
                    df_rolled=pd.DataFrame(df_rolled)
                    #df_rolled[df_rolled<0]=0 #uncomment if you want just the positive
                    dfs_visual[ SUBJECT_USE_ANALYSIS + '_' + sh] = df_rolled
            
            if brain_region == 'ips':
                for sh in sheets:
                    Matrix_results = pd.read_excel(Matrix_results_name, sheet_name=sh)  
                    #df_rolled = Matrix_results
                    df_rolled = Matrix_results.iloc[:180, :] 
                    df_rolled=np.roll(df_rolled, -2*ref_angle, 0) #roll a 0 sie l prefreed es el 45
                    df_rolled=pd.DataFrame(df_rolled)
                    #df_rolled[df_rolled<0]=0 #uncomment if you want just the positive
                    dfs_ips[ SUBJECT_USE_ANALYSIS + '_' + sh] = df_rolled
    
    
    
    
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
    b_reg_by_subj = []
    TIMES = list(np.array([float(Matrix_results.columns.values[i]) for i in range(len(Matrix_results.columns.values))]) * 2 )
    #    
    for brain_region in ['visual', 'ips']:
        # by_subj
        for Subj in df_heatmaps_by_subj[brain_region].keys():
            values= [ round(decode_sprague(df_heatmaps_by_subj[brain_region][Subj].iloc[:, TR]), 3) for TR in range(0, np.shape(df_heatmaps_by_subj[brain_region][Subj])[1])]
            #times= list(df_heatmaps[brain_region].columns)
            df_together_s = pd.DataFrame({'Decoding':values, 'timepoint':TIMES})
            df_together_s['ROI'] = [brain_region  for i in range(0, len(df_together_s))]
            df_together_s['subj'] = Subj.split('_')[0]
            b_reg_by_subj.append(df_together_s)

    
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
    
    diff_values=[]    
    for i in range(0,len(subjects)):
        diff_values.append(b_reg_by_subj[i]['Decoding'] - b_reg_by_subj[ len(subjects)+i]['Decoding'] )
    
    
    diff_values = pd.concat(diff_values, ignore_index=True)
    df_all_by_subj = pd.concat(b_reg_by_subj[:len(subjects)], ignore_index=True)  
    df_all_by_subj['Decoding'] = diff_values
    df_all_by_subj['ROI'] = 'visual-ips'
    
    
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
    
    y_vl_min = -8 #df_all_by_subj.Decoding.min()
    y_vl_max = 10 #df_all_by_subj.Decoding.max()
    
    ####
    ####
    
    ##### plot mean and subjects in olive (no label per subject)
    range_hrf = [float(5)/x_bins, float(6)/x_bins] #  
    paper_rc = {'lines.linewidth': 2, 'lines.markersize': 2}  
    sns.set_context("paper", rc = paper_rc) 
    sns.pointplot(ax=ax, x='timepoint', y='Decoding', data=df_all_by_subj, size=5, color ='salmon', aspect=1.5) # 
    ##all subj visual
    paper_rc = {'lines.linewidth': 0.4, 'lines.markersize': 0.5}  
    sns.set_context("paper", rc = paper_rc)
    Pallete = sns.color_palette("tab10", n_colors=len(df_all_by_subj.subj.unique()), desat=1).as_hex()
    for idx, s in enumerate(df_all_by_subj.subj.unique()):
        sns.pointplot(x='timepoint', y='Decoding',
                      data=df_all_by_subj.loc[ (df_all_by_subj['ROI']=='visual-ips') & (df_all_by_subj['subj']==s) ],
                      linestyles='--', color='olive', size=5, aspect=1.5, label=s)   ## 'olive'




    ##### plot just the subjects (with labels)
#    paper_rc = {'lines.linewidth': 1, 'lines.markersize': 1}                 
#    sns.set_context("paper", rc = paper_rc)
#    Pallete = sns.color_palette("tab10", n_colors=len(df_all_by_subj.subj.unique()), desat=1).as_hex()
#    sns.pointplot(x='timepoint', y='Decoding', hue='subj', data=df_all_by_subj,
#                      linestyles='--', palette =Pallete, size=5, aspect=1.5)   ## 'olive
    
    
    ###all subj visual   
    plt.fill_between(  [ t_p1, t_p2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='b', alpha=0.3, label='target'  )
    plt.fill_between(  [ d_p1, d_p2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='g', alpha=0.3, label='distractor'  )
    plt.fill_between(  [ r_t1, r_t2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='y', alpha=0.3, label='response'  )
    plt.ylabel('Decoding value')
    plt.xlabel('time (s)')
    TITLE_BR = CONDITION 
    plt.legend(frameon=False)
    plt.title(TITLE_BR)
    plt.ylim(-8,10)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().get_xaxis().tick_bottom()
    plt.gca().get_yaxis().tick_left()
    plt.gca().legend(loc= 0, frameon=False)
    

plt.tight_layout()
plt.suptitle( 'Visual - IPS: response, ' +distance + '_' + Method_analysis, fontsize=12)
plt.show(block=False)




