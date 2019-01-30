# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 17:19:05 2019

@author: David
"""

import easygui
import os

msg = "Decide computer"
choices = ["local", "cluster"]
platform = easygui.buttonbox(msg, choices=choices)

if platform == "local":
    root_use ='/mnt/c/Users/David/Desktop/KI_Desktop/IEM_data/'
    encoding_path = 'C:\\Users\\David\\Dropbox\\KAROLINSKA\\encoding_model\\'
    Conditions_enc_path = 'C:\\Users\\David\\Dropbox\\KAROLINSKA\\encoding_model\\Conditions\\'
    PLOTS_path = '\\plots'
    sys_use='wind'
    
elif platform == "cluster":
    root_use ='/home/david/Desktop/IEM_data/'
    encoding_path = '/home/david/Desktop/KAROLINSKA/encoding_model/'
    Conditions_enc_path = '/home/david/Desktop/KAROLINSKA/encoding_model/Conditions/'
    PLOTS_path = '/plots'
    sys_use='unix'
    
    
##Methods_analysis=[]
##
for SUBJECT_USE_ANALYSIS in ['d001']:
    print(SUBJECT_USE_ANALYSIS)
    for algorithm in ["visual"]:  #"ips"
        for CONDITION in ['1_0.2']: 
            Method_analysis = 'together'
            #CONDITION = '1_0.2'
            #algorithm = "visual"
            distance_ch='mix'
            #distance='mix'
            Subject_analysis=SUBJECT_USE_ANALYSIS
            os.chdir(encoding_path)
            ############################################       
            from functions_encoding_loop import *
            Method_analysis, CONDITION, distance_ch, Subject_analysis, algorithm, distance, func_encoding_sess, Beh_enc_files_sess, func_wmtask_sess, Beh_WM_files_sess, path_masks, Maskrh, Masklh, writer_matrix = variables_encoding(Method_analysis, CONDITION, distance_ch, Subject_analysis, algorithm, root_use ) 
            #############################################
            df_responses=[]
            dfs = {}
            
            for session_enc in range(0,len(func_encoding_sess)):
                print(session_enc)
                func_encoding = func_encoding_sess[session_enc] 
                Beh_enc_files = Beh_enc_files_sess[session_enc]
                #
                func_wmtask =func_wmtask_sess[session_enc]
                Beh_WM_files = Beh_WM_files_sess[session_enc]
                
                ### Imaging encoding
                ##### 1. Imaging
                enc_lens_datas=[]
                encoding_datasets=[]
                
                
                #Data to use
                #Apply the mask
                
                for i in range(0, len(func_encoding)):
                    func_filename=func_encoding[i] #+ 'setfmri3_Encoding_Ax.nii' # 'regfmcpr.nii.gz'
                    func_filename = ub_wind_path(func_filename, system=sys_use)
                    #
                    mask_img_rh= path_masks  + Maskrh #maskV1rh_2.nii.gz' #maskV1rh.nii.gz'  maskV1rh_2.nii.gz maskipsrh_2.nii.gz
                    mask_img_rh = ub_wind_path(mask_img_rh, system=sys_use)
                    #
                    mask_img_lh= path_masks + Masklh #maskV1lh_2.nii.gz' #maskV1lh.nii.gz'   maskV1lh_2.nii.gz maskipslh_2.nii.gz
                    mask_img_lh = ub_wind_path(mask_img_lh, system=sys_use)
                    ##Apply the masks and concatenate   
                    masked_data_rh = apply_mask(func_filename, mask_img_rh)
                    masked_data_lh = apply_mask(func_filename, mask_img_lh)    
                    masked_data=hstack([masked_data_rh, masked_data_lh])
                    #append it and save the data
                    encoding_datasets.append(masked_data)
                    enc_lens_datas.append(len(masked_data))