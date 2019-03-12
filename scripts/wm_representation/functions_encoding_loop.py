# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 16:16:53 2018

@author: David
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 16:31:59 2018
@author: David Bestue
"""

#Required libraries 
from numpy import shape, hstack, mean, std, zeros, genfromtxt, arange, pi, cos, array, cumsum, save, load, vstack, linspace, radians, roll, where, degrees, transpose
from nilearn.masking import apply_mask
import matplotlib.pyplot as plt
from obspy.signal import filter
from matplotlib.patches import Ellipse
import pandas as pd
from statsmodels.formula.api import ols
from numpy import dot
from numpy.linalg import inv
import nitime
from nitime.timeseries import TimeSeries
from nitime.analysis import FilterAnalyzer
import seaborn
import seaborn as sns
from sklearn.linear_model import Ridge, Lasso
import statsmodels.api as sm
import easygui
from easygui import multenterbox
import os



def variables_encoding(Method_analysis, distance_ch, Subject_analysis, brain_region, root_use )  :  
    ############################################# Comment the following if using a loop #############################################
    
    ###### Decide method
    #msg = "Method"
    #choices = ["together", "bysess"]
    #Method_analysis = easygui.buttonbox(msg, choices=choices)
    #
    ###### Decide the condition
    #msg = "Condition"
    #choices = ['1_0.2', '1_7', '2_0.2', '2_7']
    #CONDITION = easygui.buttonbox(msg, choices=choices)
    #
    ###### Decide the distance
    #msg = "Distance"
    #choices = ['close', 'far', 'mix']
    #distance_ch = easygui.buttonbox(msg, choices=choices)
    if distance_ch=='close':
        distance=1
    elif distance_ch=='far':
        distance=3
    elif distance_ch=='mix':
        distance='mix'
    #
    #
    ###### Decide the subject
    #msg = "Subject"
    #choices = ["r001", "d001", "n001", "s001", "b001", "l001"]
    #Subject_analysis = easygui.buttonbox(msg, choices=choices)
    #
    ###Decide brain region
    #msg = "Brain region"
    #choices = ["visual", "ips"]
    #brain_region = easygui.buttonbox(msg, choices=choices)
    
    
    ############################################# Comment above if using a loop #############################################
    
    root = root_use #'/mnt/c/Users/David/Desktop/KI_Desktop/IEM_data/'
    
    #print(Method_analysis)
    ##### 0. Decide the brain region
    if Method_analysis == "together":
        
        if Subject_analysis == "d001":
            
            func_encoding_sess = [[root +'d001/encoding/s08/r01/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s08/r02/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s08/r03/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s08/r04/nocfmri3_Encoding_Ax.nii',
                             root +'d001/encoding/s09/r01/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s09/r02/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s09/r03/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s09/r04/nocfmri3_Encoding_Ax.nii',
                             root +'d001/encoding/s10/r01/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s10/r02/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s10/r03/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s10/r04/nocfmri3_Encoding_Ax.nii',
                             root +'d001/encoding/s11/r01/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s11/r02/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s11/r03/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s11/r04/nocfmri3_Encoding_Ax.nii']]
            
            
            Beh_enc_files_sess = [[root +'d001/encoding/s08/r01/enc_beh.txt', root +'d001/encoding/s08/r02/enc_beh.txt', root +'d001/encoding/s08/r03/enc_beh.txt', root +'d001/encoding/s08/r04/enc_beh.txt',
                             root +'d001/encoding/s09/r01/enc_beh.txt', root +'d001/encoding/s09/r02/enc_beh.txt', root +'d001/encoding/s09/r03/enc_beh.txt', root +'d001/encoding/s09/r04/enc_beh.txt',
                             root +'d001/encoding/s10/r01/enc_beh.txt', root +'d001/encoding/s10/r02/enc_beh.txt', root +'d001/encoding/s10/r03/enc_beh.txt', root +'d001/encoding/s10/r04/enc_beh.txt',
                             root +'d001/encoding/s11/r01/enc_beh.txt', root +'d001/encoding/s11/r02/enc_beh.txt', root +'d001/encoding/s11/r03/enc_beh.txt', root +'d001/encoding/s11/r04/enc_beh.txt']]
            
            
            
            func_wmtask_sess =  [[root +'d001/WMtask/s10/r01/nocfmri5_task_Ax.nii', root +'d001/WMtask/s10/r02/nocfmri5_task_Ax.nii', root +'d001/WMtask/s10/r03/nocfmri5_task_Ax.nii', root +'d001/WMtask/s10/r04/nocfmri5_task_Ax.nii', root +'d001/WMtask/s10/r05/nocfmri5_task_Ax.nii',
                             root +'d001/WMtask/s11/r01/nocfmri5_task_Ax.nii', root +'d001/WMtask/s11/r02/nocfmri5_task_Ax.nii', root +'d001/WMtask/s11/r03/nocfmri5_task_Ax.nii',
                             root +'d001/WMtask/s12/r01/nocfmri5_task_Ax.nii', root +'d001/WMtask/s12/r02/nocfmri5_task_Ax.nii',
                             root +'d001/WMtask/s13/r01/nocfmri5_task_Ax.nii', root +'d001/WMtask/s13/r02/nocfmri5_task_Ax.nii', root +'d001/WMtask/s13/r03/nocfmri5_task_Ax.nii']]
            
            
            Beh_WM_files_sess= [[root +'d001/WMtask/s10/r01/wm_beh.txt', root +'d001/WMtask/s10/r02/wm_beh.txt', root +'d001/WMtask/s10/r03/wm_beh.txt', root +'d001/WMtask/s10/r04/wm_beh.txt', root +'d001/WMtask/s10/r05/wm_beh.txt',
                             root +'d001/WMtask/s11/r01/wm_beh.txt', root +'d001/WMtask/s11/r02/wm_beh.txt', root +'d001/WMtask/s11/r03/wm_beh.txt',
                             root +'d001/WMtask/s12/r01/wm_beh.txt', root +'d001/WMtask/s12/r02/wm_beh.txt',
                             root +'d001/WMtask/s13/r01/wm_beh.txt', root +'d001/WMtask/s13/r02/wm_beh.txt', root +'d001/WMtask/s13/r03/wm_beh.txt']]
            
            
            
            ##Mask
            path_masks = root +  'temp_masks/d001/' 
            #Chose the brain_region        
            if brain_region=="V1V2":
                Maskrh = 'maskv1v2rh.nii.gz'
                Masklh = 'maskv1v2lh.nii.gz'
            
            elif brain_region=="visual":
                Maskrh = 'visual_fsign_rh.nii.gz'
                Masklh = 'visual_fsign_lh.nii.gz'
                writer_matrix=pd.ExcelWriter("/home/david/Desktop/KAROLINSKA/encoding_model/Matrix_encoding_model/d001_visual_together_matrix.xlsx")
                #df_name ='df_sessions_visual_d001.xlsx'
            
            elif brain_region=="ips":
                Maskrh = 'ips_str_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'ips_str_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                writer_matrix=pd.ExcelWriter("/home/david/Desktop/KAROLINSKA/encoding_model/Matrix_encoding_model/d001_ips_together_matrix.xlsx")
                #df_name ='df_sessions_ips_d001.xlsx'
            
            elif brain_region=="IPS":
                Maskrh = 'maskIPSrh.nii.gz'
                Masklh = 'maskIPSlh.nii.gz'
            
            elif brain_region=="IPS_1":
                Maskrh = 'maskIPS_1rh.nii.gz'
                Masklh = 'maskIPS_1lh.nii.gz'
        
        
        
        
        if Subject_analysis == "n001":
            
            func_encoding_sess = [[root +'n001/encoding/s01/r01/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s01/r02/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s01/r03/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s01/r04/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s01/r05/nocfmri3_Encoding_Ax.nii',
                                  root +'n001/encoding/s02/r01/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s02/r02/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s02/r03/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s02/r04/nocfmri3_Encoding_Ax.nii',
                                  root +'n001/encoding/s03/r01/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s03/r02/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s03/r03/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s03/r04/nocfmri3_Encoding_Ax.nii']]
            
            Beh_enc_files_sess =[[root +'n001/encoding/s01/r01/enc_beh.txt', root +'n001/encoding/s01/r02/enc_beh.txt', root +'n001/encoding/s01/r03/enc_beh.txt', root +'n001/encoding/s01/r04/enc_beh.txt', root +'n001/encoding/s01/r05/enc_beh.txt',
                                 root +'n001/encoding/s02/r01/enc_beh.txt', root +'n001/encoding/s02/r02/enc_beh.txt', root +'n001/encoding/s02/r03/enc_beh.txt', root +'n001/encoding/s02/r04/enc_beh.txt',
                                 root +'n001/encoding/s03/r01/enc_beh.txt', root +'n001/encoding/s03/r02/enc_beh.txt', root +'n001/encoding/s03/r03/enc_beh.txt', root +'n001/encoding/s03/r04/enc_beh.txt']]
            
            
            func_wmtask_sess = [[root +'n001/WMtask/s01/r01/nocfmri5_task_Ax.nii', root +'n001/WMtask/s01/r02/nocfmri5_task_Ax.nii', root +'n001/WMtask/s01/r03/nocfmri5_task_Ax.nii', root +'n001/WMtask/s01/r04/nocfmri5_task_Ax.nii', root +'n001/WMtask/s01/r05/nocfmri5_task_Ax.nii',
                                root +'n001/WMtask/s02/r01/nocfmri5_task_Ax.nii', root +'n001/WMtask/s02/r02/nocfmri5_task_Ax.nii', root +'n001/WMtask/s02/r03/nocfmri5_task_Ax.nii', root +'n001/WMtask/s02/r04/nocfmri5_task_Ax.nii',
                                root +'n001/WMtask/s03/r01/nocfmri5_task_Ax.nii', root +'n001/WMtask/s03/r02/nocfmri5_task_Ax.nii', root +'n001/WMtask/s03/r03/nocfmri5_task_Ax.nii', root +'n001/WMtask/s03/r04/nocfmri5_task_Ax.nii', root +'n001/WMtask/s03/r05/nocfmri5_task_Ax.nii']]
            
            
            Beh_WM_files_sess=[[root +'n001/WMtask/s01/r01/wm_beh.txt', root +'n001/WMtask/s01/r02/wm_beh.txt', root +'n001/WMtask/s01/r03/wm_beh.txt', root +'n001/WMtask/s01/r04/wm_beh.txt', root +'n001/WMtask/s01/r05/wm_beh.txt',
                                root +'n001/WMtask/s02/r01/wm_beh.txt', root +'n001/WMtask/s02/r02/wm_beh.txt', root +'n001/WMtask/s02/r03/wm_beh.txt', root +'n001/WMtask/s02/r04/wm_beh.txt',
                                root +'n001/WMtask/s03/r01/wm_beh.txt', root +'n001/WMtask/s03/r02/wm_beh.txt', root +'n001/WMtask/s03/r03/wm_beh.txt', root +'n001/WMtask/s03/r04/wm_beh.txt', root +'n001/WMtask/s03/r05/wm_beh.txt']]
            
            
            
            path_masks = root +  'temp_masks/n001/'
            
            #Chose the brain_region        
            if brain_region=="visual":
                Maskrh = 'visual_fsign_rh.nii.gz'
                Masklh = 'visual_fsign_lh.nii.gz'
                writer_matrix=pd.ExcelWriter("/home/david/Desktop/KAROLINSKA/encoding_model/Matrix_encoding_model/n001_visual_together_matrix.xlsx")
                #df_name ='df_sessions_visual_n001.xlsx'
            
            elif brain_region=="ips":
                Maskrh = 'ips_str_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'ips_str_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                writer_matrix=pd.ExcelWriter("/home/david/Desktop/KAROLINSKA/encoding_model/Matrix_encoding_model/n001_ips_together_matrix.xlsx")
            
            
            
        
        
        if Subject_analysis == "b001":
            
            func_encoding_sess = [[root +'b001/encoding/s01/r01/nocfmri3_Encoding_Ax.nii', root +'b001/encoding/s01/r02/nocfmri3_Encoding_Ax.nii', root +'b001/encoding/s01/r03/nocfmri3_Encoding_Ax.nii', root +'b001/encoding/s01/r04/nocfmri3_Encoding_Ax.nii',
                                  root +'b001/encoding/s02/r01/nocfmri3_Encoding_Ax.nii', root +'b001/encoding/s02/r02/nocfmri3_Encoding_Ax.nii', root +'b001/encoding/s02/r03/nocfmri3_Encoding_Ax.nii', root +'b001/encoding/s02/r04/nocfmri3_Encoding_Ax.nii',
                                  root +'b001/encoding/s03/r01/nocfmri3_Encoding_Ax.nii', root +'b001/encoding/s03/r02/nocfmri3_Encoding_Ax.nii', root +'b001/encoding/s03/r03/nocfmri3_Encoding_Ax.nii', root +'b001/encoding/s03/r04/nocfmri3_Encoding_Ax.nii']]
            
            Beh_enc_files_sess =[[root +'b001/encoding/s01/r01/enc_beh.txt', root +'b001/encoding/s01/r02/enc_beh.txt', root +'b001/encoding/s01/r03/enc_beh.txt', root +'b001/encoding/s01/r04/enc_beh.txt',
                                  root +'b001/encoding/s02/r01/enc_beh.txt', root +'b001/encoding/s02/r02/enc_beh.txt', root +'b001/encoding/s02/r03/enc_beh.txt', root +'b001/encoding/s02/r04/enc_beh.txt',
                                  root +'b001/encoding/s03/r01/enc_beh.txt', root +'b001/encoding/s03/r02/enc_beh.txt', root +'b001/encoding/s03/r03/enc_beh.txt', root +'b001/encoding/s03/r04/enc_beh.txt']]
            
            
            func_wmtask_sess = [[root +'b001/WMtask/s01/r01/nocfmri5_task_Ax.nii', root +'b001/WMtask/s01/r02/nocfmri5_task_Ax.nii', root +'b001/WMtask/s01/r03/nocfmri5_task_Ax.nii', root +'b001/WMtask/s01/r04/nocfmri5_task_Ax.nii', root +'b001/WMtask/s01/r05/nocfmri5_task_Ax.nii',
                                root +'b001/WMtask/s02/r01/nocfmri5_task_Ax.nii', root +'b001/WMtask/s02/r02/nocfmri5_task_Ax.nii', root +'b001/WMtask/s02/r03/nocfmri5_task_Ax.nii', root +'b001/WMtask/s02/r04/nocfmri5_task_Ax.nii', root +'b001/WMtask/s02/r05/nocfmri5_task_Ax.nii',
                                root +'b001/WMtask/s03/r01/nocfmri5_task_Ax.nii', root +'b001/WMtask/s03/r02/nocfmri5_task_Ax.nii', root +'b001/WMtask/s03/r03/nocfmri5_task_Ax.nii', root +'b001/WMtask/s03/r04/nocfmri5_task_Ax.nii']]
            
            
            Beh_WM_files_sess=[[root +'b001/WMtask/s01/r01/wm_beh.txt', root +'b001/WMtask/s01/r02/wm_beh.txt', root +'b001/WMtask/s01/r03/wm_beh.txt', root +'b001/WMtask/s01/r04/wm_beh.txt', root +'b001/WMtask/s01/r05/wm_beh.txt',
                                root +'b001/WMtask/s02/r01/wm_beh.txt', root +'b001/WMtask/s02/r02/wm_beh.txt', root +'b001/WMtask/s02/r03/wm_beh.txt', root +'b001/WMtask/s02/r04/wm_beh.txt', root +'b001/WMtask/s02/r05/wm_beh.txt',
                                root +'b001/WMtask/s03/r01/wm_beh.txt', root +'b001/WMtask/s03/r02/wm_beh.txt', root +'b001/WMtask/s03/r03/wm_beh.txt', root +'b001/WMtask/s03/r04/wm_beh.txt']]
            
            
            path_masks =  root +  'temp_masks/b001/'
            
            #Chose the brain_region        
            if brain_region=="visual":
                Maskrh = 'visual_fsign_rh.nii.gz'
                Masklh = 'visual_fsign_lh.nii.gz'
                writer_matrix=pd.ExcelWriter("/home/david/Desktop/KAROLINSKA/encoding_model/Matrix_encoding_model/b001_visual_together_matrix.xlsx")
                #df_name ='df_sessions_visual_b001.xlsx'
            
            elif brain_region=="ips":
                Maskrh = 'ips_str_rh.nii.gz' #'ips_fsign_rh.nii.gz'
                Masklh = 'ips_str_lh.nii.gz' #'ips_fsign_lh.nii.gz'
                writer_matrix=pd.ExcelWriter("/home/david/Desktop/KAROLINSKA/encoding_model/Matrix_encoding_model/b001_ips_together_matrix.xlsx")
                #df_name ='df_sessions_ips_b001.xlsx'
                
        
        
        if Subject_analysis == "l001":
            
            func_encoding_sess = [[root +'l001/encoding/s01/r01/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s01/r02/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s01/r03/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s01/r04/nocfmri3_Encoding_Ax.nii',
                                  root +'l001/encoding/s02/r01/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s02/r02/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s02/r03/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s02/r04/nocfmri3_Encoding_Ax.nii',
                                  root +'l001/encoding/s03/r01/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s03/r02/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s03/r03/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s03/r04/nocfmri3_Encoding_Ax.nii',
                                  root +'l001/encoding/s04/r01/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s04/r02/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s04/r03/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s04/r04/nocfmri3_Encoding_Ax.nii']]
            
            
            Beh_enc_files_sess =[[root +'l001/encoding/s01/r01/enc_beh.txt', root +'l001/encoding/s01/r02/enc_beh.txt', root +'l001/encoding/s01/r03/enc_beh.txt', root +'l001/encoding/s01/r04/enc_beh.txt',
                                  root +'l001/encoding/s02/r01/enc_beh.txt', root +'l001/encoding/s02/r02/enc_beh.txt', root +'l001/encoding/s02/r03/enc_beh.txt', root +'l001/encoding/s02/r04/enc_beh.txt',
                                  root +'l001/encoding/s03/r01/enc_beh.txt', root +'l001/encoding/s03/r02/enc_beh.txt', root +'l001/encoding/s03/r03/enc_beh.txt', root +'l001/encoding/s03/r04/enc_beh.txt',
                                  root +'l001/encoding/s04/r01/enc_beh.txt', root +'l001/encoding/s04/r02/enc_beh.txt', root +'l001/encoding/s04/r03/enc_beh.txt', root +'l001/encoding/s04/r04/enc_beh.txt']]
            
            func_wmtask_sess = [[root +'l001/WMtask/s01/r01/nocfmri5_task_Ax.nii', root +'l001/WMtask/s01/r02/nocfmri5_task_Ax.nii', root +'l001/WMtask/s01/r03/nocfmri5_task_Ax.nii',
                                root +'l001/WMtask/s02/r01/nocfmri5_task_Ax.nii', root +'l001/WMtask/s02/r02/nocfmri5_task_Ax.nii', root +'l001/WMtask/s02/r03/nocfmri5_task_Ax.nii',
                                root +'l001/WMtask/s03/r01/nocfmri5_task_Ax.nii', root +'l001/WMtask/s03/r02/nocfmri5_task_Ax.nii', root +'l001/WMtask/s03/r03/nocfmri5_task_Ax.nii', root +'l001/WMtask/s03/r04/nocfmri5_task_Ax.nii',
                                root +'l001/WMtask/s04/r01/nocfmri5_task_Ax.nii', root +'l001/WMtask/s04/r02/nocfmri5_task_Ax.nii', root +'l001/WMtask/s04/r03/nocfmri5_task_Ax.nii', root +'l001/WMtask/s04/r04/nocfmri5_task_Ax.nii']]
            
            
            Beh_WM_files_sess=[[root +'l001/WMtask/s01/r01/wm_beh.txt', root +'l001/WMtask/s01/r02/wm_beh.txt', root +'l001/WMtask/s01/r03/wm_beh.txt',
                                root +'l001/WMtask/s02/r01/wm_beh.txt', root +'l001/WMtask/s02/r02/wm_beh.txt', root +'l001/WMtask/s02/r03/wm_beh.txt',
                                root +'l001/WMtask/s03/r01/wm_beh.txt', root +'l001/WMtask/s03/r02/wm_beh.txt', root +'l001/WMtask/s03/r03/wm_beh.txt', root +'l001/WMtask/s03/r04/wm_beh.txt',
                                root +'l001/WMtask/s04/r01/wm_beh.txt', root +'l001/WMtask/s04/r02/wm_beh.txt', root +'l001/WMtask/s04/r03/wm_beh.txt', root +'l001/WMtask/s04/r04/wm_beh.txt']]
            
            
            
            path_masks = root +  'temp_masks/l001/'
            
            #Chose the brain_region
            if brain_region=="visual":
                Maskrh = 'visual_fsign_rh.nii.gz'
                Masklh = 'visual_fsign_lh.nii.gz'
                writer_matrix=pd.ExcelWriter("/home/david/Desktop/KAROLINSKA/encoding_model/Matrix_encoding_model/l001_visual_together_matrix.xlsx")
                #df_name ='df_sessions_visual_l001.xlsx'
            
            elif brain_region=="ips":
                Maskrh = 'ips_str_rh.nii.gz' #'ips_fsign_rh.nii.gz'
                Masklh = 'ips_str_lh.nii.gz' #'ips_fsign_lh.nii.gz'
                writer_matrix=pd.ExcelWriter("/home/david/Desktop/KAROLINSKA/encoding_model/Matrix_encoding_model/l001_ips_together_matrix.xlsx")
                #df_name ='df_sessions_ips_l001.xlsx'
                
        
        
        
        
        if Subject_analysis == "s001":
            
            func_encoding_sess = [[root +'s001/encoding/s01/r01/nocfmri3_Encoding_Ax.nii', root +'s001/encoding/s01/r02/nocfmri3_Encoding_Ax.nii',
                                  root +'s001/encoding/s02/r01/nocfmri3_Encoding_Ax.nii', root +'s001/encoding/s02/r02/nocfmri3_Encoding_Ax.nii',
                                  root +'s001/encoding/s03/r01/nocfmri3_Encoding_Ax.nii', root +'s001/encoding/s03/r02/nocfmri3_Encoding_Ax.nii', root +'s001/encoding/s03/r03/nocfmri3_Encoding_Ax.nii', root +'s001/encoding/s03/r04/nocfmri3_Encoding_Ax.nii',
                                  root +'s001/encoding/s04/r01/nocfmri3_Encoding_Ax.nii', root +'s001/encoding/s04/r02/nocfmri3_Encoding_Ax.nii',
                                  root +'s001/encoding/s05/r01/nocfmri3_Encoding_Ax.nii', root +'s001/encoding/s05/r02/nocfmri3_Encoding_Ax.nii']]
            
            
            Beh_enc_files_sess =[[root +'s001/encoding/s01/r01/enc_beh.txt', root +'s001/encoding/s01/r02/enc_beh.txt',
                                  root +'s001/encoding/s02/r01/enc_beh.txt', root +'s001/encoding/s02/r02/enc_beh.txt',
                                  root +'s001/encoding/s03/r01/enc_beh.txt', root +'s001/encoding/s03/r02/enc_beh.txt', root +'s001/encoding/s03/r03/enc_beh.txt', root +'s001/encoding/s03/r04/enc_beh.txt',
                                  root +'s001/encoding/s04/r01/enc_beh.txt', root +'s001/encoding/s04/r02/enc_beh.txt',
                                  root +'s001/encoding/s05/r01/enc_beh.txt', root +'s001/encoding/s05/r02/enc_beh.txt']]
            
            func_wmtask_sess = [[root +'s001/WMtask/s01/r01/nocfmri5_task_Ax.nii', root +'s001/WMtask/s01/r02/nocfmri5_task_Ax.nii',
                                root +'s001/WMtask/s02/r01/nocfmri5_task_Ax.nii',
                                root +'s001/WMtask/s03/r01/nocfmri5_task_Ax.nii', root +'s001/WMtask/s03/r02/nocfmri5_task_Ax.nii', root +'s001/WMtask/s03/r03/nocfmri5_task_Ax.nii', root +'s001/WMtask/s03/r04/nocfmri5_task_Ax.nii', root +'s001/WMtask/s03/r05/nocfmri5_task_Ax.nii',
                                root +'s001/WMtask/s04/r01/nocfmri5_task_Ax.nii', root +'s001/WMtask/s04/r02/nocfmri5_task_Ax.nii',
                                root +'s001/WMtask/s05/r01/nocfmri5_task_Ax.nii', root +'s001/WMtask/s05/r02/nocfmri5_task_Ax.nii']]
            
            
            Beh_WM_files_sess=[[root +'s001/WMtask/s01/r01/wm_beh.txt', root +'s001/WMtask/s01/r02/wm_beh.txt',
                                root +'s001/WMtask/s02/r01/wm_beh.txt',
                                root +'s001/WMtask/s03/r01/wm_beh.txt', root +'s001/WMtask/s03/r02/wm_beh.txt', root +'s001/WMtask/s03/r03/wm_beh.txt', root +'s001/WMtask/s03/r04/wm_beh.txt', root +'s001/WMtask/s03/r05/wm_beh.txt',
                                root +'s001/WMtask/s04/r01/wm_beh.txt', root +'s001/WMtask/s04/r02/wm_beh.txt',
                                root +'s001/WMtask/s05/r01/wm_beh.txt', root +'s001/WMtask/s05/r02/wm_beh.txt']]
            
            
            
            path_masks =  root +  'temp_masks/s001/'
            
            #Chose the brain_region        
            if brain_region=="visual":
                Maskrh = 'visual_fsign_rh.nii.gz'
                Masklh = 'visual_fsign_lh.nii.gz'
                writer_matrix=pd.ExcelWriter("/home/david/Desktop/KAROLINSKA/encoding_model/Matrix_encoding_model/s001_visual_together_matrix.xlsx")
                #df_name ='df_sessions_visual_s001.xlsx'
            
            elif brain_region=="ips":
                Maskrh = 'ips_str_rh.nii.gz' #'ips_fsign_rh.nii.gz'
                Masklh = 'ips_str_lh.nii.gz' #'ips_fsign_lh.nii.gz'
                writer_matrix=pd.ExcelWriter("/home/david/Desktop/KAROLINSKA/encoding_model/Matrix_encoding_model/s001_ips_together_matrix.xlsx")
                #df_name ='df_sessions_ips_s001.xlsx'
            
            
            
        if Subject_analysis == "r001":
            
            func_encoding_sess = [[root +'r001/encoding/s05/r01/nocfmri3_Encoding_Ax.nii', root +'r001/encoding/s05/r02/nocfmri3_Encoding_Ax.nii', root +'r001/encoding/s05/r03/nocfmri3_Encoding_Ax.nii', root +'r001/encoding/s05/r04/nocfmri3_Encoding_Ax.nii',
                                  root +'r001/encoding/s06/r01/nocfmri3_Encoding_Ax.nii', root +'r001/encoding/s06/r02/nocfmri3_Encoding_Ax.nii', root +'r001/encoding/s06/r03/nocfmri3_Encoding_Ax.nii', root +'r001/encoding/s06/r04/nocfmri3_Encoding_Ax.nii',
                                  root +'r001/encoding/s07/r01/nocfmri3_Encoding_Ax.nii', root +'r001/encoding/s07/r02/nocfmri3_Encoding_Ax.nii', root +'r001/encoding/s07/r03/nocfmri3_Encoding_Ax.nii', root +'r001/encoding/s07/r04/nocfmri3_Encoding_Ax.nii']]
            
            
            Beh_enc_files_sess =[[root +'r001/encoding/s05/r01/enc_beh.txt', root +'r001/encoding/s05/r02/enc_beh.txt', root +'r001/encoding/s05/r03/enc_beh.txt', root +'r001/encoding/s05/r04/enc_beh.txt',
                                  root +'r001/encoding/s06/r01/enc_beh.txt', root +'r001/encoding/s06/r02/enc_beh.txt', root +'r001/encoding/s06/r03/enc_beh.txt', root +'r001/encoding/s06/r04/enc_beh.txt',
                                  root +'r001/encoding/s07/r01/enc_beh.txt', root +'r001/encoding/s07/r02/enc_beh.txt', root +'r001/encoding/s07/r03/enc_beh.txt', root +'r001/encoding/s07/r04/enc_beh.txt']]
            
                        
            func_wmtask_sess = [[root +'r001/WMtask/s08/r01/nocfmri5_task_Ax.nii', root +'r001/WMtask/s08/r02/nocfmri5_task_Ax.nii', root +'r001/WMtask/s08/r03/nocfmri5_task_Ax.nii', root +'r001/WMtask/s08/r04/nocfmri5_task_Ax.nii', root +'r001/WMtask/s08/r05/nocfmri5_task_Ax.nii', root +'r001/WMtask/s08/r06/nocfmri5_task_Ax.nii',
                                root +'r001/WMtask/s09/r01/nocfmri5_task_Ax.nii', root +'r001/WMtask/s09/r02/nocfmri5_task_Ax.nii', root +'r001/WMtask/s09/r03/nocfmri5_task_Ax.nii', root +'r001/WMtask/s09/r04/nocfmri5_task_Ax.nii',
                                root +'r001/WMtask/s10/r01/nocfmri5_task_Ax.nii']]
            
            
            Beh_WM_files_sess=[[root +'r001/WMtask/s08/r01/wm_beh.txt', root +'r001/WMtask/s08/r02/wm_beh.txt', root +'r001/WMtask/s08/r03/wm_beh.txt', root +'r001/WMtask/s08/r04/wm_beh.txt', root +'r001/WMtask/s08/r05/wm_beh.txt', root +'r001/WMtask/s08/r06/wm_beh.txt',
                                root +'r001/WMtask/s09/r01/wm_beh.txt', root +'r001/WMtask/s09/r02/wm_beh.txt', root +'r001/WMtask/s09/r03/wm_beh.txt', root +'r001/WMtask/s09/r04/wm_beh.txt',
                                root +'r001/WMtask/s10/r01/wm_beh.txt']]
            
            
            
            path_masks = root +  'temp_masks/r001/'
            
            #Chose the brain_region        
            if brain_region=="visual":
                Maskrh = 'visual_fsign_rh.nii.gz'
                Masklh = 'visual_fsign_lh.nii.gz'
                writer_matrix=pd.ExcelWriter("/home/david/Desktop/KAROLINSKA/encoding_model/Matrix_encoding_model/r001_visual_together_matrix.xlsx")
                #df_name ='df_sessions_visual_r001.xlsx'
            
            elif brain_region=="ips":
                Maskrh = 'ips_str_rh.nii.gz' #'ips_fsign_rh.nii.gz'
                Masklh = 'ips_str_lh.nii.gz' #'ips_fsign_lh.nii.gz'
                writer_matrix=pd.ExcelWriter("/home/david/Desktop/KAROLINSKA/encoding_model/Matrix_encoding_model/r001_ips_together_matrix.xlsx")
                #df_name ='df_sessions_ips_r001.xlsx'    
    
    
    elif Method_analysis == "bysess":    
        
        #root = '/mnt/c/Users/David/Desktop/KI_Desktop/IEM_data/'
        if Subject_analysis == "d001":  
            
            func_encoding_sess = [[root +'d001/encoding/s08/r01/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s08/r02/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s08/r03/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s08/r04/nocfmri3_Encoding_Ax.nii'],
                             [root +'d001/encoding/s09/r01/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s09/r02/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s09/r03/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s09/r04/nocfmri3_Encoding_Ax.nii'],
                             [root +'d001/encoding/s10/r01/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s10/r02/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s10/r03/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s10/r04/nocfmri3_Encoding_Ax.nii'],
                             [root +'d001/encoding/s11/r01/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s11/r02/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s11/r03/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s11/r04/nocfmri3_Encoding_Ax.nii']]
            
            
            Beh_enc_files_sess = [[root +'d001/encoding/s08/r01/enc_beh.txt', root +'d001/encoding/s08/r02/enc_beh.txt', root +'d001/encoding/s08/r03/enc_beh.txt', root +'d001/encoding/s08/r04/enc_beh.txt'],
                             [root +'d001/encoding/s09/r01/enc_beh.txt', root +'d001/encoding/s09/r02/enc_beh.txt', root +'d001/encoding/s09/r03/enc_beh.txt', root +'d001/encoding/s09/r04/enc_beh.txt'],
                             [root +'d001/encoding/s10/r01/enc_beh.txt', root +'d001/encoding/s10/r02/enc_beh.txt', root +'d001/encoding/s10/r03/enc_beh.txt', root +'d001/encoding/s10/r04/enc_beh.txt'],
                             [root +'d001/encoding/s11/r01/enc_beh.txt', root +'d001/encoding/s11/r02/enc_beh.txt', root +'d001/encoding/s11/r03/enc_beh.txt', root +'d001/encoding/s11/r04/enc_beh.txt']]
            
            
            
            func_wmtask_sess =  [[root +'d001/WMtask/s10/r01/nocfmri5_task_Ax.nii', root +'d001/WMtask/s10/r02/nocfmri5_task_Ax.nii', root +'d001/WMtask/s10/r03/nocfmri5_task_Ax.nii', root +'d001/WMtask/s10/r04/nocfmri5_task_Ax.nii', root +'d001/WMtask/s10/r05/nocfmri5_task_Ax.nii'],
                             [root +'d001/WMtask/s11/r01/nocfmri5_task_Ax.nii', root +'d001/WMtask/s11/r02/nocfmri5_task_Ax.nii', root +'d001/WMtask/s11/r03/nocfmri5_task_Ax.nii'],
                             [root +'d001/WMtask/s12/r01/nocfmri5_task_Ax.nii', root +'d001/WMtask/s12/r02/nocfmri5_task_Ax.nii'],
                             [root +'d001/WMtask/s13/r01/nocfmri5_task_Ax.nii', root +'d001/WMtask/s13/r02/nocfmri5_task_Ax.nii', root +'d001/WMtask/s13/r03/nocfmri5_task_Ax.nii']]
            
            
            Beh_WM_files_sess= [[root +'d001/WMtask/s10/r01/wm_beh.txt', root +'d001/WMtask/s10/r02/wm_beh.txt', root +'d001/WMtask/s10/r03/wm_beh.txt', root +'d001/WMtask/s10/r04/wm_beh.txt', root +'d001/WMtask/s10/r05/wm_beh.txt'],
                             [root +'d001/WMtask/s11/r01/wm_beh.txt', root +'d001/WMtask/s11/r02/wm_beh.txt', root +'d001/WMtask/s11/r03/wm_beh.txt'],
                             [root +'d001/WMtask/s12/r01/wm_beh.txt', root +'d001/WMtask/s12/r02/wm_beh.txt'],
                             [root +'d001/WMtask/s13/r01/wm_beh.txt', root +'d001/WMtask/s13/r02/wm_beh.txt', root +'d001/WMtask/s13/r03/wm_beh.txt']]
            
            
            
            ##Mask
            path_masks = root +  'temp_masks/d001/' 
            #Chose the brain_region        
            if brain_region=="V1V2":
                Maskrh = 'maskv1v2rh.nii.gz'
                Masklh = 'maskv1v2lh.nii.gz'
            
            elif brain_region=="visual":
                Maskrh = 'visual_fsign_rh.nii.gz'
                Masklh = 'visual_fsign_lh.nii.gz'
                writer_matrix=pd.ExcelWriter("/home/david/Desktop/KAROLINSKA/encoding_model/Matrix_encoding_model/d001_visual_bysess_matrix.xlsx")
                #df_name ='df_sessions_visual_d001.xlsx'
            
            elif brain_region=="ips":
                Maskrh = 'ips_str_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'ips_str_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                writer_matrix=pd.ExcelWriter("/home/david/Desktop/KAROLINSKA/encoding_model/Matrix_encoding_model/d001_ips_bysess_matrix.xlsx")
                #df_name ='df_sessions_ips_d001.xlsx'
            
            elif brain_region=="IPS":
                Maskrh = 'maskIPSrh.nii.gz'
                Masklh = 'maskIPSlh.nii.gz'
            
            elif brain_region=="IPS_1":
                Maskrh = 'maskIPS_1rh.nii.gz'
                Masklh = 'maskIPS_1lh.nii.gz'
        
        
        
        
        if Subject_analysis == "n001":
            
            func_encoding_sess = [[root +'n001/encoding/s01/r01/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s01/r02/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s01/r03/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s01/r04/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s01/r05/nocfmri3_Encoding_Ax.nii'],
                                  [root +'n001/encoding/s02/r01/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s02/r02/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s02/r03/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s02/r04/nocfmri3_Encoding_Ax.nii'],
                                  [root +'n001/encoding/s03/r01/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s03/r02/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s03/r03/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s03/r04/nocfmri3_Encoding_Ax.nii']]
            
            Beh_enc_files_sess =[[root +'n001/encoding/s01/r01/enc_beh.txt', root +'n001/encoding/s01/r02/enc_beh.txt', root +'n001/encoding/s01/r03/enc_beh.txt', root +'n001/encoding/s01/r04/enc_beh.txt', root +'n001/encoding/s01/r05/enc_beh.txt'],
                                 [root +'n001/encoding/s02/r01/enc_beh.txt', root +'n001/encoding/s02/r02/enc_beh.txt', root +'n001/encoding/s02/r03/enc_beh.txt', root +'n001/encoding/s02/r04/enc_beh.txt'],
                                 [root +'n001/encoding/s03/r01/enc_beh.txt', root +'n001/encoding/s03/r02/enc_beh.txt', root +'n001/encoding/s03/r03/enc_beh.txt', root +'n001/encoding/s03/r04/enc_beh.txt']]
            
            
            func_wmtask_sess = [[root +'n001/WMtask/s01/r01/nocfmri5_task_Ax.nii', root +'n001/WMtask/s01/r02/nocfmri5_task_Ax.nii', root +'n001/WMtask/s01/r03/nocfmri5_task_Ax.nii', root +'n001/WMtask/s01/r04/nocfmri5_task_Ax.nii', root +'n001/WMtask/s01/r05/nocfmri5_task_Ax.nii'],
                                [root +'n001/WMtask/s02/r01/nocfmri5_task_Ax.nii', root +'n001/WMtask/s02/r02/nocfmri5_task_Ax.nii', root +'n001/WMtask/s02/r03/nocfmri5_task_Ax.nii', root +'n001/WMtask/s02/r04/nocfmri5_task_Ax.nii'],
                                [root +'n001/WMtask/s03/r01/nocfmri5_task_Ax.nii', root +'n001/WMtask/s03/r02/nocfmri5_task_Ax.nii', root +'n001/WMtask/s03/r03/nocfmri5_task_Ax.nii', root +'n001/WMtask/s03/r04/nocfmri5_task_Ax.nii', root +'n001/WMtask/s03/r05/nocfmri5_task_Ax.nii']]
            
            
            Beh_WM_files_sess=[[root +'n001/WMtask/s01/r01/wm_beh.txt', root +'n001/WMtask/s01/r02/wm_beh.txt', root +'n001/WMtask/s01/r03/wm_beh.txt', root +'n001/WMtask/s01/r04/wm_beh.txt', root +'n001/WMtask/s01/r05/wm_beh.txt'],
                                [root +'n001/WMtask/s02/r01/wm_beh.txt', root +'n001/WMtask/s02/r02/wm_beh.txt', root +'n001/WMtask/s02/r03/wm_beh.txt', root +'n001/WMtask/s02/r04/wm_beh.txt'],
                                [root +'n001/WMtask/s03/r01/wm_beh.txt', root +'n001/WMtask/s03/r02/wm_beh.txt', root +'n001/WMtask/s03/r03/wm_beh.txt', root +'n001/WMtask/s03/r04/wm_beh.txt', root +'n001/WMtask/s03/r05/wm_beh.txt']]
            
            
            
            path_masks = root +  'temp_masks/n001/'
            
            #Chose the brain_region        
            if brain_region=="visual":
                Maskrh = 'visual_fsign_rh.nii.gz'
                Masklh = 'visual_fsign_lh.nii.gz'
                writer_matrix=pd.ExcelWriter("/home/david/Desktop/KAROLINSKA/encoding_model/Matrix_encoding_model/n001_visual_bysess_matrix.xlsx")
                #df_name ='df_sessions_visual_n001.xlsx'
            
            elif brain_region=="ips":
                Maskrh = 'ips_str_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'ips_str_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                writer_matrix=pd.ExcelWriter("/home/david/Desktop/KAROLINSKA/encoding_model/Matrix_encoding_model/n001_ips_bysess_matrix.xlsx")
            
            
            
        
        
        if Subject_analysis == "b001":
            
            func_encoding_sess = [[root +'b001/encoding/s01/r01/nocfmri3_Encoding_Ax.nii', root +'b001/encoding/s01/r02/nocfmri3_Encoding_Ax.nii', root +'b001/encoding/s01/r03/nocfmri3_Encoding_Ax.nii', root +'b001/encoding/s01/r04/nocfmri3_Encoding_Ax.nii'],
                                  [root +'b001/encoding/s02/r01/nocfmri3_Encoding_Ax.nii', root +'b001/encoding/s02/r02/nocfmri3_Encoding_Ax.nii', root +'b001/encoding/s02/r03/nocfmri3_Encoding_Ax.nii', root +'b001/encoding/s02/r04/nocfmri3_Encoding_Ax.nii'],
                                  [root +'b001/encoding/s03/r01/nocfmri3_Encoding_Ax.nii', root +'b001/encoding/s03/r02/nocfmri3_Encoding_Ax.nii', root +'b001/encoding/s03/r03/nocfmri3_Encoding_Ax.nii', root +'b001/encoding/s03/r04/nocfmri3_Encoding_Ax.nii']]
            
            Beh_enc_files_sess =[[root +'b001/encoding/s01/r01/enc_beh.txt', root +'b001/encoding/s01/r02/enc_beh.txt', root +'b001/encoding/s01/r03/enc_beh.txt', root +'b001/encoding/s01/r04/enc_beh.txt'],
                                  [root +'b001/encoding/s02/r01/enc_beh.txt', root +'b001/encoding/s02/r02/enc_beh.txt', root +'b001/encoding/s02/r03/enc_beh.txt', root +'b001/encoding/s02/r04/enc_beh.txt'],
                                  [root +'b001/encoding/s03/r01/enc_beh.txt', root +'b001/encoding/s03/r02/enc_beh.txt', root +'b001/encoding/s03/r03/enc_beh.txt', root +'b001/encoding/s03/r04/enc_beh.txt']]
            
            
            func_wmtask_sess = [[root +'b001/WMtask/s01/r01/nocfmri5_task_Ax.nii', root +'b001/WMtask/s01/r02/nocfmri5_task_Ax.nii', root +'b001/WMtask/s01/r03/nocfmri5_task_Ax.nii', root +'b001/WMtask/s01/r04/nocfmri5_task_Ax.nii', root +'b001/WMtask/s01/r05/nocfmri5_task_Ax.nii'],
                                [root +'b001/WMtask/s02/r01/nocfmri5_task_Ax.nii', root +'b001/WMtask/s02/r02/nocfmri5_task_Ax.nii', root +'b001/WMtask/s02/r03/nocfmri5_task_Ax.nii', root +'b001/WMtask/s02/r04/nocfmri5_task_Ax.nii', root +'b001/WMtask/s02/r05/nocfmri5_task_Ax.nii'],
                                [root +'b001/WMtask/s03/r01/nocfmri5_task_Ax.nii', root +'b001/WMtask/s03/r02/nocfmri5_task_Ax.nii', root +'b001/WMtask/s03/r03/nocfmri5_task_Ax.nii', root +'b001/WMtask/s03/r04/nocfmri5_task_Ax.nii']]
            
            
            Beh_WM_files_sess=[[root +'b001/WMtask/s01/r01/wm_beh.txt', root +'b001/WMtask/s01/r02/wm_beh.txt', root +'b001/WMtask/s01/r03/wm_beh.txt', root +'b001/WMtask/s01/r04/wm_beh.txt', root +'b001/WMtask/s01/r05/wm_beh.txt'],
                                [root +'b001/WMtask/s02/r01/wm_beh.txt', root +'b001/WMtask/s02/r02/wm_beh.txt', root +'b001/WMtask/s02/r03/wm_beh.txt', root +'b001/WMtask/s02/r04/wm_beh.txt', root +'b001/WMtask/s02/r05/wm_beh.txt'],
                                [root +'b001/WMtask/s03/r01/wm_beh.txt', root +'b001/WMtask/s03/r02/wm_beh.txt', root +'b001/WMtask/s03/r03/wm_beh.txt', root +'b001/WMtask/s03/r04/wm_beh.txt']]
            
            
            path_masks =  root +  'temp_masks/b001/'
            
            #Chose the brain_region        
            if brain_region=="visual":
                Maskrh = 'visual_fsign_rh.nii.gz'
                Masklh = 'visual_fsign_lh.nii.gz'
                writer_matrix=pd.ExcelWriter("/home/david/Desktop/KAROLINSKA/encoding_model/Matrix_encoding_model/b001_visual_bysess_matrix.xlsx")
                #df_name ='df_sessions_visual_b001.xlsx'
            
            elif brain_region=="ips":
                Maskrh = 'ips_str_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'ips_str_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                writer_matrix=pd.ExcelWriter("/home/david/Desktop/KAROLINSKA/encoding_model/Matrix_encoding_model/b001_ips_bysess_matrix.xlsx")
                #df_name ='df_sessions_ips_b001.xlsx'
                
        
        
        if Subject_analysis == "l001":
            
            func_encoding_sess = [[root +'l001/encoding/s01/r01/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s01/r02/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s01/r03/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s01/r04/nocfmri3_Encoding_Ax.nii'],
                                  [root +'l001/encoding/s02/r01/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s02/r02/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s02/r03/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s02/r04/nocfmri3_Encoding_Ax.nii'],
                                  [root +'l001/encoding/s03/r01/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s03/r02/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s03/r03/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s03/r04/nocfmri3_Encoding_Ax.nii'],
                                  [root +'l001/encoding/s04/r01/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s04/r02/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s04/r03/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s04/r04/nocfmri3_Encoding_Ax.nii']]
            
            
            Beh_enc_files_sess =[[root +'l001/encoding/s01/r01/enc_beh.txt', root +'l001/encoding/s01/r02/enc_beh.txt', root +'l001/encoding/s01/r03/enc_beh.txt', root +'l001/encoding/s01/r04/enc_beh.txt'],
                                  [root +'l001/encoding/s02/r01/enc_beh.txt', root +'l001/encoding/s02/r02/enc_beh.txt', root +'l001/encoding/s02/r03/enc_beh.txt', root +'l001/encoding/s02/r04/enc_beh.txt'],
                                  [root +'l001/encoding/s03/r01/enc_beh.txt', root +'l001/encoding/s03/r02/enc_beh.txt', root +'l001/encoding/s03/r03/enc_beh.txt', root +'l001/encoding/s03/r04/enc_beh.txt'],
                                  [root +'l001/encoding/s04/r01/enc_beh.txt', root +'l001/encoding/s04/r02/enc_beh.txt', root +'l001/encoding/s04/r03/enc_beh.txt', root +'l001/encoding/s04/r04/enc_beh.txt']]
            
            func_wmtask_sess = [[root +'l001/WMtask/s01/r01/nocfmri5_task_Ax.nii', root +'l001/WMtask/s01/r02/nocfmri5_task_Ax.nii', root +'l001/WMtask/s01/r03/nocfmri5_task_Ax.nii'],
                                [root +'l001/WMtask/s02/r01/nocfmri5_task_Ax.nii', root +'l001/WMtask/s02/r02/nocfmri5_task_Ax.nii', root +'l001/WMtask/s02/r03/nocfmri5_task_Ax.nii'],
                                [root +'l001/WMtask/s03/r01/nocfmri5_task_Ax.nii', root +'l001/WMtask/s03/r02/nocfmri5_task_Ax.nii', root +'l001/WMtask/s03/r03/nocfmri5_task_Ax.nii', root +'l001/WMtask/s03/r04/nocfmri5_task_Ax.nii'],
                                [root +'l001/WMtask/s04/r01/nocfmri5_task_Ax.nii', root +'l001/WMtask/s04/r02/nocfmri5_task_Ax.nii', root +'l001/WMtask/s04/r03/nocfmri5_task_Ax.nii', root +'l001/WMtask/s04/r04/nocfmri5_task_Ax.nii']]
            
            
            Beh_WM_files_sess=[[root +'l001/WMtask/s01/r01/wm_beh.txt', root +'l001/WMtask/s01/r02/wm_beh.txt', root +'l001/WMtask/s01/r03/wm_beh.txt'],
                                [root +'l001/WMtask/s02/r01/wm_beh.txt', root +'l001/WMtask/s02/r02/wm_beh.txt', root +'l001/WMtask/s02/r03/wm_beh.txt'],
                                [root +'l001/WMtask/s03/r01/wm_beh.txt', root +'l001/WMtask/s03/r02/wm_beh.txt', root +'l001/WMtask/s03/r03/wm_beh.txt', root +'l001/WMtask/s03/r04/wm_beh.txt'],
                                [root +'l001/WMtask/s04/r01/wm_beh.txt', root +'l001/WMtask/s04/r02/wm_beh.txt', root +'l001/WMtask/s04/r03/wm_beh.txt', root +'l001/WMtask/s04/r04/wm_beh.txt']]
            
            
            
            path_masks = root +  'temp_masks/l001/'
            
            #Chose the brain_region
            if brain_region=="visual":
                Maskrh = 'visual_fsign_rh.nii.gz'
                Masklh = 'visual_fsign_lh.nii.gz'
                writer_matrix=pd.ExcelWriter("/home/david/Desktop/KAROLINSKA/encoding_model/Matrix_encoding_model/l001_visual_bysess_matrix.xlsx")
                #df_name ='df_sessions_visual_l001.xlsx'
            
            elif brain_region=="ips":
                Maskrh = 'ips_str_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'ips_str_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                writer_matrix=pd.ExcelWriter("/home/david/Desktop/KAROLINSKA/encoding_model/Matrix_encoding_model/l001_ips_bysess_matrix.xlsx")
                #df_name ='df_sessions_ips_l001.xlsx'
                
        
        
        
        
        if Subject_analysis == "s001":
            
            func_encoding_sess = [[root +'s001/encoding/s01/r01/nocfmri3_Encoding_Ax.nii', root +'s001/encoding/s01/r02/nocfmri3_Encoding_Ax.nii'],
                                  [root +'s001/encoding/s02/r01/nocfmri3_Encoding_Ax.nii', root +'s001/encoding/s02/r02/nocfmri3_Encoding_Ax.nii'],
                                  [root +'s001/encoding/s03/r01/nocfmri3_Encoding_Ax.nii', root +'s001/encoding/s03/r02/nocfmri3_Encoding_Ax.nii', root +'s001/encoding/s03/r03/nocfmri3_Encoding_Ax.nii', root +'s001/encoding/s03/r04/nocfmri3_Encoding_Ax.nii'],
                                  [root +'s001/encoding/s04/r01/nocfmri3_Encoding_Ax.nii', root +'s001/encoding/s04/r02/nocfmri3_Encoding_Ax.nii'],
                                  [root +'s001/encoding/s05/r01/nocfmri3_Encoding_Ax.nii', root +'s001/encoding/s05/r02/nocfmri3_Encoding_Ax.nii']]
            
            
            Beh_enc_files_sess =[[root +'s001/encoding/s01/r01/enc_beh.txt', root +'s001/encoding/s01/r02/enc_beh.txt'],
                                  [root +'s001/encoding/s02/r01/enc_beh.txt', root +'s001/encoding/s02/r02/enc_beh.txt'],
                                  [root +'s001/encoding/s03/r01/enc_beh.txt', root +'s001/encoding/s03/r02/enc_beh.txt', root +'s001/encoding/s03/r03/enc_beh.txt', root +'s001/encoding/s03/r04/enc_beh.txt'],
                                  [root +'s001/encoding/s04/r01/enc_beh.txt', root +'s001/encoding/s04/r02/enc_beh.txt'],
                                  [root +'s001/encoding/s05/r01/enc_beh.txt', root +'s001/encoding/s05/r02/enc_beh.txt']]
            
            func_wmtask_sess = [[root +'s001/WMtask/s01/r01/nocfmri5_task_Ax.nii', root +'s001/WMtask/s01/r02/nocfmri5_task_Ax.nii'],
                                [root +'s001/WMtask/s02/r01/nocfmri5_task_Ax.nii'],
                                [root +'s001/WMtask/s03/r01/nocfmri5_task_Ax.nii', root +'s001/WMtask/s03/r02/nocfmri5_task_Ax.nii', root +'s001/WMtask/s03/r03/nocfmri5_task_Ax.nii', root +'s001/WMtask/s03/r04/nocfmri5_task_Ax.nii', root +'s001/WMtask/s03/r05/nocfmri5_task_Ax.nii'],
                                [root +'s001/WMtask/s04/r01/nocfmri5_task_Ax.nii', root +'s001/WMtask/s04/r02/nocfmri5_task_Ax.nii'],
                                [root +'s001/WMtask/s05/r01/nocfmri5_task_Ax.nii', root +'s001/WMtask/s05/r02/nocfmri5_task_Ax.nii']]
            
            
            Beh_WM_files_sess=[[root +'s001/WMtask/s01/r01/wm_beh.txt', root +'s001/WMtask/s01/r02/wm_beh.txt'],
                                [root +'s001/WMtask/s02/r01/wm_beh.txt'],
                                [root +'s001/WMtask/s03/r01/wm_beh.txt', root +'s001/WMtask/s03/r02/wm_beh.txt', root +'s001/WMtask/s03/r03/wm_beh.txt', root +'s001/WMtask/s03/r04/wm_beh.txt', root +'s001/WMtask/s03/r05/wm_beh.txt'],
                                [root +'s001/WMtask/s04/r01/wm_beh.txt', root +'s001/WMtask/s04/r02/wm_beh.txt'],
                                [root +'s001/WMtask/s05/r01/wm_beh.txt', root +'s001/WMtask/s05/r02/wm_beh.txt']]
            
            
            
            path_masks =  root +  'temp_masks/s001/'
            
            #Chose the brain_region        
            if brain_region=="visual":
                Maskrh = 'visual_fsign_rh.nii.gz'
                Masklh = 'visual_fsign_lh.nii.gz'
                writer_matrix=pd.ExcelWriter("/home/david/Desktop/KAROLINSKA/encoding_model/Matrix_encoding_model/s001_visual_bysess_matrix.xlsx")
                #df_name ='df_sessions_visual_s001.xlsx'
            
            elif brain_region=="ips":
                Maskrh = 'ips_str_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'ips_str_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                writer_matrix=pd.ExcelWriter("/home/david/Desktop/KAROLINSKA/encoding_model/Matrix_encoding_model/s001_ips_bysess_matrix.xlsx")
                #df_name ='df_sessions_ips_s001.xlsx'
            
            
            
        if Subject_analysis == "r001":
            
            func_encoding_sess = [[root +'r001/encoding/s05/r01/nocfmri3_Encoding_Ax.nii', root +'r001/encoding/s05/r02/nocfmri3_Encoding_Ax.nii', root +'r001/encoding/s05/r03/nocfmri3_Encoding_Ax.nii', root +'r001/encoding/s05/r04/nocfmri3_Encoding_Ax.nii'],
                                  [root +'r001/encoding/s06/r01/nocfmri3_Encoding_Ax.nii', root +'r001/encoding/s06/r02/nocfmri3_Encoding_Ax.nii', root +'r001/encoding/s06/r03/nocfmri3_Encoding_Ax.nii', root +'r001/encoding/s06/r04/nocfmri3_Encoding_Ax.nii'],
                                  [root +'r001/encoding/s07/r01/nocfmri3_Encoding_Ax.nii', root +'r001/encoding/s07/r02/nocfmri3_Encoding_Ax.nii', root +'r001/encoding/s07/r03/nocfmri3_Encoding_Ax.nii', root +'r001/encoding/s07/r04/nocfmri3_Encoding_Ax.nii']]
            
            
            Beh_enc_files_sess =[[root +'r001/encoding/s05/r01/enc_beh.txt', root +'r001/encoding/s05/r02/enc_beh.txt', root +'r001/encoding/s05/r03/enc_beh.txt', root +'r001/encoding/s05/r04/enc_beh.txt'],
                                  [root +'r001/encoding/s06/r01/enc_beh.txt', root +'r001/encoding/s06/r02/enc_beh.txt', root +'r001/encoding/s06/r03/enc_beh.txt', root +'r001/encoding/s06/r04/enc_beh.txt'],
                                  [root +'r001/encoding/s07/r01/enc_beh.txt', root +'r001/encoding/s07/r02/enc_beh.txt', root +'r001/encoding/s07/r03/enc_beh.txt', root +'r001/encoding/s07/r04/enc_beh.txt']]
            
            
            func_wmtask_sess = [[root +'r001/WMtask/s08/r01/nocfmri5_task_Ax.nii', root +'r001/WMtask/s08/r02/nocfmri5_task_Ax.nii', root +'r001/WMtask/s08/r03/nocfmri5_task_Ax.nii', root +'r001/WMtask/s08/r04/nocfmri5_task_Ax.nii', root +'r001/WMtask/s08/r05/nocfmri5_task_Ax.nii', root +'r001/WMtask/s08/r06/nocfmri5_task_Ax.nii'],
                                [root +'r001/WMtask/s09/r01/nocfmri5_task_Ax.nii', root +'r001/WMtask/s09/r02/nocfmri5_task_Ax.nii', root +'r001/WMtask/s09/r03/nocfmri5_task_Ax.nii', root +'r001/WMtask/s09/r04/nocfmri5_task_Ax.nii'],
                                [root +'r001/WMtask/s10/r01/nocfmri5_task_Ax.nii']]
            
            
            Beh_WM_files_sess=[[root +'r001/WMtask/s08/r01/wm_beh.txt', root +'r001/WMtask/s08/r02/wm_beh.txt', root +'r001/WMtask/s08/r03/wm_beh.txt', root +'r001/WMtask/s08/r04/wm_beh.txt', root +'r001/WMtask/s08/r05/wm_beh.txt', root +'r001/WMtask/s08/r06/wm_beh.txt'],
                                [root +'r001/WMtask/s09/r01/wm_beh.txt', root +'r001/WMtask/s09/r02/wm_beh.txt', root +'r001/WMtask/s09/r03/wm_beh.txt', root +'r001/WMtask/s09/r04/wm_beh.txt'],
                                [root +'r001/WMtask/s10/r01/wm_beh.txt']]
            
            
            
            path_masks = root +  'temp_masks/r001/'
            
            #Chose the brain_region        
            if brain_region=="visual":
                Maskrh = 'visual_fsign_rh.nii.gz'
                Masklh = 'visual_fsign_lh.nii.gz'
                writer_matrix=pd.ExcelWriter("/home/david/Desktop/KAROLINSKA/encoding_model/Matrix_encoding_model/r001_visual_bysess_matrix.xlsx")
                #df_name ='df_sessions_visual_r001.xlsx'
            
            elif brain_region=="ips":
                Maskrh = 'ips_str_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'ips_str_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                writer_matrix=pd.ExcelWriter("/home/david/Desktop/KAROLINSKA/encoding_model/Matrix_encoding_model/r001_ips_bysess_matrix.xlsx")
                #df_name ='df_sessions_ips_r001.xlsx'    
    
    
    
    return Method_analysis, distance_ch, Subject_analysis, brain_region, distance, func_encoding_sess, Beh_enc_files_sess, func_wmtask_sess, Beh_WM_files_sess, path_masks, Maskrh, Masklh, writer_matrix
                
        
    
####

def circ_dist(a1,a2):
    ## Returns the minimal distance in angles between to angles 
    op1=abs(a2-a1)
    angs=[a1,a2]
    op2=min(angs)+(360-max(angs))
    options=[op1,op2]
    return min(options)




def zscore(a):
    #Input list
    #Return list zscored    
    b=[(a[i] - mean(a))/std(a) for i in range(0, len(a))]
    return b



#Parameters of the model 
radius=16
sep_channels=10
size_chboard =2.7
deg_check_board= degrees(size_chboard/radius)

#size constant:  2.tcm --> 1.19v.a (calculate_visual_angle_from_stim_size.py) (d=130cm)
# 1.19 x 5 = 5.95v.a (round(1.19*5,2))
size_constant=5.95

#adjusted_size_contant (size_constant in degrees) --> 5.95v.a  -->  13.549cm (run calculate_stim_size_from_visual_angle.py) (d=130cm)
# 13.549cm/ 16 --> 48.519   convert to radians, and calculate degrees: degrees(13.549/16)   (round(degrees(13.549/16), 3)) 
adjusted_size_contant = 48.519



#Generate the positions of the channels (there will be 14)
pos_channels = arange(sep_channels/2,360,sep_channels)
pos_channels = [round(pos_channels[i],3) for i in range(0, len(pos_channels))]
#save('position_channels', pos_channels)

pos_channels2 = arange(0,360,0.5)
pos_channels2 = [round(pos_channels2[i],3) for i in range(0, len(pos_channels2))]
#save('position_channels2', pos_channels2)


def posch1_to_posch2(ch_1):
    return where(array(pos_channels2) == pos_channels[ch_1])[0][0]





def f(position_target):
    #I want to return a list of the activity of each channel in front of a stim at any location
    #
    #The f function imput is the distance from the position to the channel. That is why first we need to
    #get a distance from the locaion to each channel
    #
    #First i calculate the distance in degrees from the location to each channel
    #Once I have the distance value, I use the same formula as Sprague to extract a value of f for each channel.
    #
    
    #colculate the r : the circular distance between target position and each channel
    list_r=[]
    for channel in pos_channels:
        R = round(circ_dist(position_target, channel), 3)
        list_r.append(R)
    
    #print list_r
    
    #I need the adjusted because the r is not in visual angles, it is in degrees
    #I calculate the f for those inside the spread of the maximum, farther, it is 0
    f_list=[]
    for r in list_r:
        if r<adjusted_size_contant:
            #f = ( 0.5 + 0.5*cos(r*pi/adjusted_size_contant) )
            f = ( 0.5 + 0.5*cos(r*pi/adjusted_size_contant) )**7
            f=round(f,3)
            f_list.append(f)
        else:
            f = 0
            f_list.append(f)
    
    
    #Return the list   
    return f_list




def f2(position_target):
    list_r=[]
    for channel in pos_channels2:
        R = round(circ_dist(position_target, channel), 3)
        list_r.append(R)
    
    f_list=[]
    for r in list_r:
        if r<adjusted_size_contant:
            f = ( 0.5 + 0.5*cos(r*pi/adjusted_size_contant) )**7
            f=round(f,3)
            f_list.append(f)
        else:
            f = 0
            f_list.append(f)
    
    return f_list






def get_quadrant(angle): 
    #angle in degrees, return the quadrant which it belongs to
    if angle>=0 and angle<90:
        Q=1
    elif angle>=90 and angle<180:
        Q=2
    elif angle>=180 and angle<270:
        Q=3
    elif angle>=270 and angle<=360:
        Q=4
    
    return Q








def f_quadrant(position_target):
    #I want to return a list of the activity of each channel in front of a stim at any location
    #
    #The f function imput is the distance from the position to the channel. That is why first we need to
    #get a distance from the locaion to each channel
    #
    #
    # In this case we will create 4 lists and calculate the response in just one (append them in the right way)
    #
    #First i calculate the distance in degrees from the location to each channel
    #Once I have the distance value, I use the same formula as Sprague to extract a value of f for each channel.
    #
    #
    #Get the quadrant
    Q = get_quadrant(position_target)
    
    n_q= len(pos_channels)/4
    
    pos_channels_quad = [pos_channels[i:i+n_q] for i in range(0, len(pos_channels),n_q)]
    
    pos_channels_trial = pos_channels_quad[Q-1]
    #colculate the r : the circular distance between target position and each channel
    list_r=[]
    for channel in pos_channels_trial:
        R = round(circ_dist(position_target, channel), 3)
        list_r.append(R)
    
    #print list_r
    
    #I need the adjusted because the r is not in visual angles, it is in degrees
    #I calculate the f for those inside the spread of the maximum, farther, it is 0
    f_list=[]
    for r in list_r:
        if r<adjusted_size_contant:
            #f = ( 0.5 + 0.5*cos(r*pi/adjusted_size_contant) )
            f = ( 0.5 + 0.5*cos(r*pi/adjusted_size_contant) )**7
            f=round(f,3)
            f_list.append(f)
        else:
            f = 0
            f_list.append(f)
    
    
    #Return the list   
    if Q==1:
        mergedlist = f_list + [0]*(n_q*3)
    elif Q==2:
        mergedlist = [0]*n_q + f_list + [0]*(n_q*2)
    elif Q==3:
        mergedlist = [0]*(n_q*2) + f_list + [0]*n_q
    else:
        mergedlist = f_list + [0]*(n_q*3)
    
        
    return mergedlist









def ch2vrep(channel):
    #Input the channel activity
    #Return the visual respresentation of this channel activity
    ###
    #It multiplies each channel by its corresponding f function --> 36 values
    #It sums all the 36 values of the 36 channels  --> 36 values (a way to smooth)
    #Equivalent to the population vector
    all_basis_functions=[]
    for pos, ch_value in enumerate(pos_channels):
        all_basis_functions.append(channel[pos]*array( f(ch_value)  ))
    
    
    vrep=sum(all_basis_functions)
    return vrep






def ch2vrep3(channel):
    #Input the channel activity
    #Return the visual respresentation of this channel activity
    ###
    #It multiplies each channel by its corresponding f function --> 36 values
    #It sums all the 36 values of the 36 channels  --> 36 values (a way to smooth)
    #Equivalent to the population vector
    all_basis_functions=[]
    for pos, ch_value in enumerate(pos_channels):
        a = channel[pos]*array( f2(ch_value) )
        #a= sum(a)
        all_basis_functions.append(a)
        #all_basis_functions.append(channel[pos]*array( f2(ch_value)  ))
    
    
    vrep=sum(all_basis_functions)
    return vrep




def angle2chmax(angle):
    n=f(angle)
    ch_max = where(array(n)==max(n))[0][0]
    return ch_max






def ub_wind_path(PATH, system):
    if system=='wind':
        A = PATH                                                                                                    
        B = A.replace('/', os.path.sep)                                                                                               
        C= B.replace('\\mnt\\c\\', 'C:\\') 
    
    if system=='unix':
        C=PATH
    
    ###
    return C


