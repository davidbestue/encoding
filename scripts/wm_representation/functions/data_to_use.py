# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:01:27 2019

@author: David Bestue
"""


def data_to_use( Subject_analysis, Method_analysis, brain_region):
    root='/home/david/Desktop/IEM_data/'    
    if Method_analysis == "together": ##
        
        if Subject_analysis == "d001":
            
            func_encoding_sess = [root +'d001/encoding/s08/r01/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s08/r02/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s08/r03/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s08/r04/nocfmri3_Encoding_Ax.nii',
                             root +'d001/encoding/s09/r01/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s09/r02/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s09/r03/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s09/r04/nocfmri3_Encoding_Ax.nii',
                             root +'d001/encoding/s10/r01/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s10/r02/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s10/r03/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s10/r04/nocfmri3_Encoding_Ax.nii',
                             root +'d001/encoding/s11/r01/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s11/r02/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s11/r03/nocfmri3_Encoding_Ax.nii', root +'d001/encoding/s11/r04/nocfmri3_Encoding_Ax.nii']
            
            
            Beh_enc_files_sess = [root +'d001/encoding/s08/r01/enc_beh.txt', root +'d001/encoding/s08/r02/enc_beh.txt', root +'d001/encoding/s08/r03/enc_beh.txt', root +'d001/encoding/s08/r04/enc_beh.txt',
                             root +'d001/encoding/s09/r01/enc_beh.txt', root +'d001/encoding/s09/r02/enc_beh.txt', root +'d001/encoding/s09/r03/enc_beh.txt', root +'d001/encoding/s09/r04/enc_beh.txt',
                             root +'d001/encoding/s10/r01/enc_beh.txt', root +'d001/encoding/s10/r02/enc_beh.txt', root +'d001/encoding/s10/r03/enc_beh.txt', root +'d001/encoding/s10/r04/enc_beh.txt',
                             root +'d001/encoding/s11/r01/enc_beh.txt', root +'d001/encoding/s11/r02/enc_beh.txt', root +'d001/encoding/s11/r03/enc_beh.txt', root +'d001/encoding/s11/r04/enc_beh.txt']
            
            
            
            func_wmtask_sess =  [root +'d001/WMtask/s10/r01/nocfmri5_task_Ax.nii', root +'d001/WMtask/s10/r02/nocfmri5_task_Ax.nii', root +'d001/WMtask/s10/r03/nocfmri5_task_Ax.nii', root +'d001/WMtask/s10/r04/nocfmri5_task_Ax.nii', root +'d001/WMtask/s10/r05/nocfmri5_task_Ax.nii',
                             root +'d001/WMtask/s11/r01/nocfmri5_task_Ax.nii', root +'d001/WMtask/s11/r02/nocfmri5_task_Ax.nii', root +'d001/WMtask/s11/r03/nocfmri5_task_Ax.nii',
                             root +'d001/WMtask/s12/r01/nocfmri5_task_Ax.nii', root +'d001/WMtask/s12/r02/nocfmri5_task_Ax.nii',
                             root +'d001/WMtask/s13/r01/nocfmri5_task_Ax.nii', root +'d001/WMtask/s13/r02/nocfmri5_task_Ax.nii', root +'d001/WMtask/s13/r03/nocfmri5_task_Ax.nii']
            
            
            Beh_WM_files_sess= [root +'d001/WMtask/s10/r01/wm_beh.txt', root +'d001/WMtask/s10/r02/wm_beh.txt', root +'d001/WMtask/s10/r03/wm_beh.txt', root +'d001/WMtask/s10/r04/wm_beh.txt', root +'d001/WMtask/s10/r05/wm_beh.txt',
                             root +'d001/WMtask/s11/r01/wm_beh.txt', root +'d001/WMtask/s11/r02/wm_beh.txt', root +'d001/WMtask/s11/r03/wm_beh.txt',
                             root +'d001/WMtask/s12/r01/wm_beh.txt', root +'d001/WMtask/s12/r02/wm_beh.txt',
                             root +'d001/WMtask/s13/r01/wm_beh.txt', root +'d001/WMtask/s13/r02/wm_beh.txt', root +'d001/WMtask/s13/r03/wm_beh.txt']
            
            
            
            ##Mask
            path_masks = root +  'temp_masks/d001/' 
            #Chose the brain_region        
            if brain_region=="visual":
                Maskrh = 'visual_fsign_rh.nii.gz'
                Masklh = 'visual_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]
            
            elif brain_region=="ips":
                Maskrh = 'd001_loc_parietal_rh.nii.gz' #'ips_str_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'd001_loc_parietal_lh.nii.gz' #'ips_str_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]
            
            elif brain_region=="pfc":
                Maskrh = 'd001_loc_frontal_rh.nii.gz' #'d001_frontal_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'd001_loc_frontal_lh.nii.gz' #'d001_frontal_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]   
            
            elif brain_region=="frontsup":
                Maskrh = 'front_sup_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'front_sup_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]   
            
            elif brain_region=="frontmid":
                Maskrh = 'front_middle_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'front_middle_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]
            
            elif brain_region=="frontinf":
                Maskrh = 'front_inf_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'front_inf_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]      
        
        
        
        if Subject_analysis == "n001":
            
            func_encoding_sess = [root +'n001/encoding/s01/r01/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s01/r02/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s01/r03/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s01/r04/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s01/r05/nocfmri3_Encoding_Ax.nii',
                                  root +'n001/encoding/s02/r01/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s02/r02/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s02/r03/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s02/r04/nocfmri3_Encoding_Ax.nii',
                                  root +'n001/encoding/s03/r01/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s03/r02/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s03/r03/nocfmri3_Encoding_Ax.nii', root +'n001/encoding/s03/r04/nocfmri3_Encoding_Ax.nii']
            
            Beh_enc_files_sess =[root +'n001/encoding/s01/r01/enc_beh.txt', root +'n001/encoding/s01/r02/enc_beh.txt', root +'n001/encoding/s01/r03/enc_beh.txt', root +'n001/encoding/s01/r04/enc_beh.txt', root +'n001/encoding/s01/r05/enc_beh.txt',
                                 root +'n001/encoding/s02/r01/enc_beh.txt', root +'n001/encoding/s02/r02/enc_beh.txt', root +'n001/encoding/s02/r03/enc_beh.txt', root +'n001/encoding/s02/r04/enc_beh.txt',
                                 root +'n001/encoding/s03/r01/enc_beh.txt', root +'n001/encoding/s03/r02/enc_beh.txt', root +'n001/encoding/s03/r03/enc_beh.txt', root +'n001/encoding/s03/r04/enc_beh.txt']
            
            
            func_wmtask_sess = [root +'n001/WMtask/s01/r01/nocfmri5_task_Ax.nii', root +'n001/WMtask/s01/r02/nocfmri5_task_Ax.nii', root +'n001/WMtask/s01/r03/nocfmri5_task_Ax.nii', root +'n001/WMtask/s01/r04/nocfmri5_task_Ax.nii', root +'n001/WMtask/s01/r05/nocfmri5_task_Ax.nii',
                                root +'n001/WMtask/s02/r01/nocfmri5_task_Ax.nii', root +'n001/WMtask/s02/r02/nocfmri5_task_Ax.nii', root +'n001/WMtask/s02/r03/nocfmri5_task_Ax.nii', root +'n001/WMtask/s02/r04/nocfmri5_task_Ax.nii',
                                root +'n001/WMtask/s03/r01/nocfmri5_task_Ax.nii', root +'n001/WMtask/s03/r02/nocfmri5_task_Ax.nii', root +'n001/WMtask/s03/r03/nocfmri5_task_Ax.nii', root +'n001/WMtask/s03/r04/nocfmri5_task_Ax.nii', root +'n001/WMtask/s03/r05/nocfmri5_task_Ax.nii']
            
            
            Beh_WM_files_sess=[root +'n001/WMtask/s01/r01/wm_beh.txt', root +'n001/WMtask/s01/r02/wm_beh.txt', root +'n001/WMtask/s01/r03/wm_beh.txt', root +'n001/WMtask/s01/r04/wm_beh.txt', root +'n001/WMtask/s01/r05/wm_beh.txt',
                                root +'n001/WMtask/s02/r01/wm_beh.txt', root +'n001/WMtask/s02/r02/wm_beh.txt', root +'n001/WMtask/s02/r03/wm_beh.txt', root +'n001/WMtask/s02/r04/wm_beh.txt',
                                root +'n001/WMtask/s03/r01/wm_beh.txt', root +'n001/WMtask/s03/r02/wm_beh.txt', root +'n001/WMtask/s03/r03/wm_beh.txt', root +'n001/WMtask/s03/r04/wm_beh.txt', root +'n001/WMtask/s03/r05/wm_beh.txt']
            
            
            
            path_masks = root +  'temp_masks/n001/'
            
            #Chose the brain_region        
            if brain_region=="visual":
                Maskrh = 'visual_fsign_rh.nii.gz'
                Masklh = 'visual_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]
            
            elif brain_region=="ips":
                Maskrh = 'n001_loc_parietal_rh.nii.gz' #'ips_str_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'n001_loc_parietal_lh.nii.gz' #'ips_str_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]
            
            elif brain_region=="pfc":
                Maskrh = 'n001_loc_frontal_rh.nii.gz' #'n001_frontal_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'n001_loc_frontal_lh.nii.gz' #'n001_frontal_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]   
            
            elif brain_region=="frontsup":
                Maskrh = 'front_sup_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'front_sup_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]   
            
            elif brain_region=="frontmid":
                Maskrh = 'front_middle_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'front_middle_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]
            
            elif brain_region=="frontinf":
                Maskrh = 'front_inf_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'front_inf_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]               
            
        
        
        if Subject_analysis == "b001":
            
            func_encoding_sess = [root +'b001/encoding/s01/r01/nocfmri3_Encoding_Ax.nii', root +'b001/encoding/s01/r02/nocfmri3_Encoding_Ax.nii', root +'b001/encoding/s01/r03/nocfmri3_Encoding_Ax.nii', root +'b001/encoding/s01/r04/nocfmri3_Encoding_Ax.nii',
                                  root +'b001/encoding/s02/r01/nocfmri3_Encoding_Ax.nii', root +'b001/encoding/s02/r02/nocfmri3_Encoding_Ax.nii', root +'b001/encoding/s02/r03/nocfmri3_Encoding_Ax.nii', root +'b001/encoding/s02/r04/nocfmri3_Encoding_Ax.nii',
                                  root +'b001/encoding/s03/r01/nocfmri3_Encoding_Ax.nii', root +'b001/encoding/s03/r02/nocfmri3_Encoding_Ax.nii', root +'b001/encoding/s03/r03/nocfmri3_Encoding_Ax.nii', root +'b001/encoding/s03/r04/nocfmri3_Encoding_Ax.nii']
            
            Beh_enc_files_sess =[root +'b001/encoding/s01/r01/enc_beh.txt', root +'b001/encoding/s01/r02/enc_beh.txt', root +'b001/encoding/s01/r03/enc_beh.txt', root +'b001/encoding/s01/r04/enc_beh.txt',
                                  root +'b001/encoding/s02/r01/enc_beh.txt', root +'b001/encoding/s02/r02/enc_beh.txt', root +'b001/encoding/s02/r03/enc_beh.txt', root +'b001/encoding/s02/r04/enc_beh.txt',
                                  root +'b001/encoding/s03/r01/enc_beh.txt', root +'b001/encoding/s03/r02/enc_beh.txt', root +'b001/encoding/s03/r03/enc_beh.txt', root +'b001/encoding/s03/r04/enc_beh.txt']
            
            
            func_wmtask_sess = [root +'b001/WMtask/s01/r01/nocfmri5_task_Ax.nii', root +'b001/WMtask/s01/r02/nocfmri5_task_Ax.nii', root +'b001/WMtask/s01/r03/nocfmri5_task_Ax.nii', root +'b001/WMtask/s01/r04/nocfmri5_task_Ax.nii', root +'b001/WMtask/s01/r05/nocfmri5_task_Ax.nii',
                                root +'b001/WMtask/s02/r01/nocfmri5_task_Ax.nii', root +'b001/WMtask/s02/r02/nocfmri5_task_Ax.nii', root +'b001/WMtask/s02/r03/nocfmri5_task_Ax.nii', root +'b001/WMtask/s02/r04/nocfmri5_task_Ax.nii', root +'b001/WMtask/s02/r05/nocfmri5_task_Ax.nii',
                                root +'b001/WMtask/s03/r01/nocfmri5_task_Ax.nii', root +'b001/WMtask/s03/r02/nocfmri5_task_Ax.nii', root +'b001/WMtask/s03/r03/nocfmri5_task_Ax.nii', root +'b001/WMtask/s03/r04/nocfmri5_task_Ax.nii']
            
            
            Beh_WM_files_sess=[root +'b001/WMtask/s01/r01/wm_beh.txt', root +'b001/WMtask/s01/r02/wm_beh.txt', root +'b001/WMtask/s01/r03/wm_beh.txt', root +'b001/WMtask/s01/r04/wm_beh.txt', root +'b001/WMtask/s01/r05/wm_beh.txt',
                                root +'b001/WMtask/s02/r01/wm_beh.txt', root +'b001/WMtask/s02/r02/wm_beh.txt', root +'b001/WMtask/s02/r03/wm_beh.txt', root +'b001/WMtask/s02/r04/wm_beh.txt', root +'b001/WMtask/s02/r05/wm_beh.txt',
                                root +'b001/WMtask/s03/r01/wm_beh.txt', root +'b001/WMtask/s03/r02/wm_beh.txt', root +'b001/WMtask/s03/r03/wm_beh.txt', root +'b001/WMtask/s03/r04/wm_beh.txt']
            
            
            path_masks =  root +  'temp_masks/b001/'
            
            #Chose the brain_region        
            if brain_region=="visual":
                Maskrh = 'visual_fsign_rh.nii.gz'
                Masklh = 'visual_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh ]
            
            elif brain_region=="ips":
                Maskrh = 'ips_str_rh.nii.gz' #'ips_fsign_rh.nii.gz'
                Masklh = 'ips_str_lh.nii.gz' #'ips_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]
            
            elif brain_region=="pfc":
                Maskrh = 'b001_frontal_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'b001_frontal_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]  
            
            elif brain_region=="frontsup":
                Maskrh = 'front_sup_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'front_sup_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]   
            
            elif brain_region=="frontmid":
                Maskrh = 'front_middle_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'front_middle_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]
            
            elif brain_region=="frontinf":
                Maskrh = 'front_inf_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'front_inf_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]                   
        
        
        if Subject_analysis == "l001":
            
            func_encoding_sess = [root +'l001/encoding/s01/r01/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s01/r02/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s01/r03/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s01/r04/nocfmri3_Encoding_Ax.nii',
                                  root +'l001/encoding/s02/r01/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s02/r02/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s02/r03/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s02/r04/nocfmri3_Encoding_Ax.nii',
                                  root +'l001/encoding/s03/r01/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s03/r02/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s03/r03/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s03/r04/nocfmri3_Encoding_Ax.nii',
                                  root +'l001/encoding/s04/r01/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s04/r02/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s04/r03/nocfmri3_Encoding_Ax.nii', root +'l001/encoding/s04/r04/nocfmri3_Encoding_Ax.nii']
            
            
            Beh_enc_files_sess =[root +'l001/encoding/s01/r01/enc_beh.txt', root +'l001/encoding/s01/r02/enc_beh.txt', root +'l001/encoding/s01/r03/enc_beh.txt', root +'l001/encoding/s01/r04/enc_beh.txt',
                                  root +'l001/encoding/s02/r01/enc_beh.txt', root +'l001/encoding/s02/r02/enc_beh.txt', root +'l001/encoding/s02/r03/enc_beh.txt', root +'l001/encoding/s02/r04/enc_beh.txt',
                                  root +'l001/encoding/s03/r01/enc_beh.txt', root +'l001/encoding/s03/r02/enc_beh.txt', root +'l001/encoding/s03/r03/enc_beh.txt', root +'l001/encoding/s03/r04/enc_beh.txt',
                                  root +'l001/encoding/s04/r01/enc_beh.txt', root +'l001/encoding/s04/r02/enc_beh.txt', root +'l001/encoding/s04/r03/enc_beh.txt', root +'l001/encoding/s04/r04/enc_beh.txt']
            
            func_wmtask_sess = [root +'l001/WMtask/s01/r01/nocfmri5_task_Ax.nii', root +'l001/WMtask/s01/r02/nocfmri5_task_Ax.nii', root +'l001/WMtask/s01/r03/nocfmri5_task_Ax.nii',
                                root +'l001/WMtask/s02/r01/nocfmri5_task_Ax.nii', root +'l001/WMtask/s02/r02/nocfmri5_task_Ax.nii', root +'l001/WMtask/s02/r03/nocfmri5_task_Ax.nii',
                                root +'l001/WMtask/s03/r01/nocfmri5_task_Ax.nii', root +'l001/WMtask/s03/r02/nocfmri5_task_Ax.nii', root +'l001/WMtask/s03/r03/nocfmri5_task_Ax.nii', root +'l001/WMtask/s03/r04/nocfmri5_task_Ax.nii',
                                root +'l001/WMtask/s04/r01/nocfmri5_task_Ax.nii', root +'l001/WMtask/s04/r02/nocfmri5_task_Ax.nii', root +'l001/WMtask/s04/r03/nocfmri5_task_Ax.nii', root +'l001/WMtask/s04/r04/nocfmri5_task_Ax.nii']
            
            
            Beh_WM_files_sess=[root +'l001/WMtask/s01/r01/wm_beh.txt', root +'l001/WMtask/s01/r02/wm_beh.txt', root +'l001/WMtask/s01/r03/wm_beh.txt',
                                root +'l001/WMtask/s02/r01/wm_beh.txt', root +'l001/WMtask/s02/r02/wm_beh.txt', root +'l001/WMtask/s02/r03/wm_beh.txt',
                                root +'l001/WMtask/s03/r01/wm_beh.txt', root +'l001/WMtask/s03/r02/wm_beh.txt', root +'l001/WMtask/s03/r03/wm_beh.txt', root +'l001/WMtask/s03/r04/wm_beh.txt',
                                root +'l001/WMtask/s04/r01/wm_beh.txt', root +'l001/WMtask/s04/r02/wm_beh.txt', root +'l001/WMtask/s04/r03/wm_beh.txt', root +'l001/WMtask/s04/r04/wm_beh.txt']
            
            
            
            path_masks = root +  'temp_masks/l001/'
            
            #Chose the brain_region
            if brain_region=="visual":
                Maskrh = 'visual_fsign_rh.nii.gz'
                Masklh = 'visual_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh ]
            
            elif brain_region=="ips":
                Maskrh = 'ips_str_rh.nii.gz' #'ips_fsign_rh.nii.gz'
                Masklh = 'ips_str_lh.nii.gz' #'ips_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh ]
            
            elif brain_region=="pfc":
                Maskrh = 'l001_frontal_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'l001_frontal_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ] 
            
            elif brain_region=="frontsup":
                Maskrh = 'front_sup_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'front_sup_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]   
            
            elif brain_region=="frontmid":
                Maskrh = 'front_middle_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'front_middle_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]
            
            elif brain_region=="frontinf":
                Maskrh = 'front_inf_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'front_inf_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]           
        
        
        
        if Subject_analysis == "s001":
            
            func_encoding_sess = [root +'s001/encoding/s01/r01/nocfmri3_Encoding_Ax.nii', root +'s001/encoding/s01/r02/nocfmri3_Encoding_Ax.nii',
                                  root +'s001/encoding/s02/r01/nocfmri3_Encoding_Ax.nii', root +'s001/encoding/s02/r02/nocfmri3_Encoding_Ax.nii',
                                  root +'s001/encoding/s03/r01/nocfmri3_Encoding_Ax.nii', root +'s001/encoding/s03/r02/nocfmri3_Encoding_Ax.nii', root +'s001/encoding/s03/r03/nocfmri3_Encoding_Ax.nii', root +'s001/encoding/s03/r04/nocfmri3_Encoding_Ax.nii',
                                  root +'s001/encoding/s04/r01/nocfmri3_Encoding_Ax.nii', root +'s001/encoding/s04/r02/nocfmri3_Encoding_Ax.nii',
                                  root +'s001/encoding/s05/r01/nocfmri3_Encoding_Ax.nii', root +'s001/encoding/s05/r02/nocfmri3_Encoding_Ax.nii']
            
            
            Beh_enc_files_sess =[root +'s001/encoding/s01/r01/enc_beh.txt', root +'s001/encoding/s01/r02/enc_beh.txt',
                                  root +'s001/encoding/s02/r01/enc_beh.txt', root +'s001/encoding/s02/r02/enc_beh.txt',
                                  root +'s001/encoding/s03/r01/enc_beh.txt', root +'s001/encoding/s03/r02/enc_beh.txt', root +'s001/encoding/s03/r03/enc_beh.txt', root +'s001/encoding/s03/r04/enc_beh.txt',
                                  root +'s001/encoding/s04/r01/enc_beh.txt', root +'s001/encoding/s04/r02/enc_beh.txt',
                                  root +'s001/encoding/s05/r01/enc_beh.txt', root +'s001/encoding/s05/r02/enc_beh.txt']
            
            func_wmtask_sess = [root +'s001/WMtask/s01/r01/nocfmri5_task_Ax.nii', root +'s001/WMtask/s01/r02/nocfmri5_task_Ax.nii',
                                root +'s001/WMtask/s02/r01/nocfmri5_task_Ax.nii',
                                root +'s001/WMtask/s03/r01/nocfmri5_task_Ax.nii', root +'s001/WMtask/s03/r02/nocfmri5_task_Ax.nii', root +'s001/WMtask/s03/r03/nocfmri5_task_Ax.nii', root +'s001/WMtask/s03/r04/nocfmri5_task_Ax.nii', root +'s001/WMtask/s03/r05/nocfmri5_task_Ax.nii',
                                root +'s001/WMtask/s04/r01/nocfmri5_task_Ax.nii', root +'s001/WMtask/s04/r02/nocfmri5_task_Ax.nii',
                                root +'s001/WMtask/s05/r01/nocfmri5_task_Ax.nii', root +'s001/WMtask/s05/r02/nocfmri5_task_Ax.nii']
            
            
            Beh_WM_files_sess=[root +'s001/WMtask/s01/r01/wm_beh.txt', root +'s001/WMtask/s01/r02/wm_beh.txt',
                                root +'s001/WMtask/s02/r01/wm_beh.txt',
                                root +'s001/WMtask/s03/r01/wm_beh.txt', root +'s001/WMtask/s03/r02/wm_beh.txt', root +'s001/WMtask/s03/r03/wm_beh.txt', root +'s001/WMtask/s03/r04/wm_beh.txt', root +'s001/WMtask/s03/r05/wm_beh.txt',
                                root +'s001/WMtask/s04/r01/wm_beh.txt', root +'s001/WMtask/s04/r02/wm_beh.txt',
                                root +'s001/WMtask/s05/r01/wm_beh.txt', root +'s001/WMtask/s05/r02/wm_beh.txt']
            
            
            
            path_masks =  root +  'temp_masks/s001/'
            
            #Chose the brain_region        
            if brain_region=="visual":
                Maskrh = 'visual_fsign_rh.nii.gz'
                Masklh = 'visual_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]
            
            elif brain_region=="ips":
                Maskrh = 'ips_str_rh.nii.gz' #'ips_fsign_rh.nii.gz'
                Masklh = 'ips_str_lh.nii.gz' #'ips_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]
            
            elif brain_region=="pfc":
                Maskrh = 's001_frontal_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 's001_frontal_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ] 
            
            elif brain_region=="frontsup":
                Maskrh = 'front_sup_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'front_sup_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]   
            
            elif brain_region=="frontmid":
                Maskrh = 'front_middle_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'front_middle_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]
            
            elif brain_region=="frontinf":
                Maskrh = 'front_inf_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'front_inf_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]               
            
            
        if Subject_analysis == "r001":
            
            func_encoding_sess = [root +'r001/encoding/s05/r01/nocfmri3_Encoding_Ax.nii', root +'r001/encoding/s05/r02/nocfmri3_Encoding_Ax.nii', root +'r001/encoding/s05/r03/nocfmri3_Encoding_Ax.nii', root +'r001/encoding/s05/r04/nocfmri3_Encoding_Ax.nii',
                                  root +'r001/encoding/s06/r01/nocfmri3_Encoding_Ax.nii', root +'r001/encoding/s06/r02/nocfmri3_Encoding_Ax.nii', root +'r001/encoding/s06/r03/nocfmri3_Encoding_Ax.nii', root +'r001/encoding/s06/r04/nocfmri3_Encoding_Ax.nii',
                                  root +'r001/encoding/s07/r01/nocfmri3_Encoding_Ax.nii', root +'r001/encoding/s07/r02/nocfmri3_Encoding_Ax.nii', root +'r001/encoding/s07/r03/nocfmri3_Encoding_Ax.nii', root +'r001/encoding/s07/r04/nocfmri3_Encoding_Ax.nii']
            
            
            Beh_enc_files_sess =[root +'r001/encoding/s05/r01/enc_beh.txt', root +'r001/encoding/s05/r02/enc_beh.txt', root +'r001/encoding/s05/r03/enc_beh.txt', root +'r001/encoding/s05/r04/enc_beh.txt',
                                  root +'r001/encoding/s06/r01/enc_beh.txt', root +'r001/encoding/s06/r02/enc_beh.txt', root +'r001/encoding/s06/r03/enc_beh.txt', root +'r001/encoding/s06/r04/enc_beh.txt',
                                  root +'r001/encoding/s07/r01/enc_beh.txt', root +'r001/encoding/s07/r02/enc_beh.txt', root +'r001/encoding/s07/r03/enc_beh.txt', root +'r001/encoding/s07/r04/enc_beh.txt']
            
                        
            func_wmtask_sess = [root +'r001/WMtask/s08/r01/nocfmri5_task_Ax.nii', root +'r001/WMtask/s08/r02/nocfmri5_task_Ax.nii', root +'r001/WMtask/s08/r03/nocfmri5_task_Ax.nii', root +'r001/WMtask/s08/r04/nocfmri5_task_Ax.nii', root +'r001/WMtask/s08/r05/nocfmri5_task_Ax.nii', root +'r001/WMtask/s08/r06/nocfmri5_task_Ax.nii',
                                root +'r001/WMtask/s09/r01/nocfmri5_task_Ax.nii', root +'r001/WMtask/s09/r02/nocfmri5_task_Ax.nii', root +'r001/WMtask/s09/r03/nocfmri5_task_Ax.nii', root +'r001/WMtask/s09/r04/nocfmri5_task_Ax.nii',
                                root +'r001/WMtask/s10/r01/nocfmri5_task_Ax.nii']
            
            
            Beh_WM_files_sess=[root +'r001/WMtask/s08/r01/wm_beh.txt', root +'r001/WMtask/s08/r02/wm_beh.txt', root +'r001/WMtask/s08/r03/wm_beh.txt', root +'r001/WMtask/s08/r04/wm_beh.txt', root +'r001/WMtask/s08/r05/wm_beh.txt', root +'r001/WMtask/s08/r06/wm_beh.txt',
                                root +'r001/WMtask/s09/r01/wm_beh.txt', root +'r001/WMtask/s09/r02/wm_beh.txt', root +'r001/WMtask/s09/r03/wm_beh.txt', root +'r001/WMtask/s09/r04/wm_beh.txt',
                                root +'r001/WMtask/s10/r01/wm_beh.txt']
            
            
            
            path_masks = root +  'temp_masks/r001/'
            
            #Chose the brain_region        
            if brain_region=="visual":
                Maskrh = 'visual_fsign_rh.nii.gz'
                Masklh = 'visual_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh ]
            
            elif brain_region=="ips":
                Maskrh = 'ips_str_rh.nii.gz' #'ips_fsign_rh.nii.gz'
                Masklh = 'ips_str_lh.nii.gz' #'ips_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]   
            
            elif brain_region=="pfc":
                Maskrh = 'r001_frontal_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'r001_frontal_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]  
            
            elif brain_region=="frontsup":
                Maskrh = 'front_sup_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'front_sup_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]   
            
            elif brain_region=="frontmid":
                Maskrh = 'front_middle_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'front_middle_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]
            
            elif brain_region=="frontinf":
                Maskrh = 'front_inf_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'front_inf_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]       
    
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
            if brain_region=="visual":
                Maskrh = 'visual_fsign_rh.nii.gz'
                Masklh = 'visual_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]
            
            elif brain_region=="ips":
                Maskrh = 'ips_str_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'ips_str_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]
        
        
        
        
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
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]
            
            elif brain_region=="ips":
                Maskrh = 'ips_str_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'ips_str_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]
            
            
            
        
        
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
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]
            
            elif brain_region=="ips":
                Maskrh = 'ips_str_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'ips_str_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]
                
        
        
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
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]
            
            elif brain_region=="ips":
                Maskrh = 'ips_str_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'ips_str_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]
                
        
        
        
        
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
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]
            
            elif brain_region=="ips":
                Maskrh = 'ips_str_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'ips_str_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]
            
            
            
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
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]
            
            elif brain_region=="ips":
                Maskrh = 'ips_str_rh.nii.gz' #'ips_ext_fsign_rh.nii.gz'
                Masklh = 'ips_str_lh.nii.gz' #'ips_ext_fsign_lh.nii.gz'
                masks = [ path_masks + Maskrh, path_masks + Masklh  ]
            
        
        
    
    return func_encoding_sess, Beh_enc_files_sess, func_wmtask_sess, Beh_WM_files_sess, masks

    


#func_encoding_sess, Beh_enc_files_sess, func_wmtask_sess, Beh_WM_files_sess, masks = data_to_use( 'n001', 'together', 'ips')






  