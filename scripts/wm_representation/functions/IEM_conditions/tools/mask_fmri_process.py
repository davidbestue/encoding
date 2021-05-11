
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:05:33 2019

@author: David
"""

def mask_fmri_process(fmri_path, masks, sys_use='unix'):
    ### 1. Load and mask the data
    fmri_path = ub_wind_path(fmri_path, system=sys_use) #change the path format wind-unix
    
    mask_img_rh= masks[0] #right hemisphere mask
    mask_img_rh = ub_wind_path(mask_img_rh, system=sys_use)
    mask_img_lh= masks[1] #left hemisphere mask
    mask_img_lh = ub_wind_path(mask_img_lh, system=sys_use)
    
    #Apply the masks and concatenate   
    masked_data_rh = apply_mask(fmri_path, mask_img_rh)
    masked_data_lh = apply_mask(fmri_path, mask_img_lh)    
    masked_data=np.hstack([masked_data_rh, masked_data_lh])
    
    ### 2. Filter ####and zscore
    n_voxels = np.shape(masked_data)[1]
    for voxel in range(0, n_voxels):
        data_to_filter = masked_data[:,voxel]                        
        #apply the filter 
        data_to_filter = TimeSeries(data_to_filter, sampling_interval=2.335)
        F = FilterAnalyzer(data_to_filter, ub=0.15, lb=0.02)
        data_filtered=F.filtered_boxcar.data
        masked_data[:,voxel] = data_filtered                        
        #Z score
        masked_data[:,voxel] = np.array( stats.zscore( masked_data[:,voxel]  ) ) ; ## zscore + 5 just to get + values
    
    #append it and save the data
    return masked_data    
