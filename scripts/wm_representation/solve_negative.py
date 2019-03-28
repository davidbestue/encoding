# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 12:12:03 2018

@author: David Bestue
"""
###Decide brain region


import easygui
import os
import numpy as np

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
for SUBJECT_USE_ANALYSIS in ['n001']: #'d001', 'n001', 'r001', 'b001', 'l001', 's001'
    print(SUBJECT_USE_ANALYSIS)
    for brain_region in ["visual"]:  #"ips"
        for CONDITION in ['1_0.2']: 
            Method_analysis = 'together'
            #CONDITION = '1_0.2'
            #brain_region = "visual"
            distance_ch='mix'
            #distance='mix'
            Subject_analysis=SUBJECT_USE_ANALYSIS
            os.chdir(encoding_path)
            ############################################       
            from functions_encoding_loop import *
            Method_analysis, distance_ch, Subject_analysis, brain_region, distance, func_encoding_sess, Beh_enc_files_sess, func_wmtask_sess, Beh_WM_files_sess, path_masks, Maskrh, Masklh, writer_matrix = variables_encoding(Method_analysis, distance_ch, Subject_analysis, brain_region, root_use ) 
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
                
                
                
                
                
                
                ###### In each session I will:
                    ####   1. Apply a filter for each voxel
                    ####   2. Select the times corresponding to the delay (2TR)
                    ####   3. Subset of data corresponding to the times (all voxels)
                    ####   4. zscore + 10 in each voxel in the temporal dimension (with the other 2TR of the same session)
                    ####   5. Append both the activity and the target of each 2TR
                ####
                ###### Concatenate all the sessions (targets and activity) to create the training dataset
                
                
                Pos_targets=[]
                lens_enc_del=[]
                Enc_delay=[]
                n_voxels = shape(encoding_datasets[0])[1]
                
                
                ###High-pass filter and z-score per voxel
                
                
                for session_enc_sess in range(0, len(enc_lens_datas)):                  
                    
                    Pos_targets=[] ##  Get the targets for each trial of the session
                    Enc_delay=[] ## Get the scans to take from the data (beggining of the delay)
                    
                    ## load the file
                    Beh_enc_files_path = Beh_enc_files[session_enc_sess] ## name of the file
                    Beh_enc_files_path = ub_wind_path(Beh_enc_files_path, system=sys_use) ##function to convert paths windows-linux
                    behaviour=genfromtxt(Beh_enc_files_path, skip_header=1) ## load the file
                    
                    
                    p_target = array(behaviour[:-1,4]) ## Get the position (hypotetical channel coef)
                    ref_time=behaviour[-1, 1] ## Reference time (start scanner - begiinging of recording)
                    st_delay = behaviour[:-1, 11] -ref_time #start of the delay time & take off the reference from
                    
                    
                    hd = 6 # hemodynmic delay  SOURCE OF ERROR!!!!!!!
                    start_delay_hdf = st_delay + hd # add some seconds for the hemodynamic delay
                    
                    #convert seconds to scans (number of scan to take)
                    start_delay_hdf_scans = start_delay_hdf/2.335 
                    timestamps = [  int(round(  start_delay_hdf_scans[n] ) ) for n in range(0, len(start_delay_hdf_scans) )] #make it an integrer
                    #In case  the last one has no space, exclude it (and do the same for the ones of step 1, lin step 3 you will combie and they must have the same length)
                    #you short the timestamps and the matrix fro the hipotetical cannel coefici
                    while timestamps[-1]>len(encoding_datasets[session_enc_sess])-2:
                        timestamps=timestamps[:-1]
                        p_target = p_target[:-1]
                            
                    
                    Enc_delay.append(timestamps) ## append the scan to take
                    Pos_targets.append(p_target) ## append the position of the target for the trial
                    
                    
                    
                    
                    
                    
                    for voxel in range(0, n_voxels ):
                        data_to_filter = encoding_datasets[session_enc_sess][:,voxel]
                        
                        #apply the filter 
                        data_to_filter = TimeSeries(data_to_filter, sampling_interval=2.335)
                        F = FilterAnalyzer(data_to_filter, ub=0.15, lb=0.02)
                        data_filtered=F.filtered_boxcar.data
                        encoding_datasets[session_enc_sess][:,voxel] = data_filtered
                        
                        
                        #Z score
                        #encoding_datasets[session_enc_sess][:,voxel] = list(np.array(zscore(encoding_datasets[session_enc_sess][:,voxel])) + 10 )
                
                
                
                
                enc_lens_datas = [len(encoding_datasets[i]) for i in range(0, len(encoding_datasets))] 
                
                ##### 2. Behaviour
                
                #Load and save the matching behavioural files
                
                Pos_targets=[]
                Enc_delay=[]
                
                #Get the timestamps I want in the imaging from the behaviour
                for i in range(0, len(Beh_enc_files)):
                    #
                    Beh_enc_files_path = Beh_enc_files[session_enc_sess]
                    Beh_enc_files_path = ub_wind_path(Beh_enc_files_path, system=sys_use)
                    behaviour=genfromtxt(Beh_enc_files_path, skip_header=1)
                    ## Get the position (hypotetical channel coef)
                    p_target = array(behaviour[:-1,4])
                    ref_time=behaviour[-1, 1]
                    st_delay = behaviour[:-1, 11] -ref_time
                    
                    # take at least 6 sec for the hrf
                    hd = 6 #6
                    start_delay_hdf = st_delay + hd
                    
                    #timestamps to take (first)
                    start_delay_hdf_scans = start_delay_hdf/2.335
                    timestamps = [  int(round(  start_delay_hdf_scans[n] ) ) for n in range(0, len(start_delay_hdf_scans) )]
                    
                    #In case  the last one has no space, exclude it (and do the same for the ones of step 1, lin step 3 you will combie and they must have the same length)
                    #you short the timestamps and the matrix fro the hipotetical cannel coefici
                    while timestamps[-1]>len(encoding_datasets[session_enc_sess])-2:
                        #print 1
                        timestamps=timestamps[:-1]
                        p_target = p_target[:-1]
                            
                    
                    Enc_delay.append(timestamps)
                    Pos_targets.append(p_target)
                
                
                
                add_timestamps = [0]+list(cumsum(enc_lens_datas))[:-1]
                for i in range(0, len(Enc_delay)):
                    Enc_delay[i] = list(array(Enc_delay[i])+add_timestamps[i])
                
                
                start_delay=hstack(Enc_delay)
                
                #Now you have the timestamps (start_delay) and the Positions to imput in the f function (Pos_targets)
                #Make the matrix of the activity I want in the voxels I want
                masked_data = vstack(encoding_datasets)
                Matrix_activity = zeros(( len(start_delay), shape(masked_data)[1] ))
                
                #Take the mean of two TR    
                for idx,t in enumerate(start_delay):
                    example_ts = masked_data[t:t+2, :]
                    trial = mean(example_ts, axis=0)
                    Matrix_activity[idx, :] =trial
                
                
                #### zscore by voxel after chosing the 2TR and make the mean
                for vxl in range(0, shape(Matrix_activity)[1] ):
                    vx_act = Matrix_activity[:, vxl]
                    vx_act_zs = np.array( zscore(vx_act) ) +10 ;
                    Matrix_activity[:, vxl] = vx_act_zs
                
                ####
                # Get the hypothetical channel coeficients: the activity we expect for each channel in every trial of the behaviour
                pos_target=hstack(Pos_targets)
                
                Matrix_all=[]
                for i in pos_target:
                    channel_values=f(i)  #f #f_quadrant
                    Matrix_all.append(channel_values)
                    
                
                
                M_model=array(Matrix_all)
                
                
                
                ###############################  STEP 3 ###############################
                
                #For each voxel, I want to extract weight of each channel of our model
                #Con qué peso de canales explico mejor la actividad de este voxel a a lo largo de los trials))
                #Right now I will combine the two previous steps
                #Que canal explica mejor la activid de este voxel a lo largo de todos los trials? --> Weight para cada canal
                #If I have a voxel that responds to 27, the weight of the first channel is going to be hight because it means that the activity I have fits really weel with the activity I 
                # expect from the first channel
                
                
                channel_names = ['ch_' +str(i+1) for i in range(0, len(pos_channels))]
                Matrix_weights=zeros((shape(Matrix_activity)[1], len(pos_channels) ))
                
                
                for voxel_x in range(0, shape(Matrix_activity)[1]):
                    LM_matrix = pd.DataFrame(data=M_model)
                    LM_matrix.columns=channel_names
                    LM_matrix['Y']=Matrix_activity[:, voxel_x]
                    ###### Liniar model
                #    mod = ols(formula='Y ~ ' +  ' + '.join(channel_names) , data=LM_matrix).fit()
                #    betas=mod.params[1:]
                #    Matrix_weights[voxel_x, :]=betas
                    ### Regularization
                #    clf = Ridge(alpha=0.01, fit_intercept=True, normalize=False) #0.01
                #    clf.fit(LM_matrix[channel_names], LM_matrix['Y'])
                #    Matrix_weights[voxel_x, :]=clf.coef_
                    ### Regularization Lasso
                    clf = Lasso(alpha=0.001, fit_intercept=True, normalize=False) #0.01
                    clf.fit(LM_matrix[channel_names], LM_matrix['Y'])
                    Matrix_weights[voxel_x, :]=clf.coef_
                    ##### Liniar model 2
                #    a = sm.OLS(LM_matrix['Y'], LM_matrix[channel_names] )
                #    resul = a.fit()
                #    betas= resul.params
                #    Matrix_weights[voxel_x, :]=betas  
                
                
                
                