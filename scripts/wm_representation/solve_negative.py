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
        for CONDITION in ['1_0.2']: #, '1_7', '2_0.2', '2_7'
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
            df_responses=[] ##append the result of the reconstruction per session (just one when together)
            dfs = {}
            
            for session_enc in range(0,len(func_encoding_sess)):
                ### when together, this will run once.
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
                
                #####################
                ##################### STEP 1: GET DATA AND APPLY THE MASK
                #####################
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
                
                
                #####################
                ##################### STEP 2: PROCESS TRAINGN DATA
                #####################
                
                #### TRAING DATA: ALL SESSIONS TOGETHER
                
                ###### In each session I will:
                    ####   1. Select the times corresponding to the delay (2TR), append the target of each trial 
                    ####   2. Apply a filter for each voxel
                    ####   3. Subset of data corresponding to the delay times (all voxels)
                    ####   4. zscore + 10 in each voxel in the temporal dimension (with the other 2TR of the same session)
                    ####   5. append activity and targets of the session
                ####
                ###### Concatenate all the sessions (targets and activity) to create the training dataset
                
                Training_dataset_activity=[] ##  activity for all the trials (all the sessions) (trials, voxels)
                Training_dataset_targets=[] ##  targets for all the trials (all the sessions) (trials)
                n_voxels = shape(encoding_datasets[0])[1] ## number of voxels
                
                for session_enc_sess in range(0, len(enc_lens_datas)):                  
                    
                    ### 1. Select the times corresponding to the delay (2TR), append the target of each trial
                    Enc_delay=[] ## Get the scans to take from the data (beggining of the delay)
                    
                    ## load the file
                    Beh_enc_files_path = Beh_enc_files[session_enc_sess] ## name of the file
                    Beh_enc_files_path = ub_wind_path(Beh_enc_files_path, system=sys_use) ##function to convert paths windows-linux
                    behaviour=genfromtxt(Beh_enc_files_path, skip_header=1) ## load the file
                    
                    
                    p_target = array(behaviour[:-1,4]) ## Get the position (hypotetical channel coef)
                    
                    ### shuffle trial labels
                    v= list( p_target)
                    import random
                    random.shuffle(v)
                    p_target = v
                    
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
                        timestamps=timestamps[:-1] ##  1st scan to take in each trial
                        p_target = p_target[:-1] ## targets of the session (append to the genearl at the end)
                    
                    
                                       
                    ####   2. Apply a filter for each voxel               
                    for voxel in range(0, n_voxels ):
                        data_to_filter = encoding_datasets[session_enc_sess][:,voxel] #data of the voxel along the session
                        
                        #apply the filter 
                        data_to_filter = TimeSeries(data_to_filter, sampling_interval=2.335)
                        F = FilterAnalyzer(data_to_filter, ub=0.15, lb=0.02)
                        data_filtered=F.filtered_boxcar.data
                        encoding_datasets[session_enc_sess][:,voxel] = data_filtered ## replace old data with the filtered one.
                    
                    
                    ####   3. Subset of data corresponding to the delay times (all voxels)
                    encoding_delay_activity = zeros(( len(timestamps), n_voxels)) ## emply matrix (n_trials, n_voxels)
                    for idx,t in enumerate(timestamps): #in each trial
                        delay_TRs =  encoding_datasets[session_enc_sess][t:t+2, :] #take the first scan of the delay and the nex
                        delay_TRs_mean = mean(delay_TRs, axis=0) #make the mean in each voxel of 2TR
                        encoding_delay_activity[idx, :] =delay_TRs_mean #index the line in the matrix
                    
                    
                    ####   4. zscore + 10 in each voxel in the temporal dimension (with the other 2TR of the same session)
                    for vxl in range(0, n_voxels ): # by voxel
                        vx_act = encoding_delay_activity[:, vxl]
                        vx_act_zs = np.array( zscore(vx_act) ) +10 ; ## zscore + 10 just to get + values
                        encoding_delay_activity[:, vxl] = vx_act_zs  ## replace previos activity
                    
                    
                    ####   5. append activity and targets of the session
                    p_target = list(p_target) ### make a list that will be added to another list
                    Training_dataset_targets.extend(p_target) ## append the position of the target for the trial
                    #
                    Training_dataset_activity.append(encoding_delay_activity) ## append the activity used for the training
                
                
                ##### Concatenate sessions to create Trianing Dataset  ### ASSUMPTION: each voxel is the same across sessions!               
                Training_dataset_activity = vstack(Training_dataset_activity) #make an array (n_trials(all sessions together), voxels)
                Training_dataset_targets = array(Training_dataset_targets) ## make an array (trials, 1)
                
                ###############################
                ###############################  STEP 3: TRAIN THE MODEL
                ###############################
                #For each voxel, I want to extract weight of each channel of our model
                #Con qué peso de canales explico mejor la actividad de este voxel a a lo largo de los trials))
                #Right now I will combine the two previous steps
                #Que canal explica mejor la activid de este voxel a lo largo de todos los trials? --> Weight para cada canal
                #If I have a voxel that responds to 27, the weight of the first channel is going to be hight because it means that the activity I have fits really weel with the activity I 
                # expect from the first channel
                
                ####   1. Generate hypothetical activity per trial
                ####   2. Train the model and get matrix of weights
                ####                  
                ####   1. Generate hypothetical activity per trial                
                M_model=[] #matrix of the activity from the model
                for i in Training_dataset_targets:
                    channel_values=f(i)  #f #f_quadrant (function that generates the expectd reponse in each channel)
                    M_model.append(channel_values)
                    
                M_model=pd.DataFrame(array(M_model)) # (trials, channel_activity)
                channel_names = ['ch_' +str(i+1) for i in range(0, len(pos_channels))] #names of the channels 
                M_model.columns=channel_names
                
                
                ####   2. Train the model and get matrix of weights
                Matrix_weights=zeros(( n_voxels, len(pos_channels) )) # (voxels, channels) how each channels is represented in each voxel
                
                for voxel_x in range(0, n_voxels): #train each voxel
                    # set Y and X for the GLM
                    Y = Training_dataset_activity[:, voxel_x] ## Y is the real activity
                    X = M_model ## X is the hipothetycal activity 
                    
                    #### Lasso with penalization of 0.0001 (higher gives all zeros), fiting intercept (around 10 ) and forcing the weights to be positive
                    lin = Lasso(alpha=0.001, precompute=True,  fit_intercept=True,  positive=True, selection='random')   
                    lin.fit(X,Y) # fits the best combination of weights to explain the activity
                    betas = lin.coef_ #ignore the intercept and just get the weights of each channel
                    Matrix_weights[voxel_x, :]=betas #save the 36 weights for each voxel
                
                
                #####
                
                #Save the matrix of weights 
                Matrix_save=pd.DataFrame(Matrix_weights) #convert the array to dataframe
                Matrix_save.to_excel(writer_matrix,'sheet{}'.format(session_enc))
                Matrix_weights_transpose=Matrix_weights.transpose() #create the transpose for the IEM
                os.chdir(encoding_path)
                
                ### WM REPRESENTATION
                ###
                ###
                ###
                ###
                ###
                #### 1. Imaging                
                #Extract the encoded channel response from the WM task trails
                ## signal = weights * channels
                #  weights-1 * signal = weights-1*weights * channel --> weights-1 * signal = channeö  --> This is not allowed because is not invertable (it is not a square matrix)
                # solution:    w.t * sig = w.t * w * ch -->  (w.t * w)-1  * w.t  * sig  =  (w.t * w)-1 *  (w.t * w) * ch --->  (w.t * w)-1  * w.t  * sig  = ch
                ## Python implementation of the function
                #channel = dot( dot ( inv( dot(Matrix_weights_transpose, Matrix_weights ) ),  Matrix_weights_transpose),  signal)
                
                ##########################################
                ##########################################
                ####  1. Get the data and apply the mask
                ####  2. Process the data (filter and zscore)
                ##########################################
                ##########################################
                Testing_dataset_activity=[] ##  activity for all the trials (all the sessions) (trials, voxels)
                Testing_dataset_beh =[] ##  behavioural data for all the trials (all the sessions) (trials)
                nscans_wm = 16
                ## 1. Get the data and apply the mask
                #
                for session_wm in range(0, len(func_wmtask)):
                    func_filename=func_wmtask[session_wm] # get the file name
                    func_filename = ub_wind_path(func_filename, system=sys_use)#change depending windows/unix                 
                    mask_img_rh=path_masks + Maskrh
                    mask_img_rh = ub_wind_path(mask_img_rh, system=sys_use)
                    mask_img_lh=path_masks + Masklh 
                    mask_img_lh = ub_wind_path(mask_img_lh, system=sys_use)
                    ##Apply the masks and concatenate   
                    masked_data_rh = apply_mask(func_filename, mask_img_rh) 
                    masked_data_lh = apply_mask(func_filename, mask_img_lh)    
                    masked_data=hstack([masked_data_rh, masked_data_lh]) #merge rh and lh voxels 
                    n_voxels_wm = shape(masked_data)[1] ##number of voxels
                    ##
                    ## 2. Process the data (filter and zscore)
                    #High-pass filter & zscore per voxel in the temporal domain
                    for voxel in range(0, n_voxels_wm):
                        data_to_filter = masked_data[:,voxel]                        
                        #apply the filter 
                        data_to_filter = TimeSeries(data_to_filter, sampling_interval=2.335)
                        F = FilterAnalyzer(data_to_filter, ub=0.15, lb=0.02)
                        data_filtered=F.filtered_boxcar.data
                        masked_data[:,voxel] = data_filtered                        
                        #Z score
                        masked_data[:,voxel] = np.array( zscore( masked_data[:,voxel]  ) ) +5 ; ## zscore + 5 just to get + values
                    
                    
                    #             
                    # Behaviour 
                    Beh_WM_files_path = Beh_WM_files[session_wm] #path of the file of the corresponding session
                    Beh_WM_files_path = ub_wind_path(Beh_WM_files_path, system=sys_use) #change depending on windoxs/unix
                    behaviour=genfromtxt(Beh_WM_files_path, skip_header=1) #open the file
                    Beh = pd.DataFrame(behaviour)  #convert it to dataframe
                    headers_col = ['type', 'delay1', 'delay2', 'T', 'NT1', 'NT2', 'Dist', 'Dist_NT1', 'Dist_NT2', 'distance_T_dist', 'cue', 'order',
                                'orient', 'horiz_vertical', 'A_R', 'A_err', 'Abs_angle_error', 'Error_interference', 'A_DC', 'A_DC_dist', 'Q_DC', 
                                'A_DF', 'A_DF_dist', 'Q_DF', 'A_DVF', 'Q_DVF', 'A_DVF_dist', 'Q_DVF_dist', 'presentation_att_cue_time', 'presentation_target_time',
                                'presentation_dist_time', 'presentation_probe_time', 'R_T', 'trial_time', 'disp_time']
                    Beh.columns=headers_col #add columns
                    #take off the reference    
                    ref_time = Beh.iloc[-1, 1] # get the reference(diff between tsat¡rt the display and start de recording)
                    start_trial=Beh['presentation_att_cue_time'].iloc[0:-1]  - ref_time #take off the reference  
                    Beh = Beh.iloc[0:-1, :] # behaviour is the same except the last line (reference time) 
                    start_trial_hdf_scans = start_trial/2.335 #transform seconds to scans 
                    timestamps = [  int(round(  start_trial_hdf_scans[n] ) ) for n in range(0, len(start_trial_hdf_scans) )]
                    
                    #adjust according to the number of scans you want (avoid having an incomplete trial)
                    while timestamps[-1]>len(masked_data)-nscans_wm:
                        timestamps=timestamps[:-1] #take off one trial form activity
                        Beh = Beh.iloc[0:-1, :] #take off one trial from behaviour
                        
                            
                    #append the timestands you want from this session
                    n_trials = len(timestamps)
                    Testing_dataset_beh.append(Beh)
                    
                    ### Take the important TRs (from cue, the next 14 TRs)
                    signal_session=np.zeros(( n_trials, nscans_wm,  n_voxels_wm   )) ## np.zeros matrix with the correct dimensions of the session
                    for idx, t in enumerate(timestamps): #beginning of the trial
                        for sc in range(0, nscans_wm): #each of the 14 TRs
                            trial_activity = masked_data[t+sc, :]   
                            signal_session[idx, sc, :] =trial_activity                    
                    
                    ### 
                    Testing_dataset_activity.append(signal_session) ## append the reults of the session
                
                
                ### Concatenate the session results
                signal = np.vstack(Testing_dataset_activity)
                Behaviour = pd.concat(Testing_dataset_beh)
                
#                ##### slip cw_ccw
#                ccw = Behaviour['A_R'] > Behaviour['T'] 
#                Behaviour['resp_cw_ccw'] = ccw
#                Behaviour['resp_cw_ccw'] = Behaviour['resp_cw_ccw'].replace([True, False], ['ccw', 'cw'])
#                
#                if distance=='mix':
#                    if CONDITION == '1_0.2':
#                        Subset = signal[  array(Behaviour['delay1']==0.2)  *  array(Behaviour['order']==1) *  array(Behaviour['resp_cw_ccw']=='cw') , :, :]
#                        beh_Subset = Behaviour.loc[(Behaviour['delay1']==0.2) & (Behaviour['order']==1) & (Behaviour['resp_cw_ccw']=='cw')   ] #Behaviour[Behaviour[:,1]==0.2, :]
                
                
                
                ######### Distance (mix when not important, else when close or far)
                if distance=='mix':                    
                    if CONDITION == '1_0.2': 
                        Subset = signal[  array(Behaviour['delay1']==0.2)  *  array(Behaviour['order']==1) , :, :]
                        beh_Subset = Behaviour.loc[(Behaviour['delay1']==0.2) & (Behaviour['order']==1)  ] 
                      
                    elif CONDITION == '1_7':
                        Subset = signal[  array(Behaviour['delay1']==7)  *  array(Behaviour['order']==1) , :, :]
                        beh_Subset = Behaviour.loc[(Behaviour['delay1']==7) & (Behaviour['order']==1)  ] 
                        
                    elif CONDITION == '2_0.2':
                        Subset = signal[  array(Behaviour['delay1']==0.2)  *  array(Behaviour['order']==2)   , :, :]
                        beh_Subset = Behaviour.loc[(Behaviour['delay1']==0.2) & (Behaviour['order']==2) ] 
                      
                    elif CONDITION == '2_7':
                        Subset = signal[  array(Behaviour['delay1']==7)  *  array(Behaviour['order']==2)  , :, :]
                        beh_Subset = Behaviour.loc[(Behaviour['delay1']==7) & (Behaviour['order']==2)  ] 
                
                
                else: ### close or far
                    if CONDITION == '1_0.2':
                        Subset = signal[  array(Behaviour['delay1']==0.2)  *  array(Behaviour['order']==1) *  array(Behaviour['type']==distance)  , :, :]
                        beh_Subset = Behaviour.loc[(Behaviour['delay1']==0.2) & (Behaviour['order']==1) & (Behaviour['type']==distance) ] 
                      
                    elif CONDITION == '1_7':
                        Subset = signal[  array(Behaviour['delay1']==7)  *  array(Behaviour['order']==1) * array(Behaviour['type']==distance)  , :, :]
                        beh_Subset = Behaviour.loc[(Behaviour['delay1']==7) & (Behaviour['order']==1) & (Behaviour['type']==distance)  ] 
                        
                    elif CONDITION == '2_0.2':
                        Subset = signal[  array(Behaviour['delay1']==0.2)  *  array(Behaviour['order']==2) * array(Behaviour['type']==distance)  , :, :]
                        beh_Subset = Behaviour.loc[(Behaviour['delay1']==0.2) & (Behaviour['order']==2) & (Behaviour['type']==distance)  ] 
                      
                    elif CONDITION == '2_7':
                        Subset = signal[  array(Behaviour['delay1']==7)  *  array(Behaviour['order']==2) *  array(Behaviour['type']==distance) , :, :]
                        beh_Subset = Behaviour.loc[(Behaviour['delay1']==7) & (Behaviour['order']==2) & (Behaviour['type']==distance)  ] 
                    
                
                
                ####
                title= CONDITION
                ref_angle = 45 #Reference channel to center all
                
                ### shuffle trial labels
                v= list( beh_Subset['T'])
                import random
                random.shuffle(v)
                beh_Subset['T']=v
                
                #Lists to append all the trials rolled
                Channel_all_trials_rolled=[]
                
                for trial in range(0, len(beh_Subset)): #each trial of the condition
                    channels_trial=[] #List to append all the trial channels
                    
                    for time_scan in range(0, nscans_wm, 2 ): ## each time of the trial
                        #Get the activity of the TR in the trial      
                        Signal = array([Subset[trial,time_scan,:], Subset[trial,time_scan + 1,:]]).mean(axis=0) #mean of 2TR
                        
                        #Run the inverse model
                        channel1 = dot( dot ( inv( dot(Matrix_weights_transpose, Matrix_weights ) ),  Matrix_weights_transpose),  Signal)
                        
                        ##Convert 36 into 720 channels for the reconstruction
                        channel= ch2vrep3(channel1) ##function
                        
                        #Roll
                        angle_trial =  beh_Subset['T'].iloc[trial] ## get the angle of the target
                        to_roll = int( (ref_angle - angle_trial)*(len(channel)/360) ) ## degrees to roll
                        channel=roll(channel, to_roll) ## roll this degrees
                        channels_trial.append(channel) #Append it into the trial list
                        ####
                        
                    
                    #Once all the TR of the trial are in the trial list, append this list to the global list
                    Channel_all_trials_rolled.append(channels_trial)
                
                
                Channel_all_trials_rolled = array(Channel_all_trials_rolled)  # (trials, TRs, channels_activity) (of the session (whne together, all))
                
                #Mean of trials
                df = pd.DataFrame()
                for i in range(0, nscans_wm/2):
                    n = list(Channel_all_trials_rolled[:,i,:].mean(axis=0)) #mean of all the trials rolled
                    df[str( round(2.335*i, 2)  )] = n #name of the column
                
                
                ## plot heatmap
                #plt.figure()
                #plt.title('Heatmap decoding ' + Subject_analysis)
                ########midpoint = df.values.mean() # (df.values.max() - df.values.min()) / 2
                #ax = sns.heatmap(df, yticklabels=list(df.index), cmap="coolwarm") # cmap= viridis "jet",  "coolwarm" RdBu_r, gnuplot, YlOrRd, CMRmap  , center = midpoint
                #ax.plot([0.25, shape(df)[1]-0.25], [posch1_to_posch2(4),posch1_to_posch2(4)], 'k--')
                #plt.yticks([posch1_to_posch2(4), posch1_to_posch2(13), posch1_to_posch2(22), posch1_to_posch2(31)] ,['45','135','225', '315'])
                #plt.ylabel('Angle')
                #plt.xlabel('time (s)')
                #plt.show(block=False)
                
                #
                ##Append the df  
                #
                #### append a df for the session
                df_responses.append(df)
                dfs[session_enc]=df
                
            
            
            
            ###### After running all the sessions
            #### Save the df of the matrix_weights
            writer_matrix.save()            
            ## Create excel for the response
            df_name = Subject_analysis + '_' + brain_region + '_' + CONDITION + '_' +distance_ch + '_' + Method_analysis + '.xlsx' #name of the file
            writer_cond = pd.ExcelWriter(df_name)
            os.chdir(Conditions_enc_path + CONDITION)
            for session_tsk, df1 in enumerate(df_responses): #for each session
                df1.to_excel(writer_cond,'sheet{}'.format(session_tsk))
            
            
            writer_cond.save() #save the mean rolled by session
            os.chdir(Conditions_enc_path + CONDITION + PLOTS_path)
            
            #mean of sessions (this does nothing with together)
            panel=pd.Panel(dfs) 
            df=panel.mean(axis=0)
            
            ####
            ### PLOTS
            ############# 1. Heatmap of the region
            ############# 2. lineplor preferred 
            ####
            
            #### Heatmap region (save)
            plt.figure() 
            TITLE_HEATMAP = Subject_analysis + '_' + brain_region + '_' + CONDITION + '_' +distance_ch + '_' + Method_analysis + ' heatmap'
            plt.title(TITLE_HEATMAP)
            ax = sns.heatmap(df, yticklabels=list(df.index), cmap="coolwarm") # cmap= viridis "jet",  "coolwarm" RdBu_r, gnuplot, YlOrRd, CMRmap  , center = midpoint
            ax.plot([0.25, shape(df)[1]-0.25], [posch1_to_posch2(4),posch1_to_posch2(4)], 'k--') ##line of the average
            plt.yticks([posch1_to_posch2(4), posch1_to_posch2(13), posch1_to_posch2(22), posch1_to_posch2(31)] ,['45','135','225', '315'])
            plt.ylabel('Angle')
            plt.xlabel('time (s)')
            plt.show(block=False)
            TITLE_PLOT_H = Subject_analysis + '_' + brain_region + '_' + CONDITION + '_' +distance_ch + '_' + Method_analysis + ' heatmap.png'
            plt.savefig(TITLE_PLOT_H)
            plt.close()
            ##
            
            #### line plot preferred (save)
            ref_angle=45
            Angle_ch = ref_angle * (len(channel) / 360)            
            df_45 = df.iloc[int(Angle_ch)-20 : int(Angle_ch)+20] #just take the ones around the preferred
            df_together = df_45.melt()
            df_together['ROI'] = ['ips' for i in range(0, len(df_together))]
            df_together['voxel'] = [i+1 for i in range(0, len(df_45))]*shape(df_45)[1]
            df_together.columns = ['timepoint', 'Decoding', 'ROI', 'voxel']
            df_together['timepoint'] = [float(df_together['timepoint'].iloc[i]) for i in range(0, len(df_together))]
            a=sns.factorplot(x='timepoint', y='Decoding',  data=df_together, size=5, aspect=1.5)
            TITLE_PREFERRED = Subject_analysis + '_' + brain_region + '_' + CONDITION + '_' +distance_ch + '_' + Method_analysis + ' preferred'
            plt.title(TITLE_PREFERRED)
            plt.show(block=False)
            TITLE_PLOT = Subject_analysis + '_' + brain_region + '_' + CONDITION + '_' +distance_ch + '_' + Method_analysis + ' preferred.png'
            a.savefig(TITLE_PLOT)
            plt.close()
                


                
                
                
                