# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 12:12:03 2018

@author: David Bestue
"""

##Decide brain region

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
for algorithm in ["visual", "ips"]:
    for CONDITION in ['1_0.2', '1_7', '2_0.2', '2_7']: 
        Method_analysis = 'bysess'
        #CONDITION = '1_0.2'
        #algorithm = "visual"
        distance_ch='mix'
        distance='mix'
        Subject_analysis='l001' 
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
            
            
            
            ###High-pass filter and z-score per voxel
            
            n_voxels = shape(encoding_datasets[0])[1]
            for session_enc_sess in range(0, len(enc_lens_datas)):
                for voxel in range(0, n_voxels ):
                    data_to_filter = encoding_datasets[session_enc_sess][:,voxel]
                    
                    #apply the filter 
                    data_to_filter = TimeSeries(data_to_filter, sampling_interval=2.335)
                    F = FilterAnalyzer(data_to_filter, ub=0.15, lb=0.02)
                    data_filtered=F.filtered_boxcar.data
                    encoding_datasets[session_enc_sess][:,voxel] = data_filtered
                    
                    
                    #Z score
                    encoding_datasets[session_enc_sess][:,voxel] = zscore(encoding_datasets[session_enc_sess][:,voxel]) 
            
            
            
            
            enc_lens_datas = [len(encoding_datasets[i]) for i in range(0, len(encoding_datasets))] 
            
            ##### 2. Behaviour
            
            #Load and save the matching behavioural files
            
            Pos_targets=[]
            lens_enc_del=[]
            Enc_delay=[]
            
            #Get the timestamps I want in the imaging from the behaviour
            for i in range(0, len(Beh_enc_files)):
                #
                Beh_enc_files_path = Beh_enc_files[i]
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
                while timestamps[-1]>len(encoding_datasets[i])-2:
                    #print 1
                    timestamps=timestamps[:-1]
                    p_target = p_target[:-1]
                        
                
                Enc_delay.append(timestamps)
                lens_enc_del.append(len(timestamps))
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
            
            
            # Get the hypothetical channel coeficients: the activity we expect for each channel in every trial of the behaviour
            pos_target=hstack(Pos_targets)
            
            Matrix_all=[]
            for i in pos_target:
                channel_values=f(i)  #f #f_quadrant
                Matrix_all.append(channel_values)
                
            
            
            M_model=array(Matrix_all)
            
            
            ###############################  STEP 3 ###############################
            
            #For each voxel, I want to extract weight of each channel of our model
            #Con qué peso de canales explico mejor la actividad de este voxel a alo largo de los trials))
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
            
            
            
        #    #Histogram max channel response
        #    plt.figure()
        #    maxch_voxel = [where(Matrix_weights[i]==max(Matrix_weights[i]))[0][0] for i in range(0, len(Matrix_weights))]
        #    h = seaborn.distplot(maxch_voxel)
        #    seaborn.set_style('white')
        #    seaborn.despine()
        #    plt.xticks([0,17,35])
        #    h.set_xticklabels(['5','175','355'])
        #    plt.xlabel("Channels")
        #    plt.title('Max channel voxel')
        #    plt.show(block=False)
        #    
        #    
        #    
        #    #Histogram max channel response
        #    plt.figure()
        #    A = pd.DataFrame(Matrix_weights)
        #    A.columns=channel_names
        #    seaborn.barplot(data=A, estimator=mean, color='darkorange' )
        #    seaborn.set_style('white')
        #    seaborn.despine()
        #    plt.xlabel("Channels")
        #    plt.ylabel("Weight")
        #    plt.title('Mean weight per channel')
        #    plt.xticks(rotation='vertical')
        #    plt.show(block=False)
            
            
            
            # Chhannel weight in the mask
            #Now I have one matrix that is the estimated weight of the channel for each voxel ( Matrix_weights[voxels, weight of the channel]  )
            
            #os.chdir(Matrix_enc_path)
            Matrix_save=pd.DataFrame(Matrix_weights)
            Matrix_save.to_excel(writer_matrix,'sheet{}'.format(session_enc))
            
            Matrix_weights_transpose=Matrix_weights.transpose()
            os.chdir(encoding_path)
            
            ###
            ###
            ###
            ###
            ###
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
            
            #
            WM_lens_datas=[]
            WM_datasets=[]
            
            
            #
            
            for i in range(0, len(func_wmtask)):
                func_filename=func_wmtask[i] # 'regfmcpr.nii.gz'
                func_filename = ub_wind_path(func_filename, system=sys_use)
                
                #func_filename_rh=func_wmtask[i] + 'regfmcprrh.nii.gz'
                #func_filename_lh=func_wmtask[i] + 'regfmcprlh.nii.gz'
                
                mask_img_rh=path_masks + Maskrh
                mask_img_rh = ub_wind_path(mask_img_rh, system=sys_use)
                mask_img_lh=path_masks + Masklh 
                mask_img_lh = ub_wind_path(mask_img_lh, system=sys_use)
                ##Apply the masks and concatenate   
                masked_data_rh = apply_mask(func_filename, mask_img_rh) #func_filename func_filename_rh
                masked_data_lh = apply_mask(func_filename, mask_img_lh) #func_filename   
                
                masked_data=hstack([masked_data_rh, masked_data_lh])
                #append it and save the data
                WM_datasets.append(masked_data)
                WM_lens_datas.append(len(masked_data))
            
            
            
            
            #High-pass filter
            #for each session & for each voxel mean center and apply the filter
            for session_wm in range(0, len(WM_lens_datas)):
                for voxel in range(0, shape(WM_datasets[session_wm])[1] ):
                    data_to_filter = WM_datasets[session_wm][:,voxel]
                    
                    #apply the filter 
                    data_to_filter = TimeSeries(data_to_filter, sampling_interval=2.335)
                    F = FilterAnalyzer(data_to_filter, ub=0.15, lb=0.02)
                    data_filtered=F.filtered_boxcar.data
                    WM_datasets[session_wm][:,voxel] = data_filtered
                    
                    #Z score
                    WM_datasets[session_wm][:,voxel] = zscore( WM_datasets[session_wm][:,voxel]) 
                    
            
            
            
            WM_lens_datas = [len(WM_datasets[i]) for i in range(0, len(WM_datasets))] 
                
                
                
            WM_delay=[]
            nscans_wm = 12 #9
            Behaviour=[]
            
            headers_col = ['type', 'delay1', 'delay2', 'T', 'NT1', 'NT2', 'Dist', 'Dist_NT1', 'Dist_NT2', 'distance_T_dist', 'cue', 'order',
                            'orient', 'horiz_vertical', 'A_R', 'A_err', 'Abs_angle_error', 'Error_interference', 'A_DC', 'A_DC_dist', 'Q_DC', 
                            'A_DF', 'A_DF_dist', 'Q_DF', 'A_DVF', 'Q_DVF', 'A_DVF_dist', 'Q_DVF_dist', 'presentation_att_cue_time', 'presentation_target_time',
                            'presentation_dist_time', 'presentation_probe_time', 'R_T', 'trial_time', 'disp_time']  
            
            
            for i in range(0, len(Beh_WM_files)):
                #Open file
                Beh_WM_files_path = Beh_WM_files[i]
                Beh_WM_files_path = ub_wind_path(Beh_WM_files_path, system=sys_use)
                behaviour=genfromtxt(Beh_WM_files_path, skip_header=1)
                Beh = pd.DataFrame(behaviour) 
                Beh.columns=headers_col
                #take off the reference    
                #ref_time=behaviour[-1, 1]
                ref_time = Beh.iloc[-1, 1] 
                
                #Decide what to take of the trial    
                #start_delay = behaviour[:-1, 29] -ref_time
                start_delay=Beh['presentation_att_cue_time'].iloc[0:-1]  - ref_time
                Beh['presentation_att_cue_time'].iloc[0:-1] 
                Beh = Beh.iloc[0:-1, :] 
                #behaviour=behaviour[:-1,:]
                
                #transform to scans 
                start_delay_hdf_scans = start_delay/2.335
                timestamps = [  int(round(  start_delay_hdf_scans[n] ) ) for n in range(0, len(start_delay_hdf_scans) )]
                
                #adjust according to the number of scans you want
                while timestamps[-1]>len(WM_datasets[i])-nscans_wm:
                    #print 1
                    timestamps=timestamps[:-1]
                    Beh = Beh.iloc[0:-1, :] 
                    #behaviour=behaviour[:-1,:]
                        
                #append the timestands you want from this session
                WM_delay.append(timestamps)
                Behaviour.append(Beh)
            
            
            
            #Put together the timestamps of the sessions
            add_timestamps = [0]+list(cumsum(WM_lens_datas))[:-1]
            for i in range(0, len(WM_delay)):
                WM_delay[i] = list(array(WM_delay[i])+add_timestamps[i])
            
            
            start_delay=hstack(WM_delay)
            
            #Put together the images of the sessions 
            masked_data = vstack(WM_datasets)
            
            ## WM_dataets (sessions, times, voxels)
            ## masked_data (timesteps, voxels)
            
            #Put together the behaviour
            #Behaviour=vstack(Behaviour)
            Behaviour = pd.concat(Behaviour) 
            
            
            
            #Get the matrix (timestamp, number of scans, activity)
            signal = zeros(( len(start_delay), nscans_wm ,shape(masked_data)[1] ))
            for idx,t in enumerate(start_delay):
                for sc in range(0, nscans_wm):
                    example_ts = masked_data[t+sc, :]   
                    signal[idx, sc, :] =example_ts
            
            
            
            ######### Distance (mix when not important, else when close or far)
            if distance=='mix':
                
                if CONDITION == '1_0.2':
                    Subset = signal[  array(Behaviour['delay1']==0.2)  *  array(Behaviour['order']==1) , :, :]
                    beh_Subset = Behaviour.loc[(Behaviour['delay1']==0.2) & (Behaviour['order']==1)  ] #Behaviour[Behaviour[:,1]==0.2, :]
                  
                elif CONDITION == '1_7':
                    Subset = signal[  array(Behaviour['delay1']==7)  *  array(Behaviour['order']==1) , :, :]
                    beh_Subset = Behaviour.loc[(Behaviour['delay1']==7) & (Behaviour['order']==1)  ] #Behaviour[Behaviour[:,1]==0.2, :]
                    
                elif CONDITION == '2_0.2':
                    Subset = signal[  array(Behaviour['delay1']==0.2)  *  array(Behaviour['order']==2)   , :, :]
                    beh_Subset = Behaviour.loc[(Behaviour['delay1']==0.2) & (Behaviour['order']==2) ] #Behaviour[Behaviour[:,1]==0.2, :]
                  
                elif CONDITION == '2_7':
                    Subset = signal[  array(Behaviour['delay1']==7)  *  array(Behaviour['order']==2)  , :, :]
                    beh_Subset = Behaviour.loc[(Behaviour['delay1']==7) & (Behaviour['order']==2)  ] #Behaviour[Behaviour[:,1]==0.2, :]
            
            
            else: ### close or far
                
                if CONDITION == '1_0.2':
                    Subset = signal[  array(Behaviour['delay1']==0.2)  *  array(Behaviour['order']==1) *  array(Behaviour['type']==distance)  , :, :]
                    beh_Subset = Behaviour.loc[(Behaviour['delay1']==0.2) & (Behaviour['order']==1) & (Behaviour['type']==distance) ] #Behaviour[Behaviour[:,1]==0.2, :]
                  
                elif CONDITION == '1_7':
                    Subset = signal[  array(Behaviour['delay1']==7)  *  array(Behaviour['order']==1) * array(Behaviour['type']==distance)  , :, :]
                    beh_Subset = Behaviour.loc[(Behaviour['delay1']==7) & (Behaviour['order']==1) & (Behaviour['type']==distance)  ] #Behaviour[Behaviour[:,1]==0.2, :]
                    
                elif CONDITION == '2_0.2':
                    Subset = signal[  array(Behaviour['delay1']==0.2)  *  array(Behaviour['order']==2) * array(Behaviour['type']==distance)  , :, :]
                    beh_Subset = Behaviour.loc[(Behaviour['delay1']==0.2) & (Behaviour['order']==2) & (Behaviour['type']==distance)  ] #Behaviour[Behaviour[:,1]==0.2, :]
                  
                elif CONDITION == '2_7':
                    Subset = signal[  array(Behaviour['delay1']==7)  *  array(Behaviour['order']==2) *  array(Behaviour['type']==distance) , :, :]
                    beh_Subset = Behaviour.loc[(Behaviour['delay1']==7) & (Behaviour['order']==2) & (Behaviour['type']==distance)  ] #Behaviour[Behaviour[:,1]==0.2, :]
                
            
            
            ####
            title= CONDITION
            
            #########
            #Subset trials
            #Take the delay 1== 7 or 0.2
            #Subset = signal[Behaviour['delay1']==0.2, :, :]
            #beh_Subset = Behaviour.loc[Behaviour['delay1']==0.2, :]
            #title= '1_0.2 delay'    
            
            #Reference channel to center all
            ref_angle = 45
            
            #Lists to append
            Channel_all_trials_rolled=[]
            
            for trial in range(0, len(beh_Subset)):
                #List to append all the trial channels
                channels_trial=[]
                
                for time_scan in range(0, shape(Subset)[1] ):
                    #Get the activity of the TR in the trial        
                    Signal = Subset[trial,time_scan,:]
                    #Run the inverse model
                    channel1 = dot( dot ( inv( dot(Matrix_weights_transpose, Matrix_weights ) ),  Matrix_weights_transpose),  Signal)
                    #Roll the channel to the ref channel (according to its supposed max)        
                    #Reconstruct
                    channel= ch2vrep3(channel1)
                    
                    #Roll
                    #ref_angle =  ref_angle_q [get_quadrant(beh_Subset[trial, 14]) - 1] #3 14
                    angle_trial =  beh_Subset['A_R'].iloc[trial] #3 14
                    to_roll = int( (ref_angle - angle_trial)*(len(channel)/360) )
                    #to_roll = int(round((ref_angle_q[ref_angle] - beh_Subset[trial, 3])*(len(channel)/360)))
                    #to_roll_neg = -(720 - (int(round((ref_angle - beh_Subset[trial, 3])*(len(channel)/360)))))
                    channel=roll(channel, to_roll)
                    
                    #Roll to the second quadrant
            #        if get_quadrant(ref_angle) == 1:
            #            channel=roll(channel, int(90*(len(channel)/360)))
            #        elif get_quadrant(ref_angle) == 3:
            #            channel=roll(channel, int(-90*(len(channel)/360)))
            #        elif get_quadrant(ref_angle) == 4:
            #            channel=roll(channel, int(-180*(len(channel)/360)))            
                    
                    #Append it in the trial list
                    channels_trial.append(channel)
                    ####
                    
                
                #Once all the TR of the trial are in the trial list, append this list to the global list
                Channel_all_trials_rolled.append(channels_trial)
            
            
            Channel_all_trials_rolled = array(Channel_all_trials_rolled)  
            
            
            
            #Heatmap
            plt.figure()
            df = pd.DataFrame()
            for i in range(0,9):
                n = list(Channel_all_trials_rolled[:,i,:].mean(axis=0))
                #n.reverse()
                df[str( round(2.335*i, 2)  )] = n
            
            
            plt.title('Heatmap decoding ' + Subject_analysis)
            #midpoint = df.values.mean() # (df.values.max() - df.values.min()) / 2
            ax = sns.heatmap(df, yticklabels=list(df.index), cmap="coolwarm") # cmap= viridis "jet",  "coolwarm" RdBu_r, gnuplot, YlOrRd, CMRmap  , center = midpoint
            ax.plot([0.25, shape(df)[1]-0.25], [posch1_to_posch2(4),posch1_to_posch2(4)], 'k--')
            plt.yticks([posch1_to_posch2(4), posch1_to_posch2(13), posch1_to_posch2(22), posch1_to_posch2(31)] ,['45','135','225', '315'])
            plt.ylabel('Angle')
            plt.xlabel('time (s)')
            plt.show(block=False)
            
            #
            ##Append the df  
            #
            #### append a df for each enc_session
            df_responses.append(df)
            dfs[session_enc]=df
            
        
        
        
        
        #### Save the df of the matrix_weights
        writer_matrix.save()
        
        ## Create excel for the response
        
        df_name = Subject_analysis + '_' + algorithm + '_' + CONDITION + '_' +distance_ch + '_' + Method_analysis + '.xlsx' 
        writer_cond = pd.ExcelWriter(df_name)
        
        os.chdir(Conditions_enc_path + CONDITION)
        for session_tsk, df1 in enumerate(df_responses):
            df1.to_excel(writer_cond,'sheet{}'.format(session_tsk))
        
        
        writer_cond.save() 
        
        #os.chdir('C:\\Users\\David\\Dropbox\\KAROLINSKA\\encoding_model')
        
        
        #df_p0 =pd.read_excel(df_name, 'sheet0') 
        #df_p1 =pd.read_excel(df_name, 'sheet1') 
        #df_p2 =pd.read_excel(df_name, 'sheet2') 
        #df_ps={} 
        #df_ps['0']=df_p0  
        #df_ps['1']=df_p1  
        #df_ps['2']=df_p2  
        #pp = pd.Panel(df_ps) 
        #df_all = pp.mean(axis=0)  
        
        
        #
        #dfs_pp={} 
        #for s in range(0,3):
        #    dfs_pp[str(s)]=df_responses[s]
        #    
        
        #Mean Dataframes
        #panel=pd.Panel(dfs_pp)
        
        
        os.chdir(Conditions_enc_path + CONDITION + PLOTS_path)
        
        
        
        panel=pd.Panel(dfs)
        df=panel.mean(axis=0)
           
        
        #Heatmap
        plt.figure()
        df = pd.DataFrame()
        for i in range(0,9):
            n = list(Channel_all_trials_rolled[:,i,:].mean(axis=0))
            #n.reverse()
            df[str( round(2.335*i, 2)  )] = n
        
        
        
        TITLE_HEATMAP = Subject_analysis + '_' + algorithm + '_' + CONDITION + '_' +distance_ch + '_' + Method_analysis + ' heatmap'
        plt.title(TITLE_HEATMAP)
        #midpoint = df.values.mean() # (df.values.max() - df.values.min()) / 2
        ax = sns.heatmap(df, yticklabels=list(df.index), cmap="coolwarm") # cmap= viridis "jet",  "coolwarm" RdBu_r, gnuplot, YlOrRd, CMRmap  , center = midpoint
        #ax.invert_yaxis()
        ax.plot([0.25, shape(df)[1]-0.25], [posch1_to_posch2(4),posch1_to_posch2(4)], 'k--')
        plt.yticks([posch1_to_posch2(4), posch1_to_posch2(13), posch1_to_posch2(22), posch1_to_posch2(31)] ,['45','135','225', '315'])
        plt.ylabel('Angle')
        plt.xlabel('time (s)')
        plt.show(block=False)
        TITLE_PLOT_H = Subject_analysis + '_' + algorithm + '_' + CONDITION + '_' +distance_ch + '_' + Method_analysis + ' heatmap.png'
        plt.savefig(TITLE_PLOT_H)
        
        
        #### TSplot
        ref_angle=45
        Angle_ch = ref_angle * (len(channel) / 360)
        
        df_45 = df.iloc[int(Angle_ch)-20 : int(Angle_ch)+20]
        df_together = df_45.melt()
        df_together['ROI'] = ['ips' for i in range(0, len(df_together))]
        df_together['voxel'] = [i+1 for i in range(0, len(df_45))]*shape(df_45)[1]
        df_together.columns = ['timepoint', 'Decoding', 'ROI', 'voxel']
        df_together['timepoint'] = [float(df_together['timepoint'].iloc[i]) for i in range(0, len(df_together))]
        
        plt.figure()
        plt.title('ROI decoding preferred')
        sns.tsplot(time='timepoint', value='Decoding', condition='ROI', unit='voxel', ci='sd', data=df_together)
        plt.show(block=False)
        
        
        #### FactorPlot
        a=sns.factorplot(x='timepoint', y='Decoding',  data=df_together, size=5, aspect=1.5)
        TITLE_PREFERRED = Subject_analysis + '_' + algorithm + '_' + CONDITION + '_' +distance_ch + '_' + Method_analysis + ' preferred'
        plt.title(TITLE_PREFERRED)
        plt.show(block=False)
        TITLE_PLOT = Subject_analysis + '_' + algorithm + '_' + CONDITION + '_' +distance_ch + '_' + Method_analysis + ' preferred.png'
        a.savefig(TITLE_PLOT)
        
        
        #### FactorPlot all
        df_all = df.melt()
        df_all['ROI'] = ['ips' for i in range(0, len(df_all))]
        df_all['voxel'] = [i+1 for i in range(0, len(df))]*shape(df)[1]
        df_all.columns = ['timepoint', 'Decoding', 'ROI', 'voxel']
        df_all['timepoint'] = [float(df_all['timepoint'].iloc[i]) for i in range(0, len(df_all))]
        sns.factorplot(x='timepoint', y='Decoding',  data=df_all)
        plt.title('ROI decoding brain region')
        plt.show(block=False)
        
        
        
        
        
        
        
        ######################   Preferred by session
        
        #df_sess_pf={}
        #
        ##df_responses
        #
        #for i, Sess in enumerate( df_responses ):
        #    df1 = df_responses[i]
        #    df_45 = df1.iloc[int(Angle_ch)-20 : int(Angle_ch)+20]
        #    df_together = df_45.melt()
        #    df_together['ROI'] = ['visual' for i in range(0, len(df_together))]
        #    df_together['voxel'] = [int(Angle_ch)-20 + i for i in range(0, len(df_45))]*shape(df_45)[1]
        #    df_together.columns = ['timepoint', 'Decoding', 'ROI', 'voxel']
        #    df_together['timepoint'] = [float(df_together['timepoint'].iloc[i]) for i in range(0, len(df_together))]
        #    df_sess_pf[i] = df_together
        #
        #all_sessions = pd.concat(df_sess_pf)
        #
        #sns.factorplot(x='timepoint', y='Decoding',  data=all_sessions)
        #TITLE_PREFERRED = Subject_analysis + '_' + algorithm + '_' + CONDITION + '_' +distance_ch + '_' + Method_analysis + ' preferred'
        #plt.title(TITLE_PREFERRED)
        #plt.show(block=False)
        #
        #
        ########################   All brain region
        #
        #
        #df_sess_br={}
        #
        #for i, Sess in enumerate( df_responses ):
        #    df_together = df_responses[i].melt()
        #    df_together['ROI'] = ['visual' for i in range(0, len(df_together))]
        #    df_together['voxel'] = [i+1 for i in range(0, len(df))]*shape(df)[1]
        #    df_together.columns = ['timepoint', 'Decoding', 'ROI', 'voxel']
        #    df_together['timepoint'] = [float(df_together['timepoint'].iloc[i]) for i in range(0, len(df_together))]
        #    df_sess_br[i] = df_together
        #
        #all_sessions = pd.concat(df_sess_br)
        #
        #sns.factorplot(x='timepoint', y='Decoding',  data=all_sessions)
        #plt.title('ROI decoding brain region')
        #plt.show(block=False)
        #
        #
        
        
        #
        #
        #



        
        

