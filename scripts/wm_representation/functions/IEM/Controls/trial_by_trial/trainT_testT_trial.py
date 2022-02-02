# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:24:32 2019

@author: David Bestue
"""

############# Add to sys path the path where the tools folder is
import sys, os
#path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) ### same directory or one back options
path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)) ### same directory or one back options
sys.path.insert(1, path_tools)
from tools import *

############# Namefiles for the savings. 
path_save_signal ='/home/david/Desktop/Reconstructions/IEM/IEM_trainT_testT_trial.xlsx' 
path_save_reconstructions = '/home/david/Desktop/Reconstructions/IEM/IEM_heatmap_trainT_testT_trial.xlsx'

path_save_shuffle = '/home/david/Desktop/Reconstructions/IEM/shuff_IEM_trainT_testT_trial.xlsx'


############# Testing options
decoding_thing = 'T_alone'  #'dist_alone'  'T_alone'  

############# Training options
training_item = 'T_alone'  #'dist_alone'  'T_alone' 
cond_t = '1_7'             #'1_7'  '2_7'

Distance_to_use = 'mix' #'close' 'far'
training_time= 'delay'  #'stim_p'  'delay' 'respo'
tr_st=4
tr_end=6

############# Options de training times, the TRs used for the training will be different 

# training_time=='delay':
# tr_st=4
# tr_end=6

# training_time=='stim_p':
# tr_st=3
# tr_end=4

# training_time=='delay':
# tr_st=4
# tr_end=6

# training_time=='respo':
#     if decoding_thing=='Target':
#         tr_st=8
#         tr_end=9
#     elif decoding_thing=='Distractor':
#         tr_st=11
#         tr_end=12



############# Dictionary and List to save the files.
Reconstructions={}
Reconstructions_shuff=[]

############# Elements for the loop
Conditions=['1_0.2'] #, '1_7', '2_0.2', '2_7'] 
Subjects=['d001']
brain_regions = ['pfc']
ref_angle=180

num_shuffles = 10 #10 #100 #10



############# Analysis
#############
for Subject in Subjects:
    for Brain_region in brain_regions:
        print(Subject, Brain_region)
        enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, 'together', Brain_region)
        activity, behaviour = process_wm_task(wm_fmri_paths, masks, wm_beh_paths, nscans_wm=nscans_wm) 
        behaviour['Condition'] = behaviour['Condition'].replace(['1.0_0.2', '1.0_7.0', '2.0_0.2','2.0_7.0' ], ['1_0.2', '1_7', '2_0.2', '2_7'])
        for trial in range(len(behaviour)):
            activity_trial = activity[trial,:,:]
            beh_trial = behaviour.iloc[trial,:]
            session_trial = beh_trial.session_run 
            ###
            ### Training
            ###
            if cond_t == '1_7':
                boolean_trials_training = np.array(behaviour['delay1']==7)  *  np.array(behaviour['order']==1) *  np.array(behaviour['session_run']!=session_trial)
            elif cond_t == '2_7':
                boolean_trials_training = np.array(behaviour['delay1']==7)  *  np.array(behaviour['order']==2) *  np.array(behaviour['session_run']!=session_trial)
            #
            activity_train_model = activity[boolean_trials_training, :, :]
            activity_train_model_TRs = np.mean(activity_train_model[:, tr_st:tr_end, :], axis=1)
            behavior_train_model = behaviour[boolean_trials_training]
            training_angles = np.array(behavior_train_model[training_item])
            #
            Weights_matrix, Interc = Weights_matrix_LM(activity_train_model_TRs, training_angles)
            Weights_matrix_t = Weights_matrix.transpose()
            ###
            ### Testing
            ###
            signal_decoded_trial = [] #16 values inside (1 per TR)
            angle_decoded_trial = [] #16 values inside (1 per TR)
            for TR in range(nscans_wm):
                activity_TR = activity_trial[TR, :]
                angle_trial = beh_trial[decoding_thing]
                Inverted_encoding_model = np.dot( np.dot ( np.linalg.pinv( np.dot(Weights_matrix_t, Weights_matrix ) ),  Weights_matrix_t),  activity_TR) 
                IEM_hd = ch2vrep3(Inverted_encoding_model) #36 to 720
                to_roll = int( (ref_angle - angle_trial)*(len(IEM_hd)/360) ) ## degrees to roll
                IEM_hd_aligned=np.roll(IEM_hd, to_roll) ## roll this degree
                ###
                ### Signal of decoding
                ###
                signal_decoding = np.mean(IEM_hd_aligned[360-10: 360+10]) ## mean 5 degrees up and down
                signal_decoded_trial.append(signal_decoding)
                ###
                ### Angle decoding
                ###
                _135_ = ref_angle*2 - 45*2 
                _225_ = ref_angle*2 + 45*2 
                IEM_hd_aligned_135_225 = IEM_hd_aligned[_135_:_225_]
                N=len(IEM_hd_aligned_135_225)
                R = []
                angles = np.radians(np.linspace(135,224,180) ) 
                R=np.dot(IEM_hd_aligned_135_225,np.exp(1j*angles)) / N
                angle = np.angle(R)
                if angle < 0:
                    angle +=2*np.pi
                ##
                angle_degrees = np.degrees(angle)
                if np.all(IEM_hd_aligned_135_225==0)==True: ### if all the values are negative here, the decoded is np.nan (will not count as 0)
                    angle_degrees=np.nan 
                #
                angle_decoded = angle_degrees
                angle_decoded_trial.append(angle_decoded_trial)
                #
            #
            



#             #

### this function will report a list of 16 values (each TR) with the reconstructed angle

#             if Condition == cond_t:  ### Cross-validate if training and testing condition are the same (1_7 when training on target and 2_7 when training on distractor)
#                 #############
#                 ############# Get the data
#                 enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, 'together', Brain_region)
#                 #############
#                 ###### Process wm files (I call them activity instead of training_ or testing_ as they come from the same condition)
#                 activity, behaviour = preprocess_wm_data(wm_fmri_paths, masks, wm_beh_paths, 
#                     condition=Condition, distance=Distance_to_use, nscans_wm=nscans_wm)
#                 #############
#                 ####### IEM cross-validating all the TRs
#                 #L1out=int(len(behaviour)-1) ##instead of the default 10, do the leave one out!
#                 Reconstruction = IEM_cv_all_runsout(testing_activity=activity, testing_behaviour=behaviour,
#                  decode_item=decoding_thing, training_item=training_item, tr_st=tr_st, tr_end=tr_end)
#                 #
#                 Reconstructions[Subject + '_' + Brain_region + '_' + Condition]=Reconstruction
#                 #############
#                 # IEM shuffle cross-validating all the TRs
#                 shuff = IEM_cv_all_runsout_shuff(testing_activity=activity, testing_behaviour=behaviour, 
#                     decode_item=decoding_thing, training_item=training_item, tr_st=tr_st, tr_end=tr_end,
#                     condition=Condition, subject=Subject, region=Brain_region,
#                     iterations=num_shuffles)
#                 Reconstructions_shuff.append(shuff)
                
#             else:
#                 #############
#                 ############# Get the data
#                 enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, 'together', Brain_region)
#                 ##################
#                 ###### Process training data
#                 training_activity, training_behaviour = preprocess_wm_data(wm_fmri_paths, masks, wm_beh_paths, 
#                     condition=cond_t, distance=Distance_to_use, nscans_wm=nscans_wm)
#                 #
#                 ##################
#                 ###### Process testing data 
#                 testing_activity, testing_behaviour = preprocess_wm_data(wm_fmri_paths, masks, wm_beh_paths, 
#                     condition=Condition, distance=Distance_to_use, nscans_wm=nscans_wm)
#                 ##################
#                 ###### IEM 
#                 Reconstruction = IEM_all_runsout( training_activity=training_activity, training_behaviour=training_behaviour, 
#                     testing_activity=testing_activity, testing_behaviour=testing_behaviour, 
#                     decode_item=decoding_thing, training_item=training_item, tr_st=tr_st, tr_end=tr_end)
                    
#                 #
#                 Reconstructions[Subject + '_' + Brain_region + '_' + Condition]=Reconstruction
#                 ##################
#                 ###### IEM shuffle
#                 shuff = IEM_all_runsout_shuff(training_activity=training_activity, training_behaviour=training_behaviour, 
#                     testing_activity=testing_activity, testing_behaviour=testing_behaviour, decode_item=decoding_thing, 
#                     training_item=training_item, tr_st=tr_st, tr_end=tr_end, 
#                     condition=Condition, subject=Subject, region=Brain_region,
#                     iterations=num_shuffles, ref_angle=180)
#                 #
#                 Reconstructions_shuff.append(shuff)
                


        
# ###### Save reconstruction (heatmap)         
# ### Get signal from the reconstructions (get the signal before; not done in the function in case you want to save the whole)
# ### If you want to save the whole recosntruction, uncomment the following lines
# writer = pd.ExcelWriter(path_save_reconstructions)
# for i in range(len(Reconstructions.keys())):
#     Reconstructions[Reconstructions.keys()[i]].to_excel(writer, sheet_name=Reconstructions.keys()[i]) #each dataframe in a excel sheet

# writer.save()   #save reconstructions (heatmaps)

# ###### Save decoding signal (around the reference angle)
# Decoding_df =[]

# for dataframes in Reconstructions.keys():
#     df = Reconstructions[dataframes]
#     a = pd.DataFrame(df.iloc[ref_angle*2,:]) ##*2 because there are 720
#     a = a.reset_index()
#     a.columns = ['times', 'decoding'] # column names
#     a['decoding'] = [sum(df.iloc[:,i] * f2(ref_angle)) for i in range(len(a))] #"population vector method" scalar product
#     a['times']=a['times'].astype(float)
#     a['region'] = dataframes.split('_')[1]
#     a['subject'] = dataframes.split('_')[0]
#     a['condition'] = dataframes.split('_')[-2] + '_' + dataframes.split('_')[-1] 
#     Decoding_df.append(a)


# Df = pd.concat(Decoding_df)
# Df['label'] = 'signal' #add the label of signal (you will concatenate this df with the one of the shuffleing)
# Df.to_excel( path_save_signal ) #save signal

# ####################################################
# ## 
# ###### Save Shuffle 
# ### I do not need to do the "pop vector" step becuase it is done inside the function IEM_shuff
# ### I do it different because eventually I might be interested in saving the whole reconstruction of the signal (I am not interested in the shuffles)
# Df_shuffs = pd.concat(Reconstructions_shuff)
# Df_shuffs['label'] = 'shuffle' ## add the label of shuffle
# Df_shuffs.to_excel(path_save_shuffle)  #save shuffle


# ##################