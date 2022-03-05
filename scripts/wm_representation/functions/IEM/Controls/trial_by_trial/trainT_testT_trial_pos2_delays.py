# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:24:32 2019

@author: David Bestue
"""

#######
####### In this analysis:
####### I am doing the reconstruction training in the delay period and testing in each trial. No CV and No Shuffles
####### I am converting to positive the reconstruction and I am extracting the angle doing a Pop vector -45, 45 around the centered position
####### I am averaging the delay period 1 and 2 before reconstructing (not in each TR)
#######


############# Add to sys path the path where the tools folder is
import sys, os
#path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) ### same directory or one back options
path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)) ### same directory or one back options
sys.path.insert(1, path_tools)
from tools import *

############# Namefiles for the savings. 
path_save_behaviour ='/home/david/Desktop/Reconstructions/IEM/IEM_trainT_testT_trials_pos2_delays.xlsx' 

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
Conditions=['1_0.2', '1_7', '2_0.2', '2_7'] 
Subjects=['d001', 'n001', 'b001', 'r001', 's001', 'l001']
brain_regions = ['visual','ips', 'pfc', 'broca']
ref_angle=180


Behaviour_ = []

############# Analysis
#############
for Subject in Subjects:
    for Brain_region in brain_regions:
        print(Subject, Brain_region)
        enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, 'together', Brain_region)
        activity, behaviour = process_wm_task(wm_fmri_paths, masks, wm_beh_paths, nscans_wm=nscans_wm) 
        behaviour['Condition'] = behaviour['Condition'].replace(['1.0_0.2', '1.0_7.0', '2.0_0.2','2.0_7.0' ], ['1_0.2', '1_7', '2_0.2', '2_7'])
        signal_decoded_trial=[]
        angle_decoded_trial=[]
        for trial in range(len(behaviour)):
            activity_trial = activity[trial,:,:]
            beh_trial = behaviour.iloc[trial,:]
            session_trial = beh_trial.session_run 
            Condition_ = beh_trial.Condition
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
            signal_decoded_trs = [] #3 values inside (1 per delay)
            angle_decoded_trs = [] #3 values inside (1 per delay)
            ###
            ###
            ###
            for Delay_ in ['Delay_1', 'Delay_2', 'Delay_end']:
                if Condition_ == '1_0.2':
                    TR_st_D1 = 4
                    TR_end_D1 = 8
                    TR_st_D2 =4
                    TR_end_D2 =8 
                    TR_st_Dend =6
                    TR_end_Dend =8 
                elif Condition_ == '1_7':
                    TR_st_D1 = 4
                    TR_end_D1 = 6
                    TR_st_D2 = 6
                    TR_end_D2 = 8
                    TR_st_Dend =6
                    TR_end_Dend =8 
                elif Condition_ == '2_0.2':
                    TR_st_D1 = 4
                    TR_end_D1 = 8
                    TR_st_D2 = 4
                    TR_end_D2 = 8
                    TR_st_Dend =6
                    TR_end_Dend =8 
                elif Condition_ == '2_7':
                    TR_st_D1 = 4
                    TR_end_D1 = 6
                    TR_st_D2 = 7
                    TR_end_D2 = 12
                    TR_st_Dend = 10
                    TR_end_Dend = 12
                ###
                ###
                ###
                if Delay_ == 'Delay_1':
                    activity_TR = activity_trial[TR_st_D1:TR_end_D1, :].mean(axis=0)
                if Delay_ == 'Delay_2':
                    activity_TR = activity_trial[TR_st_D2:TR_end_D2, :].mean(axis=0)
                if Delay_ == 'Delay_end':
                    activity_TR = activity_trial[TR_st_Dend:TR_end_Dend, :].mean(axis=0)
                ###
                ###
                ###
                angle_trial = beh_trial[decoding_thing]
                Inverted_encoding_model = np.dot( np.dot ( np.linalg.pinv( np.dot(Weights_matrix_t, Weights_matrix ) ),  Weights_matrix_t),  activity_TR) 
                Inverted_encoding_model_pos = Pos_IEM2(Inverted_encoding_model)
                IEM_hd = ch2vrep3(Inverted_encoding_model_pos) #36 to 720
                to_roll = int( (ref_angle - angle_trial)*(len(IEM_hd)/360) ) ## degrees to roll
                IEM_hd_aligned=np.roll(IEM_hd, to_roll) ## roll this degree
                ###
                ### Signal of decoding
                ###
                signal_decoding = np.mean(IEM_hd_aligned[360-10: 360+10]) ## mean 5 degrees up and down
                signal_decoded_trs.append(signal_decoding)
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
                angle_decoded_trs.append(angle_decoded)
                #
            ### Add 32 columns to the behaviour matrix
            signal_decoded_trial.append(signal_decoded_trs)
            angle_decoded_trial.append(angle_decoded_trs)
        ####
        df_signal_rec = pd.DataFrame(signal_decoded_trial)
        df_signal_rec.columns=['signal_d1', 'signal_d2', 'signal_dend']
        df_angle_rec = pd.DataFrame(angle_decoded_trial)
        df_angle_rec.columns=['angle_d1', 'angle_d2', 'angle_dend']
        ##
        beh2 = behaviour.reset_index() 
        Behaviour = pd.concat([beh2, df_signal_rec, df_angle_rec], axis=1)
        Behaviour['brain_region'] = Brain_region
        Behaviour_.append(Behaviour)


#########


Behaviour_final = pd.concat(Behaviour_)

Behaviour_final.to_excel( path_save_behaviour ) #save signal

