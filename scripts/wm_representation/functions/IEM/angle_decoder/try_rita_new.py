# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:24:32 2019

@author: David Bestue
"""

############# Add to sys path the path where the tools folder is
import sys, os
path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)) ### same directory or one back options
path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) ### same directory or one back options
sys.path.insert(1, path_tools)
from tools import *

############# Namefiles for the savings. 
#path_save_signal ='/home/david/Desktop/Reconstructions/IEM/IEM_trainD_testD_resp.xlsx' 
#path_save_shuffle = '/home/david/Desktop/Reconstructions/IEM/shuff_IEM_trainD_testD_resp.xlsx'

############# Testing options
decoding_thing = 'T'  #'dist_alone'  'T_alone'  

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
Conditions=['1_0.2']   ##['1_0.2', '1_7', '2_0.2', '2_7'] 
Subjects=['n001'] ##['d001', 'n001', 'b001', 'r001', 's001', 'l001']
brain_regions = ['visual'] ##, 'ips', 'pfc']
ref_angle=180

num_shuffles = 2 #100 #10

############# Analysis
#############
for Subject in Subjects:
    for Brain_region in brain_regions:
        for idx_c, Condition in enumerate(Conditions):
            ####
            print(Subject, Brain_region, Condition )
            ####
            if Condition == cond_t:  ### Cross-validate if training and testing condition are the same (1_7 when training on target and 2_7 when training on distractor)
                #############
                ############# Get the data
                enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, 'together', Brain_region)
                #############
                ###### Process wm files (I call them activity instead of training_ or testing_ as they come from the same condition)
                activity, behaviour = preprocess_wm_data(wm_fmri_paths, masks, wm_beh_paths, 
                    condition=Condition, distance=Distance_to_use, nscans_wm=nscans_wm)
                #############
                ####### IEM cross-validating all the TRs
                #L1out=int(len(behaviour)-1) ##instead of the default 10, do the leave one out!
                Reconstruction = IEM_cv_all_runsout(testing_activity=activity, testing_behaviour=behaviour,
                 decode_item=decoding_thing, training_item=training_item, tr_st=tr_st, tr_end=tr_end)
                #
                Reconstructions[Subject + '_' + Brain_region + '_' + Condition]=Reconstruction
                #############
                # IEM shuffle cross-validating all the TRs
                shuff = IEM_cv_all_runsout_shuff(testing_activity=activity, testing_behaviour=behaviour, 
                    decode_item=decoding_thing, training_item=training_item, tr_st=tr_st, tr_end=tr_end,
                    condition=Condition, subject=Subject, region=Brain_region,
                    iterations=num_shuffles)
                Reconstructions_shuff.append(shuff)
                
            else:
                #############
                ############# Get the data
                enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, 'together', Brain_region)
                ##################
                ###### Process training data
                training_activity, training_behaviour = preprocess_wm_data(wm_fmri_paths, masks, wm_beh_paths, 
                    condition=cond_t, distance=Distance_to_use, nscans_wm=nscans_wm)
                #
                ##################
                ###### Process testing data 
                testing_activity, testing_behaviour = preprocess_wm_data(wm_fmri_paths, masks, wm_beh_paths, 
                    condition=Condition, distance=Distance_to_use, nscans_wm=nscans_wm)
                ##################
                testing_behaviour['new_index'] = np.arange(0, len(testing_behaviour),1) 
                ### testing_activity --> (trials, TRs, voxels)
                ### testing_behaviour --> (trials, columns_interest)
                ###### IEM shuffle
                shuff = Representation_angle_runsout_shuff(training_activity=training_activity, training_behaviour=training_behaviour, 
                    testing_activity=testing_activity, testing_behaviour=testing_behaviour, decode_item=decoding_thing, 
                    training_item=training_item, tr_st=tr_st, tr_end=tr_end, 
                    condition=Condition, subject=Subject, region=Brain_region,
                    iterations=num_shuffles, ref_angle=180)
                ####### IEM data

                Representation_angle_runsout_shuff





training_activity=training_activity
training_behaviour=training_behaviour
testing_activity=testing_activity
testing_behaviour=testing_behaviour
decode_item=decoding_thing
training_item=training_item
tr_st=tr_st
tr_end=tr_end
####
#### Get the Trs (no shared info, coming from different trials)
list_wm_scans= range(nscans_wm)  
list_wm_scans2 = list_wm_scans
####
####
####
#### Run the ones WITHOUT shared information the same way
#testing_behaviour = testing_behaviour.reset_index()
#training_behaviour = training_behaviour.reset_index()
training_angles = np.array(training_behaviour[training_item])   
testing_angles = np.array(testing_behaviour[decode_item])    
testing_distractors = np.array(testing_behaviour['Dist'])   
#####
Recons_trs=[]
for not_shared in list_wm_scans2:
    training_data =   np.mean(training_activity[:, tr_st:tr_end, :], axis=1) ## son los mismos siempre, pero puede haber time dependence!
    testing_data= testing_activity[:, not_shared, :]   
    reconstrction_=[]
    ###########################################################################
    ########################################################################### Get the mutliple indexes to split in train and test
    ###########################################################################
    training_indexes = []
    testing_indexes =  []
    for sess_run in testing_behaviour.session_run.unique():
        wanted = testing_behaviour.loc[testing_behaviour['session_run']==sess_run].index.values 
        testing_indexes.append( wanted )
        #
        ## I do not trust the del  lines of other files, maybe this del inside a function in paralel is not removing the indexes, also you avoid going to lists to comeback
        all_indexes = testing_behaviour.index.values
        other_indexes = all_indexes[~np.array([all_indexes[i] in wanted for i in range(len(all_indexes))])]  #take the ones that are not in wanted
        training_indexes.append( other_indexes ) 
    ###
    ### apply them to train and test
    ###
    for train_index, test_index in zip(training_indexes, testing_indexes):
        X_train, X_test = training_data[train_index], testing_data[test_index]
        y_train, y_test = training_angles[train_index], testing_angles[test_index]
        y_train_dist, y_test_dist = testing_distractors[train_index], testing_distractors[test_index]




WM2, Inter2 = Weights_matrix_LM(X_train, y_train)
WM_t2 = WM2.transpose()
## test
rep_x = Representation(testing_data=X_test, testing_angles=y_test, Weights=WM2, Weights_t=WM_t2, ref_angle=180, plot=False, intercept=Inter2)

JJJ = Representation_not_mean(testing_data=X_test, testing_angles=y_test, testing_distractors=y_test_dist, Weights=WM2, Weights_t=WM_t2, ref_angle=180, intercept=Inter2)
reconstrction_.append(rep_x)






                Reconstruction = IEM_all_runsout( training_activity=training_activity, training_behaviour=training_behaviour, 
                    testing_activity=testing_activity, testing_behaviour=testing_behaviour, 
                    decode_item=decoding_thing, training_item=training_item, tr_st=tr_st, tr_end=tr_end)
                    
                #
                Reconstructions[Subject + '_' + Brain_region + '_' + Condition]=Reconstruction
                ##################
                ###### IEM shuffle
                shuff = IEM_all_runsout_shuff(training_activity=training_activity, training_behaviour=training_behaviour, 
                    testing_activity=testing_activity, testing_behaviour=testing_behaviour, decode_item=decoding_thing, 
                    training_item=training_item, tr_st=tr_st, tr_end=tr_end, 
                    condition=Condition, subject=Subject, region=Brain_region,
                    iterations=num_shuffles, ref_angle=180)
                #
                Reconstructions_shuff.append(shuff)
                
            







            print(Subject, Brain_region, Condition )
            #############
            ############# Get the data
            enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, 'together', Brain_region)
            #############
            #############
            ###### Process wm files (I call them activity instead of training_ or testing_ as they come from the same condition)
            activity, behaviour = preprocess_wm_data(wm_fmri_paths, masks, wm_beh_paths, 
                condition=Condition, distance=Distance_to_use, nscans_wm=nscans_wm)
            #############
            #############
            ### activity --> (trials, TRs, voxels)
            ### behaviour --> (trials, columns_interest)
            df_decoded = decode_angle(fmri_activity=activity, beh_activity=behaviour)





            ###### Process testing data 
            testing_activity, testing_behaviour = preprocess_wm_data(wm_fmri_paths, masks, wm_beh_paths, 
                condition=Condition, distance=Distance_to_use, nscans_wm=nscans_wm)
            ##################
            ################## Split by error
            ##################
            ##### Get the responded ones
            if decoding_thing == 'T_alone':
                testing_activity = testing_activity[testing_behaviour[decoding_thing] == testing_behaviour['T'] ] 
                testing_behaviour = testing_behaviour[testing_behaviour[decoding_thing] == testing_behaviour['T'] ]                     
            if decoding_thing == 'dist_alone':
                testing_activity = testing_activity[testing_behaviour[decoding_thing] == testing_behaviour['Dist'] ] 
                testing_behaviour = testing_behaviour[testing_behaviour[decoding_thing] == testing_behaviour['Dist'] ]                     
            #####

            ####### Decode the angle (df)
            #
            #
            Reconstructions[Subject + '_' + Brain_region + '_' + Condition]=Reconstruction
            #############
            ####### Decode the angle (df) in shuffle
            

            Reconstructions_shuff.append(shuff)
                

#
#

        
###### Save reconstruction (heatmap)         
### Get signal from the reconstructions (get the signal before; not done in the function in case you want to save the whole)
### If you want to save the whole recosntruction, uncomment the following lines
writer = pd.ExcelWriter(path_save_reconstructions)
for i in range(len(Reconstructions.keys())):
    Reconstructions[Reconstructions.keys()[i]].to_excel(writer, sheet_name=Reconstructions.keys()[i]) #each dataframe in a excel sheet

writer.save()   #save reconstructions (heatmaps)

###### Save decoding signal (around the reference angle)
Decoding_df =[]

for dataframes in Reconstructions.keys():
    df = Reconstructions[dataframes]
    a = pd.DataFrame(df.iloc[ref_angle*2,:]) ##*2 because there are 720
    a = a.reset_index()
    a.columns = ['times', 'decoding'] # column names
    a['decoding'] = [sum(df.iloc[:,i] * f2(ref_angle)) for i in range(len(a))] #"population vector method" scalar product
    a['times']=a['times'].astype(float)
    a['region'] = dataframes.split('_')[1]
    a['subject'] = dataframes.split('_')[0]
    a['condition'] = dataframes.split('_')[-2] + '_' + dataframes.split('_')[-1] 
    Decoding_df.append(a)


Df = pd.concat(Decoding_df)
Df['label'] = 'signal' #add the label of signal (you will concatenate this df with the one of the shuffleing)
Df.to_excel( path_save_signal ) #save signal

####################################################
## 
###### Save Shuffle 
### I do not need to do the "pop vector" step becuase it is done inside the function IEM_shuff
### I do it different because eventually I might be interested in saving the whole reconstruction of the signal (I am not interested in the shuffles)
Df_shuffs = pd.concat(Reconstructions_shuff)
Df_shuffs['label'] = 'shuffle' ## add the label of shuffle
Df_shuffs.to_excel(path_save_shuffle)  #save shuffle


##################