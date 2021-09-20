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
decoding_thing = 'T_alone'  #'dist_alone'  'T_alone'  
Distance_to_use = 'mix' #'close' 'far'

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
            print(Subject, Brain_region, Condition )
            #############
            ############# Get the data
            enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, 'together', Brain_region)
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