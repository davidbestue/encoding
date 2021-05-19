# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:24:32 2019

@author: David Bestue
"""

# Add to sys path the path where the tools folder ir (in this case, I need to go one back)
import sys, os
path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 
sys.path.insert(1, path_tools)
from tools import *

##paths to save the files 
path_save_signal ='/home/david/Desktop/Reconstructions/IEM/IEM_example_cross.xlsx' #IEM_target_trtarg_isol_1_7_10.xlsx
path_save_shuffle = '/home/david/Desktop/Reconstructions/IEM/shuff_IEM_example_cross.xlsx'


## options (chek the filename too!)
training_item = 'T_alone'  ## 'dist_alone'  'T_alone' 
decoding_thing = 'Distractor' #'Distractor' #'Target'

Distance_to_use = 'mix' #'close' 'far'
training_time= 'delay' #'stim_p'  'delay' 'respo'


cond_t = '1_7'

# depending on the options, the TRs used for the training will be different
if training_time=='delay':
    tr_st=4
    tr_end=6


## dictionary and list to save the files
Reconstructions={}
Reconstructions_shuff=[]

## elements for the loop
Conditions=[ '1_7']
Subjects=['d001', 'n001', 'b001', 'r001', 's001', 'l001']
brain_regions = ['visual', 'ips', 'pfc']
ref_angle=180

num_shuffles = 5 #100 #10

for Subject in Subjects:
    for Brain_region in brain_regions:
        for idx_c, Condition in enumerate(Conditions):
            print(Subject, Brain_region, Condition )
            #        
            if (Condition == cond_t) & (decoding_thing=='Distractor'):  ###cross validation
                #get the data
                enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, 'together', Brain_region)
                #
                # preprocess wm files 
                testing_activity, testing_behaviour = preprocess_wm_files_alone(wm_fmri_paths, masks, wm_beh_paths, 
                    condition=Condition, distance=Distance_to_use, sys_use='unix', nscans_wm=nscans_wm, TR=2.335)
                #
                # IEM 
                Reconstruction = IEM_cv_all(testing_activity=testing_activity, testing_behaviour=testing_behaviour,
                 decode_item=decoding_thing, training_item=training_item, tr_st=tr_st, tr_end=tr_end, n_slpits=10)
                Reconstructions[Subject + '_' + Brain_region + '_' + Condition]=Reconstruction
                #
                # IEM shuffle
                #shuff = IEM_cross_condition_kfold_shuff_allTRs_alone(testing_activity=testing_activity, testing_behaviour=testing_behaviour, 
                #    decode_item=decoding_thing, WM=WM, WM_t=WM_t, Inter=Inter, condition=Condition, subject=Subject, region=Brain_region,
                #    iterations=num_shuffles, tr_st=tr_st, tr_end=tr_end, ref_angle=180, n_slpits=10)
                #Reconstructions_shuff.append(shuff)
                ###Reconstructions_shuff.append(shuff)
            else:
                print('Error')
                


        
### Save signal         
### Get signal from the reconstructions (get the signal before; not done in the function in case you want to save the whole)
### If you want to save the whole recosntruction, uncomment the following lines

path_save_reconstructions ='/home/david/Desktop/Reconstructions/IEM/IEM_example_cross_recons.xlsx' #IEM_target_trtarg_isol_1_7_10.xlsx

### Save Recosntructions
# path_save_reconstructions = #
writer = pd.ExcelWriter(path_save_reconstructions)
for i in range(len(Reconstructions.keys())):
    Reconstructions[Reconstructions.keys()[i]].to_excel(writer, sheet_name=Reconstructions.keys()[i]) #each dataframe in a excel sheet

writer.save()   #save reconstructions (heatmaps)

#Save just the signal (around the decoding thing)
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
Df['label'] = 'signal' #ad the label of signal (you will concatenate this df with the one of the shuffleing)
Df.to_excel( path_save_signal ) #save signal

### Save Shuffle (in shuffles you do not need to get the *2 thing becuase it is done inside the function)
# Df_shuffs = pd.concat(Reconstructions_shuff)
# Df_shuffs['label'] = 'shuffle' ## add the label of shuffle
# Df_shuffs.to_excel(path_save_shuffle)  #save shuffle




# previous_2_path =  os.path.abspath(os.path.join(previous_path, os.pardir)) 
# sys.path.insert(1, previous_2_path)

# from model_functions import *
# from fake_data_generator import *
# from Weights_matrixs import *
# from Representation import *
# from process_encoding import *
# from process_wm import *
# from data_to_use import *
# from bootstrap_functions import *
# from isolation_reconstruction import *
# from joblib import Parallel, delayed
# import multiprocessing
# import time
# import random
# from sklearn.model_selection import KFold


# import multiprocessing
# multiprocessing.cpu_count() 

# ### use the cores so we do not run out of memory
# numcores = multiprocessing.cpu_count() 
# if numcores>20:
#     numcores=numcores-10
# if numcores<10:
#     numcores=numcores-3
