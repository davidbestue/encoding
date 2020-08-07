# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:24:32 2019

@author: David Bestue
"""
from model_functions import *
from fake_data_generator import *
from Weights_matrixs import *
from Representation import *
from process_encoding import *
from process_wm import *
from data_to_use import *
from bootstrap_functions import *
from leave_one_out import *
from joblib import Parallel, delayed
import multiprocessing
import time
import random
#

numcores = multiprocessing.cpu_count() #- 10

##paths to save the 3 files 
decoding_thing = 'Target' #'Distractor' #'Target'
Distance_to_use = 'close'
#path_save_reconstructions = '/home/david/Desktop/all_target_mix.xlsx' 
Reconstructions={}
#path_save_signal ='/home/david/Desktop/target_close/signal_all_target_mix.xlsx'
#path_save_shuff = '/home/david/Desktop/target_close/shuff_all_target_mix.xlsx'

Reconstructions_shuff=[]

Conditions=['1_0.2']#, '1_7', '2_0.2', '2_7'] #
Subjects=['d001'] #, 'n001', 'b001', 'r001', 's001', 'l001'] #, 'r001', 'd001', 'b001', 's001', 'l001'
brain_regions = ['visual'] #, 'ips', 'pfc']# 'frontinf'] #, 'ips', 'frontsup', 'frontmid', 'frontinf'
#ref_angle=180


for Subject in Subjects:
    for Brain_region in brain_regions:
        for idx_c, Condition in enumerate(Conditions):
            print(Condition)
            Reconstruction, shuff = leave_one_out_shuff( Subject=Subject, Brain_Region=Brain_region, Condition=Condition, iterations=10, 
                distance=Distance_to_use, decode_item=decoding_thing, method='together', heatmap=False)
            Reconstructions[Subject + '_' + Brain_region + '_' + Condition]=Reconstruction
            #Reconstructions_boots.append(boots)
            Reconstructions_shuff.append(shuff)


###

### Save signal from the reconstructions
Decoding_df =[]

for dataframes in Reconstructions.keys():
    a=pd.DataFrame(Reconstruction.iloc[0,:].values) #before it was transpose to mimic the shuffle ones 
    a['time']=[i * TR for i in range(nscans_wm)]
    a.columns=['decoding', 'time']  
    a['region'] = dataframes.split('_')[1]
    a['subject'] = dataframes.split('_')[0]
    a['condition'] = dataframes.split('_')[-2] + '_' + dataframes.split('_')[-1] 
    Decoding_df.append(a)



Df = pd.concat(Decoding_df)
Df['label'] = 'signal' #ad the label of signal (you will concatenate this df with the one of the shuffleing)
Df.to_excel( path_save_signal ) #save signal
# Subject='d001'
# method='together' 
# Brain_Region='visual'
# enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, 'together', Brain_region)
# Condition='1_0.2'
# Subject='d001'
# distance='close'
# testing_activity, testing_behaviour = preprocess_wm_files(wm_fmri_paths, masks, wm_beh_paths, condition=Condition, distance=distance, sys_use='unix', nscans_wm=nscans_wm, TR=2.335)
# decode_item = 'Target'
# dec_I = 'T'
# testing_angles_beh = np.array(testing_behaviour[dec_I])    # A_R # T # Dist
# angles_paralel= [testing_angles_beh for i in range(nscans_wm)]
# signal_paralel =[ testing_activity[:, i, :] for i in range(nscans_wm)]


# testing_data=signal_paralel[0]
# testing_angles=angles_paralel[0]

# error_TR = Parallel(n_jobs = numcores)(delayed(Pop_vect_leave_one_out)(testing_data = signal, testing_angles= angles)  for signal, angles in zip(signal_paralel, angles_paralel))    #### reconstruction standard (paralel)


# iterations=3
# itera_paralel=[iterations for i in range(nscans_wm)]
# shuffled_rec = Parallel(n_jobs = numcores)(delayed(shuff_Pop_vect_leave_one_out)(testing_data=signal_s, testing_angles=angles_s, iterations=itera) for signal_s, angles_s, itera in zip(signal_paralel, angles_paralel, itera_paralel))


# Reconstruction_sh = pd.DataFrame(shuffled_rec) #
# Reconstruction_sh = Reconstruction_sh.transpose()
# Reconstruction_sh.columns =  [str(i * TR) for i in range(nscans_wm)]  #mean error en each TR (n_iterations filas con n_scans columnas)
