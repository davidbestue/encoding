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
from joblib import Parallel, delayed
import multiprocessing
import time
import random

numcores = multiprocessing.cpu_count() - 10

#paths to save the 3 files 
decoding_thing = 'Response'
Distance_to_use = 'mix'
path_save_reconstructions = '/home/david/Desktop/Reconst_LM_response_nit.xlsx' 
Reconstructions={}
path_save_signal ='/home/david/Desktop/signal_LM_response_boot_nit.xlsx'
path_save_boots = '/home/david/Desktop/boots_LM_response_boot_nit.xlsx'
path_save_shuff = '/home/david/Desktop/shuff_LM_response_boot_nit.xlsx'

Reconstructions_boots=[]
Reconstructions_shuff=[]



Conditions=['1_0.2', '1_7', '2_0.2', '2_7'] #
Subjects=['s001', 'l001'] #, 'r001', 'd001', 'b001', 's001', 'l001'
brain_regions = ['visual', 'ips', 'frontsup', 'frontinf'] #, 'ips', 'frontsup', 'frontmid', 'frontinf'
ref_angle=180


for Subject in Subjects:
    for Brain_region in brain_regions:
        #plt.figure()
        ### Data to use
        enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, 'together', Brain_region)
        ##### Process training data
        training_dataset, training_targets = process_encoding_files(enc_fmri_paths, masks, enc_beh_paths, sys_use='unix', hd=6, TR=2.335) #4
        ##### Train your weigths
        WM, Inter = Weights_matrix_LM( training_dataset, training_targets )
        WM_t = WM.transpose()
        for idx_c, Condition in enumerate(Conditions):
            #plt.subplot(2,2,idx_c+1)
            Reconstruction, boots, shuff = all_process_condition_shuff_boot( Subject=Subject, Brain_Region=Brain_region, WM=WM, WM_t=WM_t, 
            distance=Distance_to_use, decode_item= decoding_thing, iterations=50, Inter=Inter, Condition=Condition, method='together',  heatmap=False) #100

            Reconstructions[Subject + '_' + Brain_region + '_' + Condition]=Reconstruction
            Reconstructions_boots.append(boots)
            Reconstructions_shuff.append(shuff)
            ## Plot the 4 heatmaps
            #plt.title(Condition)
            #ax = sns.heatmap(Reconstruction, yticklabels=list(Reconstruction.index), cmap="coolwarm") # cmap= viridis "jet",  "coolwarm" RdBu_r, gnuplot, YlOrRd, CMRmap  , center = midpoint
            #ax.plot([0.25, np.shape(Reconstruction)[1]-0.25], [posch1_to_posch2(18),posch1_to_posch2(18)], 'k--')
            #plt.yticks([posch1_to_posch2(4), posch1_to_posch2(13), posch1_to_posch2(22), posch1_to_posch2(31)] ,['45','135','225', '315'])
            #plt.ylabel('Angle')
            #plt.xlabel('time (s)')
            
            
        #plt.suptitle( Subject + ' ' + Brain_region , fontsize=12)
        #plt.tight_layout()
        #plt.show(block=False)
        
        
        
        

### Save Recosntructions
writer = pd.ExcelWriter(path_save_reconstructions)
for i in range(len(Reconstructions.keys())):
    Reconstructions[Reconstructions.keys()[i]].to_excel(writer, sheet_name=Reconstructions.keys()[i]) #each dataframe in a excel sheet

writer.save()   #save reconstructions (heatmaps)


### Save signal from the reconstyructions
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


### Save bootstraps
Df_boots = pd.concat(Reconstructions_boots)
Df_boots['label'] = 'boots' ## add the label of shuffle
Df_boots.to_excel(path_save_boots)  #save shuffle


### Save Shuffle
Df_boots = pd.concat(Reconstructions_shuff)
Df_boots['label'] = 'shuffle' ## add the label of shuffle
Df_boots.to_excel(path_save_shuff)  #save shuffle



