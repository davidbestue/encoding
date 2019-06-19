# -*- coding: utf-8 -*-
"""
Created on Wed May 22 12:53:25 2019

@author: David Bestue
"""

from model_functions import *
from fake_data_generator import *
from Weights_matrixs import *
from Representation import *
from process_encoding import *
from process_wm import *
from data_to_use import *
from joblib import Parallel, delayed
import multiprocessing
import time
import random


numcores = multiprocessing.cpu_count()

def shuffled_reconstruction(signal_paralel, targets, iterations, WM, WM_t, Inter, region, condition, subject, ref_angle=180):
    ### shuffle the targets
    testing_angles_sh=[]
    for n_rep in range(iterations):
        #new_targets = random.sample(targets, len(targets))
        #testing_angles_sh.append(new_targets)
        testing_angles_sh.append( np.array([random.choice([0, 90, 180, 270]) for i in range(len(targets))])) ## instead of shuffle, take a region where there is no activity!
    
    ### make the reconstryctions and append them
    Reconstructions_sh=[]
    for n_rep in range(iterations):
        time_rec_shuff_start = time.time()
        Reconstructions_i = Parallel(n_jobs = numcores)(delayed(Representation)(signal, testing_angles_sh[n_rep], WM, WM_t, intercept=Inter, ref_angle=180, plot=False)  for signal in signal_paralel) 
        Reconstruction_i = pd.concat(Reconstructions_i, axis=1) 
        Reconstruction_i.columns =  [str(i * TR) for i in range(nscans_wm)]
        Reconstructions_sh.append(Reconstruction_i)
        time_rec_shuff_end = time.time()
        time_rec_shuff = time_rec_shuff_end - time_rec_shuff_start
        print('shuff_' + str(n_rep) + ': ' +str(time_rec_shuff) )
    
    ### Get just the supposed target location
    df_shuffle=[]
    for i in range(len(Reconstructions_sh)):
        n = Reconstructions_sh[i].iloc[360, :]
        n = n.reset_index()
        n.columns = ['times', 'decoding']
        n['decoding'] = [sum(Reconstructions_sh[i].iloc[:, ts] * f2(180)) for ts in range(len(n))]
        n['times']=n['times'].astype(float)
        n['region'] = region
        n['subject'] = subject
        n['condition'] = condition
        df_shuffle.append(n)
    
    ##
    df_shuffle = pd.concat(df_shuffle)
    
    return df_shuffle




def all_process_condition_shuff( Subject, Brain_Region, WM, WM_t, Inter, Condition, iterations, method='together', heatmap=False):
    enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, method, Brain_Region)
    ##### Process testing data
    testing_activity, testing_behaviour = preprocess_wm_files(wm_fmri_paths, masks, wm_beh_paths, condition=Condition, distance='mix', sys_use='unix', nscans_wm=nscans_wm, TR=2.335)
    testing_angles = np.array(testing_behaviour['Dist'])    
    
    ### Respresentation
    start_repres = time.time()    
    # TR separartion
    signal_paralel =[ testing_activity[:, i, :] for i in range(nscans_wm)]
    Reconstructions = Parallel(n_jobs = numcores)(delayed(Representation)(signal, testing_angles, WM, WM_t, ref_angle=180, plot=False, intercept=Inter)  for signal in signal_paralel)    ####
    Reconstruction = pd.concat(Reconstructions, axis=1) 
    Reconstruction.columns =  [str(i * TR) for i in range(nscans_wm)]
    
    #Plot heatmap
    if heatmap==True:
        plt.figure()
        plt.title(Condition)
        ######midpoint = df.values.mean() # (df.values.max() - df.values.min()) / 2
        ax = sns.heatmap(Reconstruction, yticklabels=list(Reconstruction.index), cmap="coolwarm") # cmap= viridis "jet",  "coolwarm" RdBu_r, gnuplot, YlOrRd, CMRmap  , center = midpoint
        ax.plot([0.25, np.shape(Reconstruction)[1]-0.25], [posch1_to_posch2(18),posch1_to_posch2(18)], 'k--')
        plt.yticks([posch1_to_posch2(4), posch1_to_posch2(13), posch1_to_posch2(22), posch1_to_posch2(31)] ,['45','135','225', '315'])
        plt.ylabel('Angle')
        plt.xlabel('time (s)')
        plt.tight_layout()
        plt.show(block=False)
    
    ######
    ######
    ######
    
    end_repres = time.time()
    process_recons = end_repres - start_repres
    print( 'Time process reconstruction: ' +str(process_recons))
    
    ####### Shuff
    #### Compute the shuffleing
    shuffled_rec = shuffled_reconstruction(signal_paralel, testing_angles, iterations, WM, WM_t, Inter=Inter, region=Brain_Region, condition=Condition, subject=Subject, ref_angle=180)
    
    return Reconstruction, shuffled_rec


##########################################################################################################
##########################################################################################################


path_save_reconstructions = '/home/david/Desktop/Reconst_LM_dist.xlsx'
Reconstructions={}
path_save_signal ='/home/david/Desktop/signal_LM_dist.xlsx'
path_save_shuffle = '/home/david/Desktop/shuff_LM_dist.xlsx'
Reconstructions_shuff=[]


Conditions=['1_0.2', '1_7', '2_0.2', '2_7']
Subjects=['n001', 'r001', 'd001', 'b001', 's001', 'l001'] #, 'r001', 'd001', 'b001', 's001', 'l001'
brain_regions = ['visual', 'ips', 'frontsup', 'frontmid', 'frontinf']


for Subject in Subjects:
    for Brain_region in brain_regions:
        plt.figure()
        ### Data to use
        enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, 'together', Brain_region)
        ##### Process training data
        training_dataset, training_targets = process_encoding_files(enc_fmri_paths, masks, enc_beh_paths, sys_use='unix', hd=6, TR=2.335) #4
        ##### Train your weigths
        WM, Inter = Weights_matrix_LM( training_dataset, training_targets )
        WM_t = WM.transpose()
        for idx_c, Condition in enumerate(Conditions):
            plt.subplot(2,2,idx_c+1)
            Reconstruction, shuff = all_process_condition_shuff( Subject=Subject, Brain_Region=Brain_region, WM=WM, WM_t=WM_t, iterations=100, Inter=Inter, Condition=Condition, method='together',  heatmap=False) #100
            Reconstructions[Subject + '_' + Brain_region + '_' + Condition]=Reconstruction
            Reconstructions_shuff.append(shuff)
            ## Plot the 4 heatmaps
            plt.title(Condition)
            ######midpoint = df.values.mean() # (df.values.max() - df.values.min()) / 2
            ax = sns.heatmap(Reconstruction, yticklabels=list(Reconstruction.index), cmap="coolwarm") # cmap= viridis "jet",  "coolwarm" RdBu_r, gnuplot, YlOrRd, CMRmap  , center = midpoint
            ax.plot([0.25, np.shape(Reconstruction)[1]-0.25], [posch1_to_posch2(18),posch1_to_posch2(18)], 'k--')
            plt.yticks([posch1_to_posch2(4), posch1_to_posch2(13), posch1_to_posch2(22), posch1_to_posch2(31)] ,['45','135','225', '315'])
            plt.ylabel('Angle')
            plt.xlabel('time (s)')
            
            
        plt.suptitle( Subject + ' ' + Brain_region , fontsize=12)
        plt.tight_layout()
        plt.show(block=False)
        


### Save Recosntructions
writer = pd.ExcelWriter(path_save_reconstructions)
for i in range(len(Reconstructions.keys())):
    Reconstructions[Reconstructions.keys()[i]].to_excel(writer, sheet_name=Reconstructions.keys()[i])

writer.save()   


### Save signal
Decoding_df =[]

for dataframes in Reconstructions.keys():
    df = Reconstructions[dataframes]
    a = pd.DataFrame(df.iloc[360,:])
    a = a.reset_index()
    a.columns = ['times', 'decoding']
    a['decoding'] = [sum(df.iloc[:,i] * f2(180)) for i in range(len(a))]
    a['times']=a['times'].astype(float)
    a['region'] = dataframes.split('_')[1]
    #a['region'] = dataframes.split('_')[1] + '_' + dataframes.split('_')[2]
    a['subject'] = dataframes.split('_')[0]
    a['condition'] = dataframes.split('_')[-2] + '_' + dataframes.split('_')[-1] 
    Decoding_df.append(a)



Df = pd.concat(Decoding_df)
Df['label'] = 'signal'
Df.to_excel( path_save_signal )


### Save Shuffle
Df_shuffle = pd.concat(Reconstructions_shuff)
Df_shuffle['label'] = 'shuffle'
Df_shuffle.to_excel(path_save_shuffle)





