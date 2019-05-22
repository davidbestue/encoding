# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:36:53 2019

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


numcores = multiprocessing.cpu_count()



def all_process_condition( Subject, Brain_Region, WM, WM_t, Inter, Condition, method='together', heatmap=True):
    enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, method, Brain_Region)
    ##### Process testing data
    testing_activity, testing_behaviour = preprocess_wm_files(wm_fmri_paths, masks, wm_beh_paths, condition=Condition, distance='mix', sys_use='unix', nscans_wm=nscans_wm, TR=2.335)
    testing_angles = np.array(testing_behaviour['T'])    
    
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
    
    return Reconstruction




############ More optimal

path_save = '/home/david/Desktop/KAROLINSKA/Reconstructions_Lasso_i.xlsx'
    
Reconstructions={}
Reconstructions_shuff=[]


Conditions=['1_0.2', '1_7', '2_0.2', '2_7']
Subjects=['n001', 'r001', 'd001', 'b001', 's001', 'l001'] #, 'r001', 'd001', 'b001', 's001', 'l001'
brain_regions = ['visual', 'ips']


for Subject in Subjects:
    for Brain_region in brain_regions:
        plt.figure()
        ### Data to use
        enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, 'together', Brain_region)
        ##### Process training data
        training_dataset, training_targets = process_encoding_files(enc_fmri_paths, masks, enc_beh_paths, sys_use='unix', hd=4, TR=2.335)
        ##### Train your weigths
        WM, Inter = Weights_matrix_Lasso_i( training_dataset, training_targets )
        WM_t = WM.transpose()
        for idx_c, Condition in enumerate(Conditions):
            plt.subplot(2,2,idx_c+1)
            Reconstruction = all_process_condition( Subject=Subject, Brain_Region=Brain_region, WM=WM, WM_t=WM_t, Inter=Inter, Condition=Condition, method='together',  heatmap=False)
            Reconstructions[Subject + '_' + Brain_region + '_' + Condition]=Reconstruction
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
        


writer = pd.ExcelWriter(path_save)

for i in range(len(Reconstructions.keys())):
    Reconstructions[Reconstructions.keys()[i]].to_excel(writer, sheet_name=Reconstructions.keys()[i])

        
writer.save()   



#xls = pd.ExcelFile('Reconstructions_n001.xlsx')
#sheets = xls.sheet_names
###
#R={}
#for sh in sheets:
#    R[sh]  = pd.read_excel(xls, sheet_name=sh)




def all_process_condition_shuff( Subject, Brain_Region, WM, WM_t, Inter, Condition, iterations, method='together', heatmap=True):
    enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, method, Brain_Region)
    ##### Process testing data
    testing_activity, testing_behaviour = preprocess_wm_files(wm_fmri_paths, masks, wm_beh_paths, condition=Condition, distance='mix', sys_use='unix', nscans_wm=nscans_wm, TR=2.335)
    testing_angles = np.array(testing_behaviour['T'])    
    
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
    shuffled_rec = shuffled_reconstruction(signal_paralel, testing_angles, iterations, WM, WM_t, Inter=Inter, region=Brain_Region, condition=Condition, subject=Subject)
    
    
    
    
    return Reconstruction, shuffled_rec




def shuffled_reconstruction(signal_paralel, targets, iterations, WM, WM_t, Inter, region, condition, subject, ref_angle=180, intercept=False, nscans_wm=16 ):
    ### shuffle the targets
    testing_angles_sh=[]
    for n_rep in range(iterations):
        new_targets = sample(testing_angles, len(testing_angles))
        testing_angles_sh.append(new_targets)
    
    
    ### make the reconstryctions and append them
    Reconstructions_sh=[]
    for n_rep in range(iterations):
        Reconstructions_i = Parallel(n_jobs = numcores)(delayed(Representation)(signal, testing_angles_sh[n_rep], WM, WM_t, intercept=Inter, ref_angle=180, plot=False)  for signal in signal_paralel) 
        Reconstruction_i = pd.concat(Reconstructions_i, axis=1) 
        Reconstruction_i.columns =  [str(i * TR) for i in range(nscans_wm)]
        Reconstructions_sh.append(Reconstruction_i)
    
    ### Get just the supposed target location
    df_shuffle=[]
    for i in range(len(Reconstructions_sh)):
        n = shuffled_rec[i].iloc[360, :]
        n = n.reset_index()
        n.columns = ['times', 'decoding']
        n['times']=n['times'].astype(float)
        n['region'] = region
        n['subject'] = subject
        n['condition'] = condition
        df_shuffle.append(n)
    
    ##
    df_shuffle = pd.concat(df_shuffle)
    return df_shuffle




path_save = '/home/david/Desktop/KAROLINSKA/Reconstructions_LM_shuff.xlsx'
    
Reconstructions={}
Reconstructions_shuff=[]


Conditions=['1_0.2']
Subjects=['n001'] #, 'r001', 'd001', 'b001', 's001', 'l001'
brain_regions = ['visual', 'ips']


for Subject in Subjects:
    for Brain_region in brain_regions:
        plt.figure()
        ### Data to use
        enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, 'together', Brain_region)
        ##### Process training data
        training_dataset, training_targets = process_encoding_files(enc_fmri_paths, masks, enc_beh_paths, sys_use='unix', hd=4, TR=2.335)
        ##### Train your weigths
        WM, Inter = Weights_matrix_LM( training_dataset, training_targets )
        WM_t = WM.transpose()
        for idx_c, Condition in enumerate(Conditions):
            plt.subplot(2,2,idx_c+1)
            Reconstruction, shuff = all_process_condition_shuff( Subject=Subject, Brain_Region=Brain_region, WM=WM, WM_t=WM_t, iterations=2, Inter=Inter, Condition=Condition, method='together',  heatmap=False, nscans_wm=16)
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
        




