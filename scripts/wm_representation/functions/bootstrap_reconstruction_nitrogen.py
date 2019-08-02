# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:24:32 2019

@author: David
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


numcores = multiprocessing.cpu_count() - 10

def shuffled_reconstruction(signal_paralel, targets, iterations, WM, WM_t, Inter, region, condition, subject, ref_angle=180):
    ### shuffle the targets
    testing_angles_sh=[] #new targets shuffled
    for n_rep in range(iterations):
        #new_targets = random.sample(targets, len(targets)) #shuffle the labels of the target
        #testing_angles_sh.append(new_targets)
        testing_angles_sh.append( np.array([random.choice([0, 90, 180, 270]) for i in range(len(targets))])) ## instead of shuffle, take a region where there is no activity!
    
    ### make the reconstryctions and append them
    Reconstructions_sh=[]
    for n_rep in range(iterations):
        time_rec_shuff_start = time.time() #time it takes
        Reconstructions_i = Parallel(n_jobs = numcores)(delayed(Representation)(signal, testing_angles_sh[n_rep], WM, WM_t, intercept=Inter, ref_angle=180, plot=False)  for signal in signal_paralel) 
        Reconstruction_i = pd.concat(Reconstructions_i, axis=1) #mean of all the trials
        Reconstruction_i.columns =  [str(i * TR) for i in range(nscans_wm)] #column names
        Reconstructions_sh.append(Reconstruction_i) #append the reconstruction (of the current iteration)
        time_rec_shuff_end = time.time() #time
        time_rec_shuff = time_rec_shuff_end - time_rec_shuff_start
        print('shuff_' + str(n_rep) + ': ' +str(time_rec_shuff) ) #print time of the reconstruction shuffled
    
    ### Get just the supposed target location
    df_shuffle=[]
    for i in range(len(Reconstructions_sh)):
        n = Reconstructions_sh[i].iloc[ref_angle*2, :] #around the ref_angle (x2 beacuse now we have 720 instead of 360)
        n = n.reset_index()
        n.columns = ['times', 'decoding']
        n['decoding'] = [sum(Reconstructions_sh[i].iloc[:, ts] * f2(ref_angle)) for ts in range(len(n))] #population vector method (scalar product)
        n['times']=n['times'].astype(float)
        n['region'] = region
        n['subject'] = subject
        n['condition'] = condition
        df_shuffle.append(n) #save thhis
    
    ##
    df_shuffle = pd.concat(df_shuffle)    #same shape as the decosing of the signal
    return df_shuffle



def bootstrap_reconstruction(testing_activity, targets, iterations, WM, WM_t, Inter, region, condition, subject, ref_angle=180):
    Reconstructions_boots=[]
    for n_rep in range(iterations):
        time_rec_boot_start=time.time()
        indexes_boots = np.random.randint(0,len(targets), len(targets))  #bootstraped indexes for reconstruction
        ### make the reconstryctions and append them
        targets_boot = targets[indexes_boots]
        signal_boots = testing_activity[indexes_boots, :, :] 
        signal_boots_paralel =[ signal_boots[:, i, :] for i in range(nscans_wm)]
        
        Reconstructions_boot = Parallel(n_jobs = numcores)(delayed(Representation)(signal, targets_boot, WM, WM_t, ref_angle=180, plot=False, intercept=Inter)  for signal in signal_boots_paralel)    #### reconstruction standard (paralel)
        Reconstructions_boot = pd.concat(Reconstructions_boot, axis=1) #mean of all the trials
        Reconstructions_boot.columns =  [str(i * TR) for i in range(nscans_wm)] #column names
        Reconstructions_boots.append(Reconstructions_boot) #append the reconstruction (of the current iteration)
        time_rec_boot_end = time.time() #time
        time_rec_boot = time_rec_boot_end - time_rec_boot_start
        print('boot_' + str(n_rep) + ': ' +str(time_rec_boot) ) #print time of the reconstruction shuffled
        
    ### Get just the supposed target location
    df_boots=[]
    for i in range(len(Reconstructions_boots)):
        n = Reconstructions_boots[i].iloc[ref_angle*2, :] #around the ref_angle (x2 beacuse now we have 720 instead of 360)
        n = n.reset_index()
        n.columns = ['times', 'decoding']
        n['decoding'] = [sum(Reconstructions_boots[i].iloc[:, ts] * f2(ref_angle)) for ts in range(len(n))] #population vector method (scalar product)
        n['times']=n['times'].astype(float)
        n['region'] = region
        n['subject'] = subject
        n['condition'] = condition
        df_boots.append(n) #save thhis
    
    ##
    df_boots = pd.concat(df_boots)    #same shape as the decosing of the signal
    return df_boots



def all_process_condition_shuff_boot( Subject, Brain_Region, WM, WM_t, Inter, Condition, iterations, distance, method='together', heatmap=False):
    enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, method, Brain_Region)
    ##### Process testing data
    testing_activity, testing_behaviour = preprocess_wm_files(wm_fmri_paths, masks, wm_beh_paths, condition=Condition, distance=distance, sys_use='unix', nscans_wm=nscans_wm, TR=2.335)
    testing_angles = np.array(testing_behaviour['Dist'])    
    ### Respresentation
    start_repres = time.time()    
    # TR separartion
    signal_paralel =[ testing_activity[:, i, :] for i in range(nscans_wm)]
    Reconstructions = Parallel(n_jobs = numcores)(delayed(Representation)(signal, testing_angles, WM, WM_t, ref_angle=180, plot=False, intercept=Inter)  for signal in signal_paralel)    #### reconstruction standard (paralel)
    Reconstruction = pd.concat(Reconstructions, axis=1) #mean of the reconstructions (all trials)
    Reconstruction.columns =  [str(i * TR) for i in range(nscans_wm)]    ##column names
    #Plot heatmap
    if heatmap==True:
        plt.figure()
        plt.title(Condition)
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
    print( 'Time process reconstruction: ' +str(process_recons)) #print time of the process
    
    df_boots = bootstrap_reconstruction(testing_activity, testing_angles, iterations, WM, WM_t, Inter, Brain_Region, Condition, Subject, ref_angle=180)    
    ####### Shuff
    #### Compute the shuffleing
    shuffled_rec = shuffled_reconstruction(signal_paralel, testing_angles, iterations, WM, WM_t, Inter=Inter, region=Brain_Region, condition=Condition, subject=Subject, ref_angle=180)
    
    return Reconstruction, df_boots, shuffled_rec




##paths to save the 3 files 
path_save_reconstructions = '/home/david/Desktop/Reconst_LM_dist_boot_close_nit.xlsx' 
Reconstructions={}
path_save_signal ='/home/david/Desktop/signal_LM_dist_boot_close_nit.xlsx'
path_save_boots = '/home/david/Desktop/boots_LM_dist_boot_close_nit.xlsx'
path_save_shuff = '/home/david/Desktop/shuff_LM_dist_boot_close_nit.xlsx'

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
            Reconstruction, boots, shuff = all_process_condition_shuff_boot( Subject=Subject, Brain_Region=Brain_region, WM=WM, WM_t=WM_t, distance='close', iterations=50, Inter=Inter, Condition=Condition, method='together',  heatmap=False) #100
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



