# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:24:32 2019

@author: David
"""



def bootstrap_reconstruction(testing_activity, targets, iterations, WM, WM_t, Inter, region, condition, subject, ref_angle=180):
    Reconstructions_boots=[]
    signal_paralel =[ testing_activity[:, i, :] for i in range(nscans_wm)]
    ### shuffle the targets
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



def all_process_condition_shuff_boot( Subject, Brain_Region, WM, WM_t, Inter, Condition, iterations, method='together', heatmap=False):
    enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, method, Brain_Region)
    ##### Process testing data
    testing_activity, testing_behaviour = preprocess_wm_files(wm_fmri_paths, masks, wm_beh_paths, condition=Condition, distance='mix', sys_use='unix', nscans_wm=nscans_wm, TR=2.335)
    testing_angles = np.array(testing_behaviour['A_R'])    
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
    
    df_boots = bootstrap_reconstruction(testing_activity, targets, iterations, WM, WM_t, Inter, region, condition, subject, ref_angle=180)    
    ####### Shuff
    #### Compute the shuffleing
    shuffled_rec = shuffled_reconstruction(signal_paralel, testing_angles, iterations, WM, WM_t, Inter=Inter, region=Brain_Region, condition=Condition, subject=Subject, ref_angle=180)
    
    return Reconstruction, shuffled_rec, df_boots