# -*- coding: utf-8 -*-
"""
Created on Thu May 16 10:26:31 2019

@author: David Bestue
"""

## Shuffle function to get decoding value!

nscans_wm=16
signal_paralel =[ testing_activity[:, i, :] for i in range(nscans_wm)]


def shuffled_reconstruction(signal_paralel, targets, iterations, WM, WM_t, ref_angle=180):
    testing_angles_sh=[]
    for n_rep in range(iterations):
        random.shuffle(targets)
        testing_angles_sh.append(targets)
    
    ##
    Reconstructions_sh=[]
    for n_rep in range(iterations):
        Reconstructions_i = Parallel(n_jobs = numcores)(delayed(Representation)(signal, testing_angles_sh[n_rep], WM, WM_t, ref_angle=180, plot=False)  for signal in signal_paralel) 
        Reconstruction_i = pd.concat(Reconstructions_i, axis=1) 
        Reconstruction_i.columns =  [str(i * TR) for i in range(nscans_wm)]
        Reconstructions_sh.append(Reconstruction_i)
    
    return Reconstructions_sh



shuffled_reconstruction(signal_paralel, testing_angles, 3, WM, WM_t)



Reconstructions_sh=[]
for n_rep in range(iterations):
    Reconstructions_i = Parallel(n_jobs = numcores)(delayed(Representation)(signal, testing_angles, WM, WM_t, ref_angle=180, plot=False)  for signal in signal_paralel) 
    Reconstructions_sh.append(Reconstructions_i)



def shuffled_reconstruction(testing_dataset, targets):
    
    ### Compares the decoding you get with the shuffled one.
    
    ##random.shuffle(testing_angles)

    ### Respresentation
    start_repres = time.time()
    numcores = multiprocessing.cpu_count()
    
    # TR separartion
    signal_paralel =[ testing_activity[:, i, :] for i in range(nscans_wm)]
    Reconstructions = Parallel(n_jobs = numcores)(delayed(Representation)(signal, testing_angles, WM, WM_t, ref_angle=180, plot=False)  for signal in signal_paralel)    ####
    Reconstruction = pd.concat(Reconstructions, axis=1) 
    Reconstruction.columns =  [str(i * TR) for i in range(nscans_wm)]



def shuffled_reconstruction(testing_activity, testing_angles):
    signal_paralel =[ testing_activity[:, i, :] for i in range(nscans_wm)]
    Reconstructions = Parallel(n_jobs = numcores)(delayed(Representation)(signal, testing_angles, WM, WM_t, ref_angle=180, plot=False)  for signal in signal_paralel) 






def trial_rep(Signal, angle_trial, Weights, Weights_t, ref):
    channel_36 = np.dot( np.dot ( inv( np.dot(Weights_t, Weights ) ),  Weights_t),  Signal) #Run the inverse model
    channel= ch2vrep3(channel_36) ###Convert 36 into 720 channels for the reconstruction
    to_roll = int( (ref - angle_trial)*(len(channel)/360) ) ## degrees to roll
    channel=np.roll(channel, to_roll) ## roll this degrees
    return channel



def Representation(testing_data, testing_angles, Weights, Weights_t, ref_angle=180, plot=False):
    ## Make the data parallelizable
    n_trials_test = len(testing_data) #number trials
    data_prall = []
    for i in range(n_trials_test):
        data_prall.append(testing_data[i, :])
    
    ###
    numcores = multiprocessing.cpu_count()
    Channel_all_trials_rolled = Parallel(n_jobs = numcores)(delayed(trial_rep)(Signal, angle_trial, Weights, Weights_t, ref=ref_angle)  for Signal, angle_trial in zip( data_prall, testing_angles))    ####
    Channel_all_trials_rolled = np.array(Channel_all_trials_rolled)
    
    df = pd.DataFrame()
    n = list(Channel_all_trials_rolled.mean(axis=0)) #mean of all the trials rolled
    df['TR'] = n #Name of the column
    if plot==True:
        #Plot heatmap
        plt.figure()
        plt.title('Heatmap decoding')
        ######midpoint = df.values.mean() # (df.values.max() - df.values.min()) / 2
        ax = sns.heatmap(df, yticklabels=list(df.index), cmap="coolwarm") # cmap= viridis "jet",  "coolwarm" RdBu_r, gnuplot, YlOrRd, CMRmap  , center = midpoint
        ax.plot([0.25, np.shape(df)[1]-0.25], [posch1_to_posch2(18),posch1_to_posch2(18)], 'k--')
        plt.yticks([posch1_to_posch2(4), posch1_to_posch2(13), posch1_to_posch2(22), posch1_to_posch2(31)] ,['45','135','225', '315'])
        plt.ylabel('Angle')
        plt.xlabel('time (s)')
        plt.show(block=False)
    
    return df





