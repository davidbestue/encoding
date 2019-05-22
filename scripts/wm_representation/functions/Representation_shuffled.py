# -*- coding: utf-8 -*-
"""
Created on Thu May 16 10:26:31 2019

@author: David Bestue
"""

## Shuffle function to get decoding value!


#iterations = 4
#
#testing_angles_sh=[]
#for n_rep in range(iterations):
#    random.shuffle(testing_angles)
#    testing_angles_sh.append(testing_angles)
#
#
####   
#
#rep_shuff = Parallel(n_jobs = numcores)(delayed(Representation)(testing_activity, testing_angles, WM, WM_t, ref_angle=180, plot=False) for testing_angles in testing_angles_sh) 



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



shuffled_rec = shuffled_reconstruction(signal_paralel, testing_angles, 25, WM, WM_t, Inter=False, region=, condition=, subject=)



#df_shuffle=[]
#for i in range(len(shuffled_rec)):
#    n = shuffled_rec[i].iloc[360, :]
#    n = n.reset_index()
#    n.columns = ['times', 'decoding']
#    n['times']=n['times'].astype(float)
#    
#    
#    df_shuffle.append(n)
#
#
#df_shuffle=pd.concat(df_shuffle)
#
#
#    df = R[dataframes]
#    a = pd.DataFrame(df.iloc[360,:])
#    a = a.reset_index()
#    a.columns = ['times', 'decoding']
#    a['times']=a['times'].astype(float)
#    a['region'] = dataframes.split('_')[1]
#    a['subject'] = dataframes.split('_')[0]
#    a['condition'] = dataframes.split('_')[-2] + '_' + dataframes.split('_')[-1] 
#    Decoding_df.append(a)



#df_shuff = {}
#for i in range(len(sh2_rec)):
#    df_shuff[str(i)] = sh2_rec[i]
#
#
#panel = pd.Panel(df_shuff)
#df_shuff_100_means = panel.mean(axis=0)
#df_shuff_100_means.columns =  [str(i * TR) for i in range(nscans_wm)]
#df_shuff_100_stds = panel.std(axis=0)
#df_shuff_100_stds.columns =  [str(i * TR) for i in range(nscans_wm)]


#Plot heatmap
plt.figure()
plt.title('Heatmap decoding')
######midpoint = df.values.mean() # (df.values.max() - df.values.min()) / 2
ax = sns.heatmap(df_shuff_100, yticklabels=list(Reconstruction.index), cmap="coolwarm", vmin=-0.4, vmax = 0.6) # cmap= viridis "jet",  "coolwarm" RdBu_r, gnuplot, YlOrRd, CMRmap  , center = midpoint
ax.plot([0.25, np.shape(Reconstruction)[1]-0.25], [posch1_to_posch2(18),posch1_to_posch2(18)], 'k--')
plt.yticks([posch1_to_posch2(4), posch1_to_posch2(13), posch1_to_posch2(22), posch1_to_posch2(31)] ,['45','135','225', '315'])
plt.ylabel('Angle')
plt.xlabel('time (s)')
plt.tight_layout()
plt.show(block=False)


a = pd.DataFrame(Reconstruction.iloc[360,:])
a = a.reset_index()
a.columns = ['times', 'decoding']
a['times']=a['times'].astype(float)


#b = pd.DataFrame(df_shuff_100_means.iloc[360,:])
#b= b.reset_index()
#b.columns = ['times', 'decoding']



#sh2_rec = shuffled_reconstruction(signal_paralel, testing_angles, 100, WM, WM_t)

df_shuffle=[]
for i in range(len(sh2_rec)):
    n = sh2_rec[i].iloc[360, :]
    n = n.reset_index()
    n.columns = ['times', 'decoding']
    n['times']=n['times'].astype(float)
    df_shuffle.append(n)


df_shuffle=pd.concat(df_shuffle)

#
#fmri = sns.load_dataset("fmri")
#ax = sns.lineplot(x="timepoint", y="signal", data=fmri)
#


#
#
#Reconstructions_sh=[]
#for n_rep in range(iterations):
#    Reconstructions_i = Parallel(n_jobs = numcores)(delayed(Representation)(signal, testing_angles, WM, WM_t, ref_angle=180, plot=False)  for signal in signal_paralel) 
#    Reconstructions_sh.append(Reconstructions_i)
#
#
#
#def shuffled_reconstruction(testing_dataset, targets):
#    
#    ### Compares the decoding you get with the shuffled one.
#    
#    ##random.shuffle(testing_angles)
#
#    ### Respresentation
#    start_repres = time.time()
#    numcores = multiprocessing.cpu_count()
#    
#    # TR separartion
#    signal_paralel =[ testing_activity[:, i, :] for i in range(nscans_wm)]
#    Reconstructions = Parallel(n_jobs = numcores)(delayed(Representation)(signal, testing_angles, WM, WM_t, ref_angle=180, plot=False)  for signal in signal_paralel)    ####
#    Reconstruction = pd.concat(Reconstructions, axis=1) 
#    Reconstruction.columns =  [str(i * TR) for i in range(nscans_wm)]
#
#
#
#def shuffled_reconstruction(testing_activity, testing_angles):
#    signal_paralel =[ testing_activity[:, i, :] for i in range(nscans_wm)]
#    Reconstructions = Parallel(n_jobs = numcores)(delayed(Representation)(signal, testing_angles, WM, WM_t, ref_angle=180, plot=False)  for signal in signal_paralel) 
#
#
#
#
#
#
#def trial_rep(Signal, angle_trial, Weights, Weights_t, ref):
#    channel_36 = np.dot( np.dot ( inv( np.dot(Weights_t, Weights ) ),  Weights_t),  Signal) #Run the inverse model
#    channel= ch2vrep3(channel_36) ###Convert 36 into 720 channels for the reconstruction
#    to_roll = int( (ref - angle_trial)*(len(channel)/360) ) ## degrees to roll
#    channel=np.roll(channel, to_roll) ## roll this degrees
#    return channel
#
#
#
#def Representation(testing_data, testing_angles, Weights, Weights_t, ref_angle=180, plot=False):
#    ## Make the data parallelizable
#    n_trials_test = len(testing_data) #number trials
#    data_prall = []
#    for i in range(n_trials_test):
#        data_prall.append(testing_data[i, :])
#    
#    ###
#    numcores = multiprocessing.cpu_count()
#    Channel_all_trials_rolled = Parallel(n_jobs = numcores)(delayed(trial_rep)(Signal, angle_trial, Weights, Weights_t, ref=ref_angle)  for Signal, angle_trial in zip( data_prall, testing_angles))    ####
#    Channel_all_trials_rolled = np.array(Channel_all_trials_rolled)
#    
#    df = pd.DataFrame()
#    n = list(Channel_all_trials_rolled.mean(axis=0)) #mean of all the trials rolled
#    df['TR'] = n #Name of the column
#    if plot==True:
#        #Plot heatmap
#        plt.figure()
#        plt.title('Heatmap decoding')
#        ######midpoint = df.values.mean() # (df.values.max() - df.values.min()) / 2
#        ax = sns.heatmap(df, yticklabels=list(df.index), cmap="coolwarm") # cmap= viridis "jet",  "coolwarm" RdBu_r, gnuplot, YlOrRd, CMRmap  , center = midpoint
#        ax.plot([0.25, np.shape(df)[1]-0.25], [posch1_to_posch2(18),posch1_to_posch2(18)], 'k--')
#        plt.yticks([posch1_to_posch2(4), posch1_to_posch2(13), posch1_to_posch2(22), posch1_to_posch2(31)] ,['45','135','225', '315'])
#        plt.ylabel('Angle')
#        plt.xlabel('time (s)')
#        plt.show(block=False)
#    
#    return df
#
#
#
#

