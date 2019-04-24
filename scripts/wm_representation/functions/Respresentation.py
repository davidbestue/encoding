# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:16:21 2019

@author: David Bestue
"""



def Representation(Weights, testing_data, testing_angles):    
    Channel_all_trials_rolled=[] #Lists to append all the trials rolled
    for trial in range(len(testing_angles)):
        Signal = testing_data[trial, :]
        channel_36 = dot( dot ( inv( dot(Matrix_weights_transpose, Matrix_weights ) ),  Matrix_weights_transpose),  Signal) #Run the inverse model
        channel= ch2vrep3(channel_36) ###Convert 36 into 720 channels for the reconstruction

            
            #Roll
            angle_trial =  beh_Subset['T'].iloc[trial] ## get the angle of the target
            to_roll = int( (ref_angle - angle_trial)*(len(channel)/360) ) ## degrees to roll
            channel=roll(channel, to_roll) ## roll this degrees
            channels_trial.append(channel) #Append it into the trial list
            ####
            
        
        #Once all the TR of the trial are in the trial list, append this list to the global list
        Channel_all_trials_rolled.append(channels_trial)
    
    ##
    Channel_without_rolling = array(Channel_without_rolling)
    Channel_all_trials_rolled = array(Channel_all_trials_rolled)  # (trials, TRs, channels_activity) (of the session (whne together, all))
    
    #Mean of trials
    df_wr = pd.DataFrame()
    for i in range(0, nscans_wm/2):
        n = list(Channel_without_rolling[:,i,:].mean(axis=0)) #mean of all the trials rolled
        df_wr[str( round(2.335*i, 2)  )] = n #name of the column
    
    
    df = pd.DataFrame()
    for i in range(0, nscans_wm/2):
        n = list(Channel_all_trials_rolled[:,i,:].mean(axis=0)) #mean of all the trials rolled
        df[str( round(2.335*i, 2)  )] = n #name of the column
    
    
    ## plot heatmap
    #plt.figure()
    #plt.title('Heatmap decoding ' + Subject_analysis)
    ########midpoint = df.values.mean() # (df.values.max() - df.values.min()) / 2
    #ax = sns.heatmap(df, yticklabels=list(df.index), cmap="coolwarm") # cmap= viridis "jet",  "coolwarm" RdBu_r, gnuplot, YlOrRd, CMRmap  , center = midpoint
    #ax.plot([0.25, shape(df)[1]-0.25], [posch1_to_posch2(4),posch1_to_posch2(4)], 'k--')
    #plt.yticks([posch1_to_posch2(4), posch1_to_posch2(13), posch1_to_posch2(22), posch1_to_posch2(31)] ,['45','135','225', '315'])
    #plt.ylabel('Angle')
    #plt.xlabel('time (s)')
    #plt.show(block=False)
            
