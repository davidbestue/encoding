# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:16:21 2019

@author: David Bestue
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from basic_functions import *

def Representation(Weights, testing_data, testing_angles):   
    ref_angle=45
    Channel_all_trials_rolled=[] #Lists to append all the trials rolled
    for trial in range(len(testing_angles)):
        Signal = testing_data[trial, :]
        channel_36 = np.dot( np.dot ( np.inv( np.dot(Matrix_weights_transpose, Matrix_weights ) ),  Matrix_weights_transpose),  Signal) #Run the inverse model
        channel= ch2vrep3(channel_36) ###Convert 36 into 720 channels for the reconstruction
        #Roll
        angle_trial =  testing_angles[angle]
        to_roll = int( (ref_angle - angle_trial)*(len(channel)/360) ) ## degrees to roll
        channel=np.roll(channel, to_roll) ## roll this degrees
        Channel_all_trials_rolled.append(channel)
    
    ##
    Channel_all_trials_rolled = np.array(Channel_all_trials_rolled)  # (trials, TRs, channels_activity) (of the session (whne together, all))
    df = pd.DataFrame()
    n = list(Channel_all_trials_rolled.mean(axis=0)) #mean of all the trials rolled
    df['time_x'] = n #name of the column
    
    # plot heatmap
    plt.figure()
    plt.title('Heatmap decoding')
    #######midpoint = df.values.mean() # (df.values.max() - df.values.min()) / 2
    ax = sns.heatmap(df, yticklabels=list(df.index), cmap="coolwarm") # cmap= viridis "jet",  "coolwarm" RdBu_r, gnuplot, YlOrRd, CMRmap  , center = midpoint
    ax.plot([0.25, np.shape(df)[1]-0.25], [posch1_to_posch2(4),posch1_to_posch2(4)], 'k--')
    plt.yticks([posch1_to_posch2(4), posch1_to_posch2(13), posch1_to_posch2(22), posch1_to_posch2(31)] ,['45','135','225', '315'])
    plt.ylabel('Angle')
    plt.xlabel('time (s)')
    plt.show(block=False)
    
    return df
    
    

            
