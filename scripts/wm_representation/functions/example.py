# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:32:56 2019

@author: David Bestue
"""

from basic_functions import *
from fake_data_generator import *
from Weigths_matrix import *
from Representation import *
import random
from joblib import Parallel, delayed
import multiprocessing


numcores = multiprocessing.cpu_count()


n_trials_train=900
training_angles = np.array([ random.randint(0,359) for i in range(n_trials_train)])
training_data = fake_data(training_angles) 

##
WM = Weights_matrix( training_data, training_angles )
WM_t = WM.transpose()
##

n_trials_test=500
testing_angles = np.array([ random.randint(0,359) for i in range(n_trials_test)])
testing_data = fake_data(testing_angles)
data_prall = []
for i in range(300):
    data_prall.append(testing_data[i, :])
#random.shuffle(testing_angles)


#### paralel!

def trial_rep(Signal, angle_trial, Weights, Weights_t, ref_angle=180):
    channel_36 = np.dot( np.dot ( inv( np.dot(Weights_t, Weights ) ),  Weights_t),  Signal) #Run the inverse model
    channel= ch2vrep3(channel_36) ###Convert 36 into 720 channels for the reconstruction
    to_roll = int( (ref_angle - angle_trial)*(len(channel)/360) ) ## degrees to roll
    channel=np.roll(channel, to_roll) ## roll this degrees
    return channel

      
numcores = multiprocessing.cpu_count()
Channel_all_trials_rolled = Parallel(n_jobs = numcores)(delayed(trial_rep)(Signal, angle_trial, WM, WM_t, ref_angle=180)  for Signal, angle_trial in zip( data_prall, testing_angles))

Channel_all_trials_rolled = np.array(Channel_all_trials_rolled)
df = pd.DataFrame()
n = list(Channel_all_trials_rolled.mean(axis=0)) #mean of all the trials rolled
df['time_x'] = n #Name of the column
#Plot heatmap
plt.figure()
plt.title('Heatmap decoding')
#######midpoint = df.values.mean() # (df.values.max() - df.values.min()) / 2
ax = sns.heatmap(df, yticklabels=list(df.index), cmap="coolwarm") # cmap= viridis "jet",  "coolwarm" RdBu_r, gnuplot, YlOrRd, CMRmap  , center = midpoint
ax.plot([0.25, np.shape(df)[1]-0.25], [posch1_to_posch2(18),posch1_to_posch2(18)], 'k--')
plt.yticks([posch1_to_posch2(4), posch1_to_posch2(13), posch1_to_posch2(22), posch1_to_posch2(31)] ,['45','135','225', '315'])
plt.ylabel('Angle')
plt.xlabel('time (s)')
plt.show(block=False)






