# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:32:56 2019
@author: David Bestue
"""

import statsmodels.api as sm
import sys
sys.path.append('/home/david/Desktop/GitHub/encoding/scripts/wm_representation/functions/')
from data_to_use import *
from process_encoding import *


Subject='n001'
Brain_region='visual'


### Data to use
enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, 'together', Brain_region)
##### Process training data
training_dataset, training_targets = process_encoding_files(enc_fmri_paths, masks, enc_beh_paths, sys_use='unix', hd=6, TR=2.335) #4
##### Train your weigths
WM, Inter = Weights_matrix_LM( training_dataset, training_targets )
WM_t = WM.transpose()



def train_each_vxl( training_dataset, training_targets ):
    #
    ### X Training
    ## X matrix (intercept and spikes)
    X = np.column_stack([np.ones(np.shape(training_dataset)[0]),  training_dataset])
    ## Y (sinus and cos of the target)
    sinus =np.sin([np.radians(np.array(training_targets)[i]) for i in range(0, len(training_targets))])
    cosinus = np.cos([np.radians(np.array(training_targets)[i]) for i in range(0, len(training_targets))])
    Y = np.column_stack([cosinus, sinus])
    Y = Y.astype(float) #to make it work in the cluster
    X = X.astype(float)
    model = sm.OLS(Y, X)
    ##train the model
    training_weights = model.fit()  ## training_weights.params
    return training_weights







    return weights



spikes_train, spikes_test, beh_train, beh_test = train_test_split(df.groupby('neuron').get_group(Neur)['spikes'],
                                                                  df.groupby('neuron').get_group(Neur)['beh'],
                                                                  test_size=size_test)  

######## Trainning #########
## X matrix (intercept and spikes)
X = np.column_stack([np.ones(np.shape(spikes_train)[0]),spikes_train])
## Y (sinus and cos)
sinus =np.sin([np.radians(np.array(beh_train)[i]) for i in range(0, len(beh_train))])
cosinus = np.cos([np.radians(np.array(beh_train)[i]) for i in range(0, len(beh_train))])
Y = np.column_stack([cosinus, sinus])
### one OLS for sin and cos: output: beta of intercetp and bea of spikes (two B intercepts and 2 B for spikes )
Y = Y.astype(float) #to make it work in the cluster
X = X.astype(float)
model = sm.OLS(Y, X)
##train the model
training_weights = model.fit()
return training_weights





######### Testing ###########
X = np.column_stack([np.ones(np.shape(spikes_test)[0]),spikes_test])
p = fit.predict(X)
x = p[:,0]
y = p[:,1]
#####
##### Error --> take the resulting vector in sin/cos space
### from sin and cos get the angle (-pi, pi)
#pred_angle = [ np.degrees(np.arctan2(y[i], x[i]) + np.pi) for i in range(0, len(y))]
pred_angle = [ np.degrees(np.arctan2(y[i], x[i])) for i in range(0, len(y))]
for i in range(0, len(pred_angle)):
    if pred_angle[i]<0:
        pred_angle[i]=360+pred_angle[i]
##
#error=[ circdist(beh_test[i], pred_angle[i]) for i in range(0, len(pred_angle))]
error=[ circdist(beh_test.values[i], pred_angle[i]) for i in range(0, len(pred_angle))]

#low_value --> predicted positionns close to real
neur_err.append(np.mean(error))













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


### Data to use
enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( 'n001', 'together', 'visual')

##### Process training data
training_dataset, training_targets = process_encoding_files(enc_fmri_paths, masks, enc_beh_paths, sys_use='unix', hd=4, TR=2.335)

##### Train your weigths
WM, Inter = Weights_matrix_Lasso_i( training_dataset, training_targets )
WM_t = WM.transpose()

##### Process testing data
testing_activity, testing_behaviour = preprocess_wm_files(wm_fmri_paths, masks, wm_beh_paths, condition='2_7', distance='mix', sys_use='unix', nscans_wm=nscans_wm, TR=2.335)
testing_angles = np.array(testing_behaviour['T'])
##random.shuffle(testing_angles)

### Respresentation
start_repres = time.time()
numcores = multiprocessing.cpu_count()

# TR separartion
signal_paralel =[ testing_activity[:, i, :] for i in range(nscans_wm)]
Reconstructions = Parallel(n_jobs = numcores)(delayed(Representation)(signal, testing_angles, WM, WM_t, ref_angle=180, plot=False, intercept=Inter)  for signal in signal_paralel)    ####
Reconstruction = pd.concat(Reconstructions, axis=1) 
Reconstruction.columns =  [str(i * TR) for i in range(nscans_wm)]

#Plot heatmap
plt.figure()
plt.title('Heatmap decoding')
######midpoint = df.values.mean() # (df.values.max() - df.values.min()) / 2
ax = sns.heatmap(Reconstruction, yticklabels=list(Reconstruction.index), cmap="coolwarm") #, vmin=-0.4, vmax = 0.6) # cmap= viridis "jet",  "coolwarm" RdBu_r, gnuplot, YlOrRd, CMRmap  , center = midpoint
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

