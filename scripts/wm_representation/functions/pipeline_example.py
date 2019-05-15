# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:32:56 2019
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


### Data to use
enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( 'n001', 'together', 'visual')

##### Process training data
training_dataset, training_targets = process_encoding_files(enc_fmri_paths, masks, enc_beh_paths, sys_use='unix', hd=6, TR=2.335)

##### Train your weigths
WM = Weights_matrix_LM( training_dataset, training_targets )
WM_t = WM.transpose()

##### Process testing data
testing_activity, testing_behaviour = process_wm_files(wm_fmri_paths, masks, wm_beh_paths, condition='1_0.2', distance='mix', sys_use='unix', nscans_wm=nscans_wm, TR=2.335)
testing_angles = np.array(testing_behaviour['T'])
#random.shuffle(testing_angles)

### Respresentation
start_repres = time.time()
numcores = multiprocessing.cpu_count()

# TR separartion
signal_paralel =[ testing_activity[:, i, :] for i in range(nscans_wm)]
Reconstructions = Parallel(n_jobs = numcores)(delayed(Representation)(signal, testing_angles, WM, WM_t, ref_angle=180, plot=False)  for signal in signal_paralel)    ####
Reconstruction = pd.concat(Reconstructions, axis=1) 
Reconstruction.columns =  [str(i * TR) for i in range(nscans_wm)]

#Plot heatmap
plt.figure()
plt.title('Heatmap decoding')
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





