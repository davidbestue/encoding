

from model_functions import *
from fake_data_generator import *
from Weights_matrixs import *
from Representation import *
from process_encoding import *
from process_wm import *
from data_to_use import *
from bootstrap_functions import *
from leave_one_out import *
from joblib import Parallel, delayed
import multiprocessing
import time
import random
#
numcores = multiprocessing.cpu_count() - 3


Subjects=['d001', 'n001', 'b001', 'r001', 's001', 'l001']
brain_regions = ['visual', 'ips', 'pfc']


path_save_signal='/home/david/Desktop/target_close/signal_encoding.xlsx'
path_save_shuffle='/home/david/Desktop/target_close/shuffle_encoding.xlsx'


Reconstructions=[]
Reconstructions_shuff=[]

for Subject in Subjects:
    for Brain_region in brain_regions:
        print(Subject + ', ' + Brain_region)
        #plt.figure()
        ### Data to use
        enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, 'together', Brain_region)
        ##### Process training data
        training_dataset, training_targets = process_encoding_files(enc_fmri_paths, masks, enc_beh_paths, sys_use='unix', hd=6, TR=2.335) #4
        error_= Pop_vect_leave_one_out(training_dataset, training_targets) #no hay que hacer paralel porque no hay multiple wm
        Reconstruction = pd.DataFrame([error_]) #solo hay 1!
        Reconstruction.columns=['decoding']  
        Reconstruction['region'] = Brain_region
        Reconstruction['subject'] = Subject
        Reconstruction['label'] = 'signal'
        Reconstructions.append(Reconstruction)
        #
        error_shuff = shuff_Pop_vect_leave_one_out2(training_dataset, training_targets, 10)
        Reconstruction_shuff = pd.DataFrame(error_shuff)
        Reconstruction_shuff.columns=['decoding']  
        Reconstruction_shuff['region'] = Brain_region
        Reconstruction_shuff['subject'] = Subject
        Reconstruction_shuff['label'] = 'shuffle'
        Reconstructions_shuff.append(Reconstruction_shuff)



### Save signal from the reconstructions and shuffles
Decoding_df = pd.concat(Reconstructions, axis=0) 
Decoding_df.to_excel( path_save_signal )

Shuffle_df = pd.concat(Reconstructions_shuff, axis=0) 
Shuffle_df.to_excel( path_save_shuffle )