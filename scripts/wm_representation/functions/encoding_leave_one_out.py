

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
numcores = multiprocessing.cpu_count() - 10


Subjects=['d001', 'n001', 'b001', 'r001', 's001', 'l001']
brain_regions = ['visual', 'ips', 'pfc']


path_save_signal='/home/david/Desktop/target_close/signal_encoding.xlsx'
path_save_shuffle='/home/david/Desktop/target_close/shuffle_encoding.xlsx'


def shuff_Pop_vect_leave_one_out(testing_data, testing_angles, iterations):
    ## A esta función entrarán los datos de un TR y haré el shuffleing. 
    ## Es como Pop_vect_leave_one_out pero en vez de dar un solo error para un scan, 
    ## de tantas iterations shuffled (contiene un loop for y un shuffle )
    ## Alternativa: En vez de hacer n_iterations, hacer el shuffleing una vez y hacer una media de todos los errores
    ## Pro alternativa: menos tiempo de computacion
    ## Contra: mas variabilidad (barras de error menos robustas)
    loo = LeaveOneOut()
    errors_shuffle=[]
    #########
    ########
    for i in range(iterations):
        # aquí estoy haciendo un shuffle normal (mezclar A_t)
        testing_angles_sh = np.array(random.sample(testing_angles, len(testing_angles)) )
        # una alternativa para que sea igual, sería asignar random 0, 90, 180 y 270
        #testing_angles_sh = np.array([random.choice([0, 90, 180, 270]) for i in range(len(testing_angles))])
        errors_=[]
        for train_index, test_index in loo.split(testing_data):
            X_train, X_test = testing_data[train_index], testing_data[test_index]
            y_train, y_test = testing_angles_sh[train_index], testing_angles_sh[test_index]
            ##
            ## correr el modelo en cada uno de los sets y guardar el error en cada uno de los trials
            ## la std no la hare con estos errores, sinó con el shuffle. No necesito guardar el error en cada repetición.
            model_trained_err = model_PV(X_train, X_test, y_train, y_test)
            errors_.append(model_trained_err) ## error de todos los train-test
        ##
        error_shuff_abs = np.mean([abs(errors_[i]) for i in range(0, len(errors_))]) 
        errors_shuffle.append(error_shuff_abs)
        #
    return errors_shuffle




Reconstructions=[]
Reconstructions_shuff=[]

for Subject in Subjects:
    for Brain_region in brain_regions:
        print(Subject, Brain_region)
        #plt.figure()
        ### Data to use
        enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, 'together', Brain_region)
        ##### Process training data
        training_dataset, training_targets = process_encoding_files(enc_fmri_paths, masks, enc_beh_paths, sys_use='unix', hd=6, TR=2.335) #4
        error_= Pop_vect_leave_one_out(training_dataset, training_targets) #no hay que hacer paralel porque no hay multiple wm
        Reconstruction = pd.DataFrame([error_]) #solo hay 1!
        Reconstruction.columns=['decoding']  
        Reconstruction['region'] = Brain_Region
        Reconstruction['subject'] = Subject
        Reconstruction['label'] = 'signal'
        Reconstructions.append(Reconstruction)
        #
        error_shuff = shuff_Pop_vect_leave_one_out(training_dataset, training_targets, 10)
        Reconstruction_shuff = pd.DataFrame(error_shuff)
        Reconstruction_shuff.columns=['decoding']  
        Reconstruction_shuff['region'] = Brain_Region
        Reconstruction_shuff['subject'] = Subject
        Reconstruction_shuff['label'] = 'shuffle'
        Reconstructions_shuff.append(Reconstruction_shuff)



### Save signal from the reconstructions and shuffles
Decoding_df = pd.concat(Reconstructions, axis=0) 
Decoding_df.to_excel( path_save_signal )

Shuffle_df = pd.concat(Reconstructions_shuff, axis=0) 
Shuffle_df.to_excel( path_save_shuffle )