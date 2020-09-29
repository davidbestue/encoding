# -*- coding: utf-8 -*-

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
from sklearn.model_selection import LeaveOneOut
from sklearn import svm


numcores = multiprocessing.cpu_count() - 10



def get_octave(angle):
    #
    if angle>=0 and angle<=45:
        oc=1
    elif angle>45 and angle<=90:
        oc=2
    elif angle>90 and angle<=135:
        oc=3
    elif angle>135 and angle<=180:
        oc=4
    elif angle>180 and angle<=225:
        oc=5
    elif angle>225 and angle<=270:
        oc=6
    elif angle>270 and angle<=315:
        oc=7
    elif angle>315 and angle<=360:
        oc=8
    ###
    return oc


###‘linear’, ‘rbf’, ‘sigmoid’, ‘precomputed’

# def model_SVM(X_train, X_test, y_train, y_test):
#     ##
#     ######## Trainning #########
#     ker = 'linear'
#     clf = svm.NuSVC(gamma='auto', kernel=ker,  nu=0.1)
#     clf.fit(X_train, y_train)
#     ######## Testing ##########
#     prediction = clf.predict(X_test)
#     ##### accuracy
#     accuracy_ = np.mean(y_test==prediction)
#     ##
#     return accuracy_


def model_SVM(X_train, X_test, y_train, y_test):
    ##
    ######## Trainning #########
    ### clf = svm.SVC(kernel='rbf', nu=0.1)
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    ######## Testing ##########
    prediction = clf.predict(X_test)
    ##### accuracy
    accuracy_ = np.mean(y_test==prediction)
    ##
    return accuracy_




#####
#####

def SVM_l1o_octv(testing_data, testing_octaves):
    ## A esta función entrarán los datos de un TR. 
    ## Como se ha de hacer el leave one out para estimar el error, no puedo paralelizar por trials
    ## Separar en train and test para leave on out procedure
    ## Hago esto para tener la mejor estimación posible del error (no hay training task)
    ## Si hubiese training task (aquí no la uso), no sería necesario el leave one out
    loo = LeaveOneOut()
    accuracies_=[]
    for train_index, test_index in loo.split(testing_data):
        X_train, X_test = testing_data[train_index], testing_data[test_index]
        y_train, y_test = testing_octaves[train_index], testing_octaves[test_index]
        ##
        ## correr el modelo en cada uno de los sets y guardar el error en cada uno de los trials
        ## la std no la hare con estos errores, sinó con el shuffle. No necesito guardar el error en cada repetición.
        model_trained_acc = model_SVM(X_train, X_test, y_train, y_test)
        #### aquí se guarda 1 o 0 (el que falta está bien o mal clasificado)
        accuracies_.append(model_trained_acc)
    ##
    l10_acc = np.mean(accuracies_) 
    return l10_acc



#####
#####

def shuff_SVM_l1o_octv(testing_data, testing_octaves, iterations):
    ## A esta función entrarán los datos de un TR y haré el shuffleing. 
    ## Es como Pop_vect_leave_one_out pero en vez de dar un solo error para un scan, 
    ## de tantas iterations shuffled (contiene un loop for y un shuffle )
    ## Alternativa: En vez de hacer n_iterations, hacer el shuffleing una vez y hacer una media de todos los errores
    ## Pro alternativa: menos tiempo de computacion
    ## Contra: mas variabilidad (barras de error menos robustas)
    loo = LeaveOneOut()
    accuracies_shuffle=[]
    #########
    ########
    for i in range(iterations):
        # aquí estoy haciendo un shuffle normal (mezclar A_t)
        testing_octaves_sh = np.array([random.choice([1,2,3,4,5,6,7,8]) for i in range(len(testing_octaves))])
        # aquí estoy haciendo un shuffle forzando que acabe en uno de los otros 3 quadrantes
        #testing_quadrants_sh = np.array( [random.choice(list(set([1,2,3,4]) - set([testing_quadrants[i]]))) for i in range(len(testing_quadrants))])
        ##
        accs_=[]
        for train_index, test_index in loo.split(testing_data):
            X_train, X_test = testing_data[train_index], testing_data[test_index]
            y_train, y_test = testing_octaves_sh[train_index], testing_octaves_sh[test_index]
            ##
            ## correr el modelo en cada uno de los sets y guardar el error en cada uno de los trials
            ## la std no la hare con estos errores, sinó con el shuffle. No necesito guardar el error en cada repetición.
            model_trained_acc = model_SVM(X_train, X_test, y_train, y_test)
            accs_.append(model_trained_acc) ## error de todos los train-test
        ##
        error_shuff_abs = np.mean(accs_) 
        accuracies_shuffle.append(error_shuff_abs)
        #
    return accuracies_shuffle



#####
#####
def shuff_SVM_l1o2_octv(testing_data, testing_octaves, iterations):
    ## A esta función entrarán los datos de un TR y haré el shuffleing. 
    ## Es como Pop_vect_leave_one_out pero en vez de dar un solo error para un scan, 
    ## de tantas iterations shuffled (contiene un loop for y un shuffle )
    ## Alternativa: En vez de hacer n_iterations, hacer el shuffleing una vez y hacer una media de todos los errores
    ## Pro alternativa: menos tiempo de computacion
    ## Contra: mas variabilidad (barras de error menos robustas)
    loo = LeaveOneOut()
    accuracies_shuffle=[]
    #########
    ########
    for i in range(iterations):
        # aquí estoy haciendo un shuffle normal (mezclar A_t)
        #testing_quadrants_sh = np.array([random.choice([1,2,3,4]) for i in range(len(testing_quadrants))])
        # aquí estoy haciendo un shuffle forzando que acabe en uno de los otros 3 quadrantes
        testing_octaves_sh = np.array( [random.choice(list(set([1,2,3,4,5,6,7,8]) - set([testing_octaves[i]]))) for i in range(len(testing_octaves))])
        ##
        accs_=[]
        for train_index, test_index in loo.split(testing_data):
            X_train, X_test = testing_data[train_index], testing_data[test_index]
            y_train, y_test = testing_octaves_sh[train_index], testing_octaves_sh[test_index]
            ##
            ## correr el modelo en cada uno de los sets y guardar el error en cada uno de los trials
            ## la std no la hare con estos errores, sinó con el shuffle. No necesito guardar el error en cada repetición.
            model_trained_acc = model_SVM(X_train, X_test, y_train, y_test)
            accs_.append(model_trained_acc) ## error de todos los train-test
        ##
        acc_shuff_abs = np.mean(accs_) 
        accuracies_shuffle.append(acc_shuff_abs)
        #
    return accuracies_shuffle



#####
##### 
def get_octvs_missing(angleT, angleNT1, angleNT2, angleD, angleDNT1, angleDNT2):
    o_t = get_octave(angleT)
    o_nt1 = get_octave(angleNT1)
    o_nt2 = get_octave(angleNT2)
    o_d = get_octave(angleD)
    o_dnt1 = get_octave(angleDNT1)
    o_dnt2 = get_octave(angleDNT2)
    octaves__ = [o_t, o_nt1, o_nt2, o_d, o_dnt1, o_dnt2]
    ##
    octaves=[1,2,3,4,5,6,7,8]
    ##
    missing = list(set(octaves) - set(octaves__))
    ##
    return missing



def shuff_SVM_l1o3_octv(testing_data, test_beh, iterations):
    ## A esta función entrarán los datos de un TR y haré el shuffleing. 
    ## Es como Pop_vect_leave_one_out pero en vez de dar un solo error para un scan, 
    ## de tantas iterations shuffled (contiene un loop for y un shuffle )
    ## Alternativa: En vez de hacer n_iterations, hacer el shuffleing una vez y hacer una media de todos los errores
    ## Pro alternativa: menos tiempo de computacion
    ## Contra: mas variabilidad (barras de error menos robustas)
    loo = LeaveOneOut()
    accuracies_shuffle=[]
    #########
    ########
    for i in range(iterations):
        # aquí estoy haciendo un shuffle forzando que acabe en una octava en la que no haya nada
        miss_octvs_trials = [get_octvs_missing(test_beh['T'].iloc[i], test_beh['NT1'].iloc[i], test_beh['NT2'].iloc[i], 
            test_beh['Dist'].iloc[i], test_beh['Dist_NT1'].iloc[i], test_beh['Dist_NT2'].iloc[i]) for i in range(len(test_beh))]
        testing_octaves_sh = np.array( [random.choice(miss_octvs_trials[i]) for i in range(len(test_beh))])
        ##
        accs_=[]
        for train_index, test_index in loo.split(testing_data):
            X_train, X_test = testing_data[train_index], testing_data[test_index]
            y_train, y_test = testing_octaves_sh[train_index], testing_octaves_sh[test_index]
            ##
            ## correr el modelo en cada uno de los sets y guardar el error en cada uno de los trials
            ## la std no la hare con estos errores, sinó con el shuffle. No necesito guardar el error en cada repetición.
            model_trained_acc = model_SVM(X_train, X_test, y_train, y_test)
            accs_.append(model_trained_acc) ## error de todos los train-test
        ##
        acc_shuff_abs = np.mean(accs_) 
        accuracies_shuffle.append(acc_shuff_abs)
        #
    return accuracies_shuffle






def l1o_octv_SVM_shuff( Subject, Brain_Region, Condition, iterations, distance, decode_item, method='together', heatmap=False):
    enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, method, Brain_Region)
    ##### Process testing data
    testing_activity, testing_behaviour = preprocess_wm_files(wm_fmri_paths, masks, wm_beh_paths, condition=Condition, distance=distance, sys_use='unix', nscans_wm=nscans_wm, TR=2.335)
    if decode_item == 'Target':
        dec_I = 'T'
    elif decode_item == 'Response':
        dec_I = 'A_R'
    elif decode_item == 'Distractor':
        dec_I = 'Dist'
    else:
        'Error specifying the decode item'
    #
    #
    start_l1out = time.time()  
    testing_angles_beh = np.array(testing_behaviour[dec_I])    # A_R # T # Dist
    octaves_angles_beh = np.array([get_octave(testing_angles_beh[i]) for i in range(len(testing_angles_beh))] )
    octaves_paralel= [octaves_angles_beh for i in range(nscans_wm)]
    ##
    signal_paralel_testing =[ testing_activity[:, i, :] for i in range(nscans_wm)] #separate for nscans (to run in parallel)
    ### Error in each TR done with leave one out
    acc_TR = Parallel(n_jobs = numcores)(delayed(SVM_l1o_octv)(testing_data = signal, testing_octaves= octv)  for signal, octv in zip(signal_paralel, octaves_paralel))    #### reconstruction standard (paralel)
    ### save in the right format for the plots
    Reconstruction = pd.DataFrame(acc_TR) #mean error en each TR (1 fila con n_scans columnas)
    Reconstruction['times']=[i * TR for i in range(nscans_wm)]
    Reconstruction.columns=['decoding', 'time']  
    Reconstruction['region'] = Brain_Region
    Reconstruction['subject'] = Subject
    Reconstruction['condition'] = Condition
    ###
    ###
    end_l1out = time.time()
    process_l1out = end_l1out - start_l1out
    print( 'Time process leave one out: ' +str(process_l1out)) #print time of the process
    ####### Shuff
    #### Compute the shuffleing (n_iterations defined on top)
    start_shuff = time.time()
    itera_paralel=[iterations for i in range(nscans_wm)]
    #
    # Aquí hay que diferenciar segun el shuffle que se usa un metodo de shuffle u otro, ya que el input no es el mismo (hay que tener uno comentado)
    ### shuff_SVM_l1o_octv and shuff_SVM_l1o2_octv
    #shuffled_rec = Parallel(n_jobs = numcores)(delayed(shuff_SVM_l1o2_octv)(testing_data=signal_s, testing_octaves=octv_s, iterations=itera) for signal_s, octv_s, itera in zip(signal_paralel, octaves_paralel, itera_paralel))
    ### shuff_SVM_l1o3_octv
    testing_angles_beh_paralel = [testing_behaviour for i in range(nscans_wm)]
    shuffled_rec = Parallel(n_jobs = numcores)(delayed(shuff_SVM_l1o3_octv)(testing_data=signal_s, test_beh=beh_s, iterations=itera) for signal_s, beh_s, itera in zip(signal_paralel, testing_angles_beh_paralel, itera_paralel))
    ### Save in the right format for the plots
    Reconstruction_sh = pd.DataFrame(shuffled_rec) #
    Reconstruction_sh = Reconstruction_sh.transpose()
    Reconstruction_sh.columns =  [str(i * TR) for i in range(nscans_wm)]  #mean error en each TR (n_iterations filas con n_scans columnas)
    Reconstruction_sh=Reconstruction_sh.melt()
    Reconstruction_sh.columns=['times', 'decoding'] 
    Reconstruction_sh['times'] = Reconstruction_sh['times'].astype(float) 
    Reconstruction_sh['region'] = Brain_Region
    Reconstruction_sh['subject'] = Subject
    Reconstruction_sh['condition'] = Condition
    #
    end_shuff = time.time()
    process_shuff = end_shuff - start_shuff
    print( 'Time shuff: ' +str(process_shuff))
    #
    return Reconstruction, Reconstruction_sh




##########################
##########################
###########################

# Subject='b001'
# Brain_Region='visual'
# Condition='1_7'
# iterations=4
# distance='close'
# decode_item='Target'
# method='together'
# heatmap=False

