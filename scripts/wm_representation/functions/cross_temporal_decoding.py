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



def shuff_cross_temporal3( activity, test_beh, test_octaves, iterations):
    ## A esta función entrarán los datos de un TR y haré el shuffleing. 
    ## Es como Pop_vect_leave_one_out pero en vez de dar un solo error para un scan, 
    ## de tantas iterations shuffled (contiene un loop for y un shuffle )
    ## Alternativa: En vez de hacer n_iterations, hacer el shuffleing una vez y hacer una media de todos los errores
    ## Pro alternativa: menos tiempo de computacion
    ## Contra: mas variabilidad (barras de error menos robustas)
    loo = LeaveOneOut()
    dfs_shuffle=[]
    #########
    ########
    octaves_paralel= [test_octaves for i in range(nscans_wm)] ##octaves_angles_beh
    for i in range(iterations):
        # aquí estoy haciendo un shuffle forzando que acabe en una octava en la que no haya nada
        miss_octvs_trials = [get_octvs_missing(test_beh['T'].iloc[i], test_beh['NT1'].iloc[i], test_beh['NT2'].iloc[i], 
            test_beh['Dist'].iloc[i], test_beh['Dist_NT1'].iloc[i], test_beh['Dist_NT2'].iloc[i]) for i in range(len(test_beh))]
        testing_octaves_sh = np.array( [random.choice(miss_octvs_trials[i]) for i in range(len(test_beh))])
        training_octa_sh_p = [ testing_octaves_sh for i in range(nscans_wm)]
        ## el testing és el correcto, solo cambias el training
        ##
        signal_paralel_testing =[ activity[:, i, :] for i in range(nscans_wm)] 
        accs_cross_temporal=[]
        for n_training in range(nscans_wm): ##train in each TR and test in the rest
            signal_paralel_training =[ activity[:, n_training, :] for i in range(nscans_wm)]
            acc_cross = Parallel(n_jobs = numcores)(delayed(model_SVM)(X_train=X_tr, X_test=X_tst, y_train=y_tr, y_test=y_tst)  for X_tr, X_tst, y_tr, y_tst in zip(signal_paralel_training, signal_paralel_testing, training_octa_sh_p, octaves_paralel))    #### reconstruction standard (paralel)
            accs_cross_temporal.append(acc_cross)
        ### 
        df_cross_temporal = pd.DataFrame(accs_cross_temporal) #each row is training, column is testing!
        dfs_shuffle.append(df_cross_temporal)
        ##
    return dfs_shuffle




def cross_tempo_SVM_shuff( Subject, Brain_Region, Condition, iterations, distance, decode_item, method='together', heatmap=False):
    enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, method, Brain_Region)
    ##### Process testing data
    activity, testing_behaviour = preprocess_wm_files(wm_fmri_paths, masks, wm_beh_paths, condition=Condition, distance=distance, sys_use='unix', nscans_wm=nscans_wm, TR=2.335)
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
    signal_paralel_testing =[ activity[:, i, :] for i in range(nscans_wm)] 
    ##
    accs_cross_temporal=[]
    for n_training in range(nscans_wm): ##train in each TR and test in the rest
        signal_paralel_training =[ activity[:, n_training, :] for i in range(nscans_wm)]
        acc_cross = Parallel(n_jobs = numcores)(delayed(model_SVM)(X_train=X_tr, X_test=X_tst, y_train=y_tr, y_test=y_tst)  for X_tr, X_tst, y_tr, y_tst in zip(signal_paralel_training, signal_paralel_testing, octaves_paralel, octaves_paralel))    #### reconstruction standard (paralel)
        accs_cross_temporal.append(acc_cross)
    ### 
    df_cross_temporal = pd.DataFrame(accs_cross_temporal) #each row is training, column is testing!
    ###
    end_l1out = time.time()
    process_l1out = end_l1out - start_l1out
    print( 'Cross-decoging signal: ' +str(process_l1out)) #print time of the process
    ####### Shuff
    start_shuff = time.time()
    itera_paralel=[iterations for i in range(nscans_wm)]
    ##
    ##
    dfs_shuffle = shuff_cross_temporal3(activity= activity, test_beh=testing_behaviour, test_octaves=octaves_angles_beh, iterations=iterations)
    ##
    ##
    end_shuff = time.time()
    process_shuff = end_shuff - start_shuff
    print( 'Time shuff: ' +str(process_shuff))
    
    return df_cross_temporal, dfs_shuffle




########################
######################## by condition
########################
########################

def shuff_cross_temporal3_condition( activity, test_octaves, iterations, signal_paralel_training, training_behaviour):
    ## A esta función entrarán los datos de un TR y haré el shuffleing. 
    ## Es como Pop_vect_leave_one_out pero en vez de dar un solo error para un scan, 
    ## de tantas iterations shuffled (contiene un loop for y un shuffle )
    ## Alternativa: En vez de hacer n_iterations, hacer el shuffleing una vez y hacer una media de todos los errores
    ## Pro alternativa: menos tiempo de computacion
    ## Contra: mas variabilidad (barras de error menos robustas)
    loo = LeaveOneOut()
    dfs_shuffle=[]
    #########
    ########
    octaves_paralel= [test_octaves for i in range(nscans_wm)] ##octaves_angles_beh
    for i in range(iterations):
        # aquí estoy haciendo un shuffle forzando que acabe en una octava en la que no haya nada
        miss_octvs_trials = [get_octvs_missing(training_behaviour['T'].iloc[i], training_behaviour['NT1'].iloc[i], training_behaviour['NT2'].iloc[i], 
            training_behaviour['Dist'].iloc[i], training_behaviour['Dist_NT1'].iloc[i], training_behaviour['Dist_NT2'].iloc[i]) for i in range(len(training_behaviour))]
        training_octaves_sh = np.array( [random.choice(miss_octvs_trials[i]) for i in range(len(training_behaviour))])
        training_octa_sh_p = [ training_octaves_sh for i in range(nscans_wm)]
        ## el testing és el correcto, solo cambias el training
        ##
        signal_paralel_testing =[ activity[:, i, :] for i in range(nscans_wm)] 
        #
        acc_cross = Parallel(n_jobs = numcores)(delayed(model_SVM)(X_train=X_tr, X_test=X_tst, y_train=y_tr, y_test=y_tst)  for X_tr, X_tst, y_tr, y_tst in zip(signal_paralel_training, signal_paralel_testing, training_octa_sh_p, octaves_paralel))    #### reconstruction standard (paralel)
        ### 
        df_cross_temporal = pd.DataFrame(acc_cross) #each row is training, column is testing!
        dfs_shuffle.append(df_cross_temporal)
        ##
    return dfs_shuffle



def cross_tempo_SVM_shuff_condition( Subject, Brain_Region, Condition, iterations, distance, decode_item, signal_paralel_training, training_behaviour, method='together', heatmap=False):
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
    signal_paralel_testing =[ testing_activity[:, i, :] for i in range(nscans_wm)] 
    ##
    training_angles_beh = np.array(training_behaviour[dec_I]) 
    octaves_angles_beh_trian = np.array([get_octave(training_angles_beh[i]) for i in range(len(training_angles_beh))] )
    training_behaviour_paralel =[octaves_angles_beh_trian for i in range(nscans_wm)]
    ##
    acc_cross = Parallel(n_jobs = numcores)(delayed(model_SVM)(X_train=X_tr, X_test=X_tst, y_train=y_tr, y_test=y_tst)  for X_tr, X_tst, y_tr, y_tst in zip(signal_paralel_training, signal_paralel_testing, training_behaviour_paralel, octaves_paralel))    #### reconstruction standard (paralel)
    ### 
    df_cross_temporal = pd.DataFrame(acc_cross) #each row is training, column is testing!
    ###
    end_l1out = time.time()
    process_l1out = end_l1out - start_l1out
    print( 'Cross-decoging signal: ' +str(process_l1out)) #print time of the process
    ####### Shuff
    start_shuff = time.time()
    itera_paralel=[iterations for i in range(nscans_wm)]
    ##
    ##
    dfs_shuffle =shuff_cross_temporal3_condition( activity=testing_activity, test_octaves=octaves_angles_beh, iterations=iterations, signal_paralel_training=signal_paralel_training, training_behaviour=training_behaviour)
    ##
    ##
    end_shuff = time.time()
    process_shuff = end_shuff - start_shuff
    print( 'Time shuff: ' +str(process_shuff))
    
    return df_cross_temporal, dfs_shuffle



##########################

def SVM_l10(signal_, behaviour_):
        loo = LeaveOneOut()
        ctd_=[]
        for train_index, test_index in loo.split(signal_):
            X_tr, X_tst = signal_[train_index], signal_[test_index]
            y_tr, y_tst = behaviour_[train_index], behaviour_[test_index]
            ##
            ## correr el modelo en cada uno de los sets y guardar el error en cada uno de los trials
            ## la std no la hare con estos errores, sinó con el shuffle. No necesito guardar el error en cada repetición.
            model_trained_err = model_SVM(X_train=X_tr, X_test=X_tst, y_train=y_tr, y_test=y_tst)
            ctd_.append(model_trained_err) ## error de todos los train-test
            ##
        ##
        acc_cross = np.mean(ctd_) 
        return acc_cross
        



def SVM_l10_condition( Subject, Brain_Region, Condition, Condition_train, iterations, distance, decode_item, signal_paralel_training, training_behaviour, method='together', heatmap=False):
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
    ## Prep testing
    start_l1out = time.time()  
    testing_angles_beh = np.array(testing_behaviour[dec_I])    # A_R # T # Dist
    octaves_angles_beh = np.array([get_octave(testing_angles_beh[i]) for i in range(len(testing_angles_beh))] )
    octaves_paralel= [octaves_angles_beh for i in range(nscans_wm)]
    signal_paralel_testing =[ testing_activity[:, i, :] for i in range(nscans_wm)] 
    ##
    ##
    if Condition_train==Condition:
        #### In this case, leave one out procedure
        acc_cross = Parallel(n_jobs = numcores)(delayed(SVM_l10)(signal_=s_t, behaviour_=b_h)  for s_t, b_h in zip(signal_paralel_testing, octaves_paralel))  
        df_cross_temporal = pd.DataFrame(acc_cross)
    else:
        ## Prep training (just beh needs to be traines)
        training_angles_beh = np.array(training_behaviour[dec_I]) 
        octaves_angles_beh_trian = np.array([get_octave(training_angles_beh[i]) for i in range(len(training_angles_beh))] )
        training_behaviour_paralel =[octaves_angles_beh_trian for i in range(nscans_wm)]
        ## standard train and test
        acc_cross = Parallel(n_jobs = numcores)(delayed(model_SVM)(X_train=X_tr, X_test=X_tst, y_train=y_tr, y_test=y_tst)  for X_tr, X_tst, y_tr, y_tst in zip(signal_paralel_training, signal_paralel_testing, training_behaviour_paralel, octaves_paralel))    #### reconstruction standard (paralel)
        df_cross_temporal = pd.DataFrame(acc_cross) #each row is training, column is testing!
    ##
    ## time
    end_l1out = time.time()
    process_l1out = end_l1out - start_l1out
    print( 'Cross-decoging signal: ' +str(process_l1out)) #print time of the process
    ####### Shuff
    start_shuff = time.time()
    itera_paralel=[iterations for i in range(nscans_wm)]
    ##
    ##
    dfs_shuffle =shuff_cross_temporal3_condition( activity=testing_activity, test_octaves=octaves_angles_beh, iterations=iterations, signal_paralel_training=signal_paralel_training, training_behaviour=training_behaviour)
    ##
    ##
    end_shuff = time.time()
    process_shuff = end_shuff - start_shuff
    print( 'Time shuff: ' +str(process_shuff))
    
    return df_cross_temporal, dfs_shuffle




########################
######################## by quadrant
########################
########################

def get_quadrant(angle):
    #
    if angle>=0 and angle<=90:
        q=1
    elif angle>90 and angle<=180:
        q=2
    elif angle>180 and angle<=270:
        q=3
    elif angle>270 and angle<=360:
        q=4
    ###
    return q



def get_dec_I(decode_item):
    if decode_item == 'Target':
        dec_I = 'T'
    elif decode_item == 'Response':
        dec_I = 'A_R'
    elif decode_item == 'Distractor':
        dec_I = 'Dist'
    #
    return dec_I



def cross_temporal_quadrant( activity, behaviour, dec_I):
    ## A esta función entrarán los datos de un TR  
    ## Coger Octaves (al ser de un quadrante, sera 1 o 2) --> no haré shuffle
    octaves_ = np.array([get_octave(behaviour[dec_I].iloc[i]) for i in range(len(behaviour))])  
    ########
    octaves_paralel= [octaves_ for i in range(nscans_wm)] ##octaves_angles_beh
    signal_paralel_testing =[ activity[:, i, :] for i in range(nscans_wm)] 
    ########
    accs_cross_temporal=[]
    for n_training in range(nscans_wm): ##train in each TR and test in the rest
        signal_paralel_training =[ activity[:, n_training, :] for i in range(nscans_wm)]
        acc_cross = Parallel(n_jobs = numcores)(delayed(model_SVM)(X_train=X_tr, X_test=X_tst, y_train=y_tr, y_test=y_tst)  for X_tr, X_tst, y_tr, y_tst in zip(signal_paralel_training, signal_paralel_testing, octaves_paralel, octaves_paralel))    #### reconstruction standard (paralel)
        accs_cross_temporal.append(acc_cross)
        #
    #
    df_cross_temporal = pd.DataFrame(accs_cross_temporal) #each row is training, column is testing!
    return df_cross_temporal


############################

# Subject='b001'
# Brain_Region='visual'
# Condition='1_7'
# iterations=4
# distance='close'
# decode_item='Target'
# method='together'
# heatmap=False

