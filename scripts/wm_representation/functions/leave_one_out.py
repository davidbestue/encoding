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


numcores = multiprocessing.cpu_count() #- 10


def err_deg(a1,ref):
    ### Calculate the error ref-a1 in an efficient way in the circular space
    ### it uses complex numbers!
    ### Input in degrees (0-360)
    a1=np.radians(a1)
    ref=np.radians(ref)
    err = np.angle(np.exp(1j*ref)/np.exp(1j*(a1) ), deg=True) 
    err=round(err, 2)
    return err


def model_PV(X_train, X_test, y_train, y_test):
    ##
    ######## Trainning #########
    ## X matrix (intercept and spikes)
    X = np.column_stack([np.ones(np.shape(X_train)[0]), X_train])
    ## Y (sinus and cos)
    sinus =np.sin([np.radians(np.array(y_train)[i]) for i in range(0, len(y_train))])
    cosinus = np.cos([np.radians(np.array(y_train)[i]) for i in range(0, len(y_train))])
    Y = np.column_stack([cosinus, sinus])
    ### one OLS for sin and cos: output: beta of intercetp and bea of spikes (two B intercepts and 2 B for spikes )
    Y = Y.astype(float) #to make it work in the cluster
    X = X.astype(float)
    model = sm.OLS(Y, X)
    ##train the model
    fit=model.fit()
    ######### Testing the remaining one ###########
    X_ = np.column_stack([np.ones(np.shape(X_test)[0]), X_test])
    p = fit.predict(X_)
    x = p[0][0] ## estan dentro de una list 
    y = p[0][1]
    #####
    ##### Error --> take the resulting vector in sin/cos space
    ### from sin and cos get the angle (-pi, pi)
    pred_angle = np.degrees(np.arctan2(y, x)) 
    if pred_angle<0:
            pred_angle=360+pred_angle
    ##
    #print(pred_angle, y_test)
    error_ = err_deg(pred_angle, y_test)
    ##
    return error_




def Pop_vect_leave_one_out(testing_data, testing_angles):
    ## A esta función entrarán los datos de un TR. 
    ## Como se ha de hacer el leave one out para estimar el error, no puedo paralelizar por trials
    ## Separar en train and test para leave on out procedure
    ## Hago esto para tener la mejor estimación posible del error (no hay training task)
    ## Si hubiese training task (aquí no la uso), no sería necesario el leave one out
    loo = LeaveOneOut()
    errors_=[]
    for train_index, test_index in loo.split(testing_data):
        X_train, X_test = testing_data[train_index], testing_data[test_index]
        y_train, y_test = testing_angles[train_index], testing_angles[test_index]
        ##
        ## correr el modelo en cada uno de los sets y guardar el error en cada uno de los trials
        ## la std no la hare con estos errores, sinó con el shuffle. No necesito guardar el error en cada repetición.
        model_trained_err = model_PV(X_train, X_test, y_train, y_test)
        errors_.append(model_trained_err)
    ##
    errors_abs = np.mean([abs(errors_[i]) for i in range(0, len(errors_))]) 
    error = np.mean(errors_abs) ##you need the absolute error to get an estimate of the decoding
    return error




def shuff_Pop_vect_leave_one_out(testing_data, testing_angles, iterations):
    ## A esta función entrarán los datos de un TR y haré el shuffleing. 
    ## Es como Pop_vect_leave_one_out pero en vez de dar un solo error para un scan, 
    ## de tantas iterations shuffled (contiene un loop for y un shuffle )
    loo = LeaveOneOut()
    errors_shuffle=[]
    #########
    ########
    for i in range(iterations):
        #
        testing_angles_sh = np.array(random.sample(testing_angles, len(testing_angles)) )
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




def leave_one_out_shuff( Subject, Brain_Region, Condition, iterations, distance, decode_item, method='together', heatmap=False):
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
    angles_paralel= [testing_angles_beh for i in range(nscans_wm)]
    signal_paralel =[ testing_activity[:, i, :] for i in range(nscans_wm)] #separate for nscans (to run in parallel)
    ### Error in each TR done with leave one out
    error_TR = Parallel(n_jobs = numcores)(delayed(Pop_vect_leave_one_out)(testing_data = signal, testing_angles= angles)  for signal, angles in zip(signal_paralel, angles_paralel))    #### reconstruction standard (paralel)
    ### save in the right format for the plots
    Reconstruction = pd.DataFrame(error_TR) #mean error en each TR (1 fila con n_scans columnas)
    Reconstruction['time']=[i * TR for i in range(nscans_wm)]
    Reconstruction.columns=['decoding', 'time']  
    Reconstruction['region'] = Brain_Region
    Reconstruction['subject'] = Subject
    Reconstruction['condition'] = Condition
    #Reconstruction = Reconstruction.transpose()
    #Reconstruction.columns =  [str(i * TR) for i in range(nscans_wm)]    ##column names
    
    ######
    end_l1out = time.time()
    process_l1out = end_l1out - start_l1out
    print( 'Time process leave one out: ' +str(process_l1out)) #print time of the process
    ####### Shuff
    #### Compute the shuffleing (n_iterations defined on top)
    start_shuff = time.time()
    itera_paralel=[iterations for i in range(nscans_wm)]
    shuffled_rec = Parallel(n_jobs = numcores)(delayed(shuff_Pop_vect_leave_one_out)(testing_data=signal_s, testing_angles=angles_s, iterations=itera) for signal_s, angles_s, itera in zip(signal_paralel, angles_paralel, itera_paralel))
    end_shuff = time.time()
    process_shuff = end_shuff - start_shuff
    print( 'Time shuff: ' +str(process_shuff))
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

    return Reconstruction, Reconstruction_sh



