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

import numpy as np
from sklearn.model_selection import LeaveOneOut


def model_PV(X_train, X_test, y_train, y_test):
    ##
    ######## Trainning #########
    ## X matrix (intercept and spikes)
    X = np.column_stack([np.ones(np.shape(X_train)[0]), X_train])
    ## Y (sinus and cos)
    sinus =np.sin([np.radians(np.array(y_train)[i]) for i in range(0, len(y_train))])
    cosinus = np.cos([np.radians(np.array(y_train)[i]) for i in range(0, len(y_rain))])
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
    x = p[0]
    y = p[1]
    #####
    ##### Error --> take the resulting vector in sin/cos space
    ### from sin and cos get the angle (-pi, pi)
    pred_angle = [ np.degrees(np.arctan2(y[i], x[i])) for i in range(0, len(y))]
    for i in range(0, len(pred_angle)):
        if pred_angle[i]<0:
            pred_angle[i]=360+pred_angle[i]
    ##
    #error=[ circdist(beh_test[i], pred_angle[i]) for i in range(0, len(pred_angle))]
    error=[ circdist(beh_test.values[i], pred_angle[i]) for i in range(0, len(pred_angle))]
    
    #low_value --> predicted positionns close to real
    neur_err.append(np.mean(error))


    return angle_decoded




def Pop_vect_leave_one_out(testing_data, testing_angles, ref_angle=180):
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
    Channel_all_trials_rolled = Parallel(n_jobs = numcores)(delayed(trial_rep)(Signal, angle_trial, Weights, Weights_t, ref=ref_angle, intercept_ = intercept)  for Signal, angle_trial in zip( data_prall, testing_angles))    ####
    #Channel_all_trials_rolled = Parallel(n_jobs = numcores)(delayed(trial_rep_decode_trial_by_trial)(Signal, angle_trial, Weights, Weights_t, ref=ref_angle, intercept_ = intercept)  for Signal, angle_trial in zip( data_prall, testing_angles))    ####
    Channel_all_trials_rolled = np.array(Channel_all_trials_rolled)
    
    df = pd.DataFrame()
    n = list(Channel_all_trials_rolled.mean(axis=0)) #mean of all the trials rolled
    df['TR'] = n #Name of the column
    if plot==True:
        #Plot heatmap
        plt.figure()
        plt.title('Heatmap decoding')
        ######midpoint = df.values.mean() # (df.values.max() - df.values.min()) / 2
        ax = sns.heatmap(df, yticklabels=list(df.index), cmap="coolwarm") # cmap= viridis "jet",  "coolwarm" RdBu_r, gnuplot, YlOrRd, CMRmap  , center = midpoint
        ax.plot([0.25, np.shape(df)[1]-0.25], [posch1_to_posch2(18),posch1_to_posch2(18)], 'k--')
        plt.yticks([posch1_to_posch2(4), posch1_to_posch2(13), posch1_to_posch2(22), posch1_to_posch2(31)] ,['45','135','225', '315'])
        plt.ylabel('Angle')
        plt.xlabel('time (s)')
        plt.show(block=False)
    
    return df




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
    testing_angles = np.array(testing_behaviour[dec_I])    # A_R # T # Dist
    ### Respresentation
    start_repres = time.time()    
    # TR separartion
    signal_paralel =[ testing_activity[:, i, :] for i in range(nscans_wm)] #separate for nscans (to run in parallel)
    ### Error in each TR done with leave one out


    Reconstructions = Parallel(n_jobs = numcores)(delayed(Representation)(signal, testing_angles, WM, WM_t, ref_angle=180, plot=False, intercept=Inter)  for signal in signal_paralel)    #### reconstruction standard (paralel)
    Reconstruction = pd.concat(Reconstructions, axis=1) #mean of the reconstructions (all trials)
    Reconstruction.columns =  [str(i * TR) for i in range(nscans_wm)]    ##column names
    #Plot heatmap
    if heatmap==True:
        plt.figure()
        plt.title(Condition)
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
    print( 'Time process reconstruction: ' +str(process_recons)) #print time of the process
    
    #df_boots = bootstrap_reconstruction(testing_activity, testing_angles, iterations, WM, WM_t, Inter, Brain_Region, Condition, Subject, ref_angle=180)    
    ####### Shuff
    #### Compute the shuffleing
    shuffled_rec = shuffled_reconstruction(signal_paralel, testing_angles, iterations, WM, WM_t, Inter=Inter, region=Brain_Region, condition=Condition, subject=Subject, ref_angle=180)
    
    return Reconstruction, shuffled_rec



