# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:05:33 2019

@author: David
"""
import sys, os
path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 
sys.path.insert(1, path_tools)
from tools import *


def IEM_cross_condition_kfold(testing_activity, testing_behaviour, decode_item, WM, WM_t, Inter, 
    tr_st, tr_end, n_slpits=10):
    ####
    ####
    #### IEM usando data de WM test
    #### IEM de aquellos TRs donde se use tambien training data (condiciones 1_7 y 2_7)
    #### En vez de hacer leave one out, que tarda mucho, o usar el mismo data (overfitting), hago k_fold, con 10 splits. 
    ####
    ####
    if decode_item == 'Target':
        dec_I = 'T'
    elif decode_item == 'Response':
        dec_I = 'A_R'
    elif decode_item == 'Distractor':
        dec_I = 'Dist'
    else:
        'Error specifying the decode item'
    ####
    #### Get the Trs with shared information and the TRs without shared information
    list_wm_scans= range(nscans_wm)  
    trs_shared = range(tr_st, tr_end)
    nope=[list_wm_scans.remove(tr_s) for tr_s in trs_shared]
    list_wm_scans2 = list_wm_scans
    ####
    #### Run the ones without shared information the same way
    testing_angles = np.array(testing_behaviour[dec_I])    # A_R # T # Dist
    ### Respresentation
    signal_paralel =[ testing_activity[:, i, :] for i in list_wm_scans2 ]
    Reconstructions = Parallel(n_jobs = numcores)(delayed(Representation)(signal, testing_angles, WM, WM_t, ref_angle=180, plot=False, intercept=Inter)  for signal in signal_paralel)    #### reconstruction standard (paralel)
    Reconstruction_indep = pd.concat(Reconstructions, axis=1) #mean of the reconstructions (all trials)
    Reconstruction_indep.columns =  [str(i * TR) for i in list_wm_scans2 ]    ##column names
    ####
    #### Run the ones with shared information: k fold
    Recons_dfs_shared=[]
    for shared_TR in trs_shared:
        testing_data= testing_activity[:, shared_TR, :]            
        reconstrction_sh=[]
        kf = KFold(n_splits=n_slpits);
        kf.get_n_splits(testing_data);
        for train_index, test_index in kf.split(testing_data):
            X_train, X_test = testing_data[train_index], testing_data[test_index]
            y_train, y_test = testing_angles[train_index], testing_angles[test_index]
            ## train
            WM2, Inter2 = Weights_matrix_LM(X_train, y_train)
            WM_t2 = WM2.transpose()
            ## test
            rep_x = Representation(testing_data=X_test, testing_angles=y_test, Weights=WM2, Weights_t=WM_t2, ref_angle=180, plot=False, intercept=Inter2)
            reconstrction_sh.append(rep_x)
        ###
        reconstrction_sh = pd.concat(reconstrction_sh, axis=1) ##una al lado de la otra, de lo mismo, ahora un mean manteniendo indice
        reconstrction_sh_mean = reconstrction_sh.mean(axis = 1) #solo queda una columna con el mean de cada channel 
        Recons_dfs_shared.append(reconstrction_sh_mean)
    ####
    Reconstruction_shared = pd.concat(Recons_dfs_shared, axis=1)
    Reconstruction_shared.columns =  [str(i * TR) for i in trs_shared ]  
    #### 
    #### Merge both recosntructions dfs to get a single one
    Reconstruction = pd.concat([Reconstruction_indep, Reconstruction_shared], axis=1)
    ### sort the columns so the indep does not get at the end
    sorted_col = np.sort([float(Reconstruction.columns[i]) for i in range(len(Reconstruction.columns))])           
    sorted_col = [str(sorted_col[i]) for i in range(len(sorted_col))]
    Reconstruction = Reconstruction.reindex( sorted_col, axis=1)  
    #
    return Reconstruction


