# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:05:33 2019

@author: David
"""

import sys, os
path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 
sys.path.insert(1, path_tools)
from tools import *






def Representation_cv_angle_runsout_shuff(testing_activity, testing_behaviour, training_item, tr_st, tr_end, iterations, ref_angle=180):
    ####
    ####
    #### IEM: Inverted encoding model
    #### runsout: train in all the runs except one
    #### all: do it for all the Trs
    #### 
    #### no cv: I use this for independent conditions (not training and testing in the same trials) 
    ####
    #### I use this function for the conditions 1_7 and 2_7, because the adjacent TRs may be "contaminated"
    #### Instead of "leave one out" or kfold, I am spliting by run!
    #### 
    #### Difference when runing the reconstruction between shared and not shared TRS with training
    #### Not shared: trained in the mean of the interval tr_st - tr_end
    #### Shared: trianed in each TR of the interval
    ####
    #### Training item (decide before where you train the model (it is a column in the beh file))
    ##### training_item = 'T_alone'
    ##### training_item = 'dist_alone'    
    ####
    #### Get the Trs (no shared info, coming from different trials)
    list_wm_scans= range(nscans_wm)  
    trs_shared = range(tr_st, tr_end)
    nope=[list_wm_scans.remove(tr_s) for tr_s in trs_shared]
    list_wm_scans2 = list_wm_scans
    ####
    ####
    ####
    #### Run the ones WITHOUT shared information the same way
    #testing_behaviour = testing_behaviour.reset_index()
    #training_behaviour = training_behaviour.reset_index()
    training_angles = np.array(testing_behaviour[training_item])   
    decode_item = 'T' ##irrelevant, it is going to be transformed to 0, 90.....
    testing_angles = np.array(testing_behaviour[decode_item])    
    #####
    Reconstructions_shuffled=[]
    for It in range(iterations):
        ###########
        ###########
        ########### not shared (mean activity in training)
        ###########
        ###########
        Recons_trs=[]
        for not_shared in list_wm_scans2:
            training_data =   np.mean(testing_activity[:, tr_st:tr_end, :], axis=1) ## son los mismos siempre, pero puede haber time dependence!
            testing_data= testing_activity[:, not_shared, :]   
            reconstrction_=[]
            ###########################################################################
            ########################################################################### Get the mutliple indexes to split in train and test
            ###########################################################################
            training_indexes = []
            testing_indexes =  []
            for sess_run in testing_behaviour.session_run.unique():
                wanted = testing_behaviour.loc[testing_behaviour['session_run']==sess_run].index.values 
                testing_indexes.append( wanted )
                #
                all_indexes = testing_behaviour.index.values
                other_indexes = all_indexes[~np.array([all_indexes[i] in wanted for i in range(len(all_indexes))])]  #take the ones that are not in wanted
                training_indexes.append( other_indexes ) 
            ###
            ### apply them to train and test
            ###
            for train_index, test_index in zip(training_indexes, testing_indexes):
                X_train, X_test = training_data[train_index], testing_data[test_index]
                y_train, y_test = training_angles[train_index], testing_angles[test_index]
                ## train
                WM2, Inter2 = Weights_matrix_LM(X_train, y_train)
                WM_t2 = WM2.transpose()
                ## Shuffle
                y_test = np.array([random.choice([0, 90, 180, 270]) for i in range(len(y_test))]) 
                ## test
                rep_x = decoding_angle_sh_pvector(testing_data=X_test, testing_angles=y_test, Weights=WM2, Weights_t=WM_t2, ref_angle=ref_angle, intercept=Inter2)
                reconstrction_.append(rep_x)
            ###
            reconstrction_ = pd.concat(reconstrction_, axis = 1) ##una al lado de la otra, de lo mismo, ahora un mean manteniendo indice
            reconstrction_mean = reconstrction_.mean(axis = 1) #solo queda una columna con el mean de cada channel 
            Recons_trs.append(reconstrction_mean)
        #
        ###########
        ###########
        ###########  shared TRS (not mean activity in training)
        ###########
        ###########
        Recons_trs_shared=[]
        for shared_TR in trs_shared:
            testing_data= testing_activity[:, shared_TR, :] 
            reconstrction_=[]
            ###########################################################################
            ########################################################################### Get the mutliple indexes to split in train and test
            ###########################################################################
            training_indexes = []
            testing_indexes =  []
            for sess_run in testing_behaviour.session_run.unique():
                wanted = testing_behaviour.loc[testing_behaviour['session_run']==sess_run].index.values 
                testing_indexes.append( wanted )
                #
                all_indexes = testing_behaviour.index.values
                other_indexes = all_indexes[~np.array([all_indexes[i] in wanted for i in range(len(all_indexes))])]  #take the ones that are not in wanted
                training_indexes.append( other_indexes ) 
            ###
            ### apply them to train and test
            ###
            for train_index, test_index in zip(training_indexes, testing_indexes):
                X_train, X_test = testing_data[train_index], testing_data[test_index]
                y_train, y_test = training_angles[train_index], testing_angles[test_index]
                ## train
                WM2, Inter2 = Weights_matrix_LM(X_train, y_train)
                WM_t2 = WM2.transpose()
                ## Shuffle
                y_test = np.array([random.choice([0, 90, 180, 270]) for i in range(len(y_test))]) 
                ## test
                rep_x = decoding_angle_sh_pvector(testing_data=X_test, testing_angles=y_test, Weights=WM2, Weights_t=WM_t2, ref_angle=ref_angle, intercept=Inter2)
                reconstrction_.append(rep_x)
            ###
            reconstrction_ = pd.concat(reconstrction_, axis = 1) ##una al lado de la otra, de lo mismo, ahora un mean manteniendo indice
            reconstrction_mean = reconstrction_.mean(axis = 1) #solo queda una columna con el mean de cada channel 
            Recons_trs_shared.append(reconstrction_mean)
        ####
        ####
        ####
        Reconstruction_not_shared = pd.concat(Recons_trs, axis=1)
        Reconstruction_not_shared.columns =  [str(i * TR) for i in list_wm_scans2 ] 
        #
        Reconstruction_shared = pd.concat(Recons_trs_shared, axis=1)
        Reconstruction_shared.columns =  [str(i * TR) for i in trs_shared ] 
        ##
        ##
        Reconstruction = pd.concat([Reconstruction_shared, Reconstruction_not_shared], axis=1)
        #
        Reconstructions_shuffled.append(Reconstruction)  
        #
    #
    #####
    df_shuffle =  pd.concat(Reconstructions_shuffled).mean(level=0)  ###dimensions (720, TRs) (mean shuffle!) (average of shuffled reconstructions)
    ##
    return df_shuffle














