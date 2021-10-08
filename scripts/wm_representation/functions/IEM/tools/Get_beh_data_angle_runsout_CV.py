# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:05:33 2019

@author: David
"""

import sys, os
path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 
sys.path.insert(1, path_tools)
from tools import *




def Get_beh_data_angle_runsout_CV(testing_behaviour, training_item,  tr_st, tr_end):
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
    ##
    training_indexes = []
    testing_indexes =  []
    for sess_run in testing_behaviour.session_run.unique():
        wanted = testing_behaviour.loc[testing_behaviour['session_run']==sess_run].index.values 
        testing_indexes.append( wanted )
        #
        ## I do not trust the del  lines of other files, maybe this del inside a function in paralel is not removing the indexes, also you avoid going to lists to comeback
        all_indexes = testing_behaviour.index.values
        other_indexes = all_indexes[~np.array([all_indexes[i] in wanted for i in range(len(all_indexes))])]  #take the ones that are not in wanted
        training_indexes.append( other_indexes ) 
    #####
    #####
    #####
    ##### Not shared TRs (mean delay)
    #####
    #####
    Recons_trs=[]
    for not_shared in list_wm_scans2:
        ###########################################################################
        ########################################################################### Get the mutliple indexes to split in train and test
        ###########################################################################
        ###
        ### apply them to train and test
        ###
        for train_index, test_index in zip(training_indexes, testing_indexes):
            trials_test = testing_behaviour.iloc[test_index, :]
            Recons_trs.append(trials_test)
    ###
    Recons_trs = pd.concat(reconstrction_) #
    ####
    ####
    ####
    Recons_trs_shared=[]
    for shared_TR in trs_shared:
        testing_data= testing_activity[:, shared_TR, :] 
        reconstrction_=[]
        ###########################################################################
        ########################################################################### Get the mutliple indexes to split in train and test
        ###########################################################################
        ###
        ### apply them to train and test
        ###
        for train_index, test_index in zip(training_indexes, testing_indexes):
            trials_test = testing_behaviour.iloc[test_index, :]
            Recons_trs_shared.append(trials_test)
        ###
    Recons_trs_shared = pd.concat(reconstrction_) #
    #####
    #####
    Reconstruction = pd.concat([Recons_trs_shared, Recons_trs], axis=0)
    #
    return Reconstruction








