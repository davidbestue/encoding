# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:05:33 2019

@author: David
"""

import sys, os
path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 
sys.path.insert(1, path_tools)
from tools import *


def Representation_angle_runsout(training_activity, training_behaviour, testing_activity, testing_behaviour, training_item, tr_st, tr_end, df_shuffle):
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
    list_wm_scans2 = list_wm_scans
    ####
    ####
    ####
    #### Run the ones WITHOUT shared information the same way
    #testing_behaviour = testing_behaviour.reset_index()
    #training_behaviour = training_behaviour.reset_index()
    training_angles = np.array(training_behaviour[training_item])   
    #####
    Recons_trs=[]
    for not_shared in list_wm_scans2:
        training_data =   np.mean(training_activity[:, tr_st:tr_end, :], axis=1) ## son los mismos siempre, pero puede haber time dependence!
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
            ## I do not trust the del  lines of other files, maybe this del inside a function in paralel is not removing the indexes, also you avoid going to lists to comeback
            all_indexes = testing_behaviour.index.values
            other_indexes = all_indexes[~np.array([all_indexes[i] in wanted for i in range(len(all_indexes))])]  #take the ones that are not in wanted
            training_indexes.append( other_indexes ) 
        ###
        ### apply them to train and test
        ###
        for train_index, test_index in zip(training_indexes, testing_indexes):
            X_train, X_test = training_data[train_index], testing_data[test_index]
            y_train = training_angles[train_index]
            ## train
            WM2, Inter2 = Weights_matrix_LM(X_train, y_train)
            WM_t2 = WM2.transpose()
            ## test
            trials_test = testing_behaviour.iloc[test_index, :]
            rep_x = decoding_angles_pvector( testing_behaviour=trials_test, testing_data=X_test, df_shuffle=df_shuffle, 
                specific_tr_shuffle=not_shared, Weights=WM2, Weights_t=WM_t2, ref_angle=180, intercept=Inter2)
            rep_x['TR_'] = not_shared
            reconstrction_.append(rep_x)
        ###
        reconstrction_x = pd.concat(reconstrction_) #
        #####
        ##### Ahora tienes en por cada trial, tantos decoders como sessiones. Hacer un mean de eso. De tal manera que de cada new index solo queden 3 valores (T, NT1, NT2)
        #####
        for Idx in reconstrction_x.new_index.unique():
            for Dec_item in ['T', 'NT1', 'NT2']:
                for Tr_ in reconstrction_x.TR_.unique():                    
                    df_x = reconstrction_x.loc[(reconstrction_x['new_index']==Idx) &  (reconstrction_x['label_target']==Dec_item)  &  (reconstrction_x['TR_']==Tr_)]
                    decoded_angle_ = df_x.decoded_angle.mean() ###this ignores the Nans. It is the same as np.nanmean(df_x.decoded_angle.values) 
                    target_centered_ = df_x.target_centered.iloc[0]
                    label_target_ = df_x.label_target.iloc[0]
                    corresp_isolated_ = df_x.corresp_isolated.iloc[0]
                    distractor_centered_ = df_x.distractor_centered.iloc[0]
                    corresp_isolated_distractor_ = df_x.corresp_isolated_distractor.iloc[0]
                    label_distractor_ = df_x.label_distractor.iloc[0]
                    new_index_ = df_x.new_index.iloc[0]
                    TR_x = str(df_x.TR_.iloc[0] * TR)
                    #
                    Recons_trs.append([decoded_angle_, target_centered_, label_target_, corresp_isolated_, distractor_centered_, corresp_isolated_distractor_, 
                                                 label_distractor_, new_index_, TR_x])
            #
        #
    #
    ####
    Reconstruction = pd.DataFrame(Recons_trs)
    Reconstruction.columns =  ['decoded_angle', 'target_centered', 'label_target', 'corresp_isolated', 'distractor_centered', 'corresp_isolated_distractor', 
                                'label_distractor', 'new_index', 'TR']
    #
    return Reconstruction






