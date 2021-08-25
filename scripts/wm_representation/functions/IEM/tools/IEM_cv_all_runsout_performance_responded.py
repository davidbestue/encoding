# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:05:33 2019

@author: David
"""
import sys, os
path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 
sys.path.insert(1, path_tools)
from tools import *


def IEM_cv_all_runsout_performance_responded(Error, Error_percent, testing_activity, testing_behaviour, decode_item, training_item, tr_st, tr_end):
    ####
    ####
    #### IEM: Inverted encoding model
    #### cv: cross-validated. Trained and test on the same dataset
    #### all: cross-validate all the TRs and not just the shared for training
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
    #### Get the Trs with shared information and the TRs without shared information
    list_wm_scans= range(nscans_wm)  
    trs_shared = range(tr_st, tr_end)
    nope=[list_wm_scans.remove(tr_s) for tr_s in trs_shared]
    list_wm_scans2 = list_wm_scans
    ####
    ####
    testing_behaviour = testing_behaviour.reset_index()
    testing_angles = np.array(testing_behaviour[decode_item])  
    training_angles = np.array(testing_behaviour[training_item]) 
    ###########################################################################
    ########################################################################### Get the mutliple indexes to split in train and test
    ###########################################################################
    training_indexes = []
    testing_indexes =  []
    for sess_run in testing_behaviour.session_run.unique():
        wanted = testing_behaviour.loc[testing_behaviour['session_run']==sess_run].index.values 
        testing_indexes.append( wanted )
        #
        #unwanted = list(wanted)
        #all_indexes = list(testing_behaviour.index.values) 
        #for ele in sorted(unwanted, reverse = True): 
        #     del all_indexes[ele]
        #
        #training_indexes.append( np.array(all_indexes) )
        ##
        ## I do not trust the upper lines, maybe this del inside a function in paralel is not removing the indexes, also you avoid going to lists to comeback
        all_indexes = testing_behaviour.index.values
        other_indexes = all_indexes[~np.array([all_indexes[i] in wanted for i in range(len(all_indexes))])]  #take the ones that are not in wanted
        training_indexes.append( other_indexes )
    ###########################################################################
    ########################################################################### Run the ones without shared info
    ###########################################################################  
    #####
    Recons_dfs_not_shared=[]
    for not_shared in list_wm_scans2:
        training_data =   np.mean(testing_activity[:, tr_st:tr_end, :], axis=1) ## son los mismos siempre, pero puede haber time dependence!
        testing_data= testing_activity[:, not_shared, :]   
        reconstrction_sh=[]
        ###
        ### apply the multiple indexes to train and test
        ###
        for train_index, test_index in zip(training_indexes, testing_indexes):
            X_train, X_test = training_data[train_index], testing_data[test_index]
            y_train, y_test = training_angles[train_index], testing_angles[test_index]
            ## train
            WM2, Inter2 = Weights_matrix_LM(X_train, y_train)
            WM_t2 = WM2.transpose()
            ## test
            ########
            ##################
            ################## Split by perfromance 
            ################## We still want to try in all the data (same training). Just change the testing!
            ################## forget about previous X_test and y_test, from the remaining RUN, select by perfromance (not from all together)
            ################## 
            testing_beh = testing_behaviour.iloc[test_index, :]
            ###### Get the responded ones
            if decode_item == 'T_alone':
                testing_data = testing_data[testing_beh[decode_item] == testing_beh['T'] ] 
                testing_beh = testing_beh[testing_beh[decode_item] == testing_beh['T'] ]                 
            if decode_item == 'dist_alone':
                testing_data = testing_data[testing_beh[decode_item] == testing_beh['Dist'] ] 
                testing_beh = testing_beh[testing_beh[decode_item] == testing_beh['Dist'] ]                 
            ###
            if Error == 'high':
                abs_err_bool = abs(testing_beh.A_err) > np.percentile(abs(testing_beh.A_err), Error_percent, interpolation = 'midpoint')
                if sum(abs_err_bool)==0:
                    abs_err_bool = abs(testing_beh.A_err)==max(abs(testing_beh.A_err))

            elif Error == 'low':
                abs_err_bool = abs(testing_beh.A_err) < np.percentile(abs(testing_beh.A_err), Error_percent, interpolation = 'midpoint')
                if sum(abs_err_bool)==0:
                    abs_err_bool = abs(testing_beh.A_err)==min(abs(testing_beh.A_err))
            ##
            y_test = np.array(testing_beh[abs_err_bool][decode_item])
            X_test = testing_data[test_index][abs_err_bool]
            #
            #
            rep_x = Representation(testing_data=X_test, testing_angles=y_test, Weights=WM2, Weights_t=WM_t2, ref_angle=180, plot=False, intercept=Inter2)
            reconstrction_sh.append(rep_x)
        ###
        reconstrction_sh = pd.concat(reconstrction_sh, axis=1) ##una al lado de la otra, de lo mismo, ahora un mean manteniendo indice
        reconstrction_sh_mean = reconstrction_sh.mean(axis = 1) #solo queda una columna con el mean de cada channel 
        Recons_dfs_not_shared.append(reconstrction_sh_mean)
    ####
    Reconstruction_not_shared = pd.concat(Recons_dfs_not_shared, axis=1)
    Reconstruction_not_shared.columns =  [str(i * TR) for i in list_wm_scans2 ] 
    ####
    ###
    ###########################################################################
    ########################################################################### Run the ones with shared info
    ###########################################################################
    Recons_dfs_shared=[]
    for shared_TR in trs_shared:
        testing_data= testing_activity[:, shared_TR, :]            
        reconstrction_sh=[]
        for train_index, test_index in zip(training_indexes, testing_indexes):
            X_train, X_test = testing_data[train_index], testing_data[test_index]
            y_train, y_test = training_angles[train_index], testing_angles[test_index]
            ## train
            WM2, Inter2 = Weights_matrix_LM(X_train, y_train)
            WM_t2 = WM2.transpose()
            ## test
            ########
            ##################
            ################## Split by perfromance 
            ################## We still want to try in all the data (same training). Just change the testing!
            ################## forget about previous X_test and y_test, from the remaining RUN, select by perfromance (not from all together)
            ################## 
            testing_beh = testing_behaviour.iloc[test_index, :]
            ###### Get the responded ones
            if decode_item == 'T_alone':
                testing_data = testing_data[testing_beh[decode_item] == testing_beh['T'] ] 
                testing_beh = testing_beh[testing_beh[decode_item] == testing_beh['T'] ]                 
            if decode_item == 'dist_alone':
                testing_data = testing_data[testing_beh[decode_item] == testing_beh['Dist'] ] 
                testing_beh = testing_beh[testing_beh[decode_item] == testing_beh['Dist'] ]                 
            ###
            if Error == 'high':
                abs_err_bool = abs(testing_beh.A_err) > np.percentile(abs(testing_beh.A_err), Error_percent, interpolation = 'midpoint')
                if sum(abs_err_bool)==0:
                    abs_err_bool = abs(testing_beh.A_err)==max(abs(testing_beh.A_err))
            #
            elif Error == 'low':
                abs_err_bool = abs(testing_beh.A_err) < np.percentile(abs(testing_beh.A_err), Error_percent, interpolation = 'midpoint')
                if sum(abs_err_bool)==0:
                    abs_err_bool = abs(testing_beh.A_err)==min(abs(testing_beh.A_err))
            ##
            y_test = np.array(testing_beh[abs_err_bool][decode_item])
            X_test = testing_data[test_index][abs_err_bool]
            #
            #
            rep_x = Representation(testing_data=X_test, testing_angles=y_test, Weights=WM2, Weights_t=WM_t2, ref_angle=180, plot=False, intercept=Inter2)
            reconstrction_sh.append(rep_x)
        ###
        reconstrction_sh = pd.concat(reconstrction_sh, axis=1) ##una al lado de la otra, de lo mismo, ahora un mean manteniendo indice
        reconstrction_sh_mean = reconstrction_sh.mean(axis = 1) #solo queda una columna con el mean de cada channel 
        Recons_dfs_shared.append(reconstrction_sh_mean)
    ####
    ####
    ####
    #### Put together
    Reconstruction_shared = pd.concat(Recons_dfs_shared, axis=1)
    Reconstruction_shared.columns =  [str(i * TR) for i in trs_shared ]  
    #### 
    #### Merge both recosntructions dfs to get a single one
    Reconstruction = pd.concat([Reconstruction_not_shared, Reconstruction_shared], axis=1)
    ### sort the columns so the indep does not get at the end
    sorted_col = np.sort([float(Reconstruction.columns[i]) for i in range(len(Reconstruction.columns))])           
    sorted_col = [str(sorted_col[i]) for i in range(len(sorted_col))]
    Reconstruction = Reconstruction.reindex( sorted_col, axis=1)  
    #
    return Reconstruction


