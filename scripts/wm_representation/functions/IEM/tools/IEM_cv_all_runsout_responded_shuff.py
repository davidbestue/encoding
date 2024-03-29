# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:05:33 2019

@author: David
"""

import sys, os
path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 
sys.path.insert(1, path_tools)
from tools import *


def IEM_cv_all_runsout_responded_shuff(testing_activity, testing_behaviour, decode_item, training_item, tr_st, tr_end, 
    condition, subject, region, iterations, ref_angle=180):
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
    ####
    #### Run the ones WITHOUT shared information the same way (train the whole tr_st:tr:end)
    testing_behaviour = testing_behaviour.reset_index()
    training_angles = np.array(testing_behaviour[training_item])   
    testing_angles = np.array(testing_behaviour[decode_item]) 
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
        ## I do not trust the upper lines, maybe this del inside a function in paralel is not removing the indexes, also you avoid going to lists to comeback
        all_indexes = testing_behaviour.index.values
        other_indexes = all_indexes[~np.array([all_indexes[i] in wanted for i in range(len(all_indexes))])]  #take the ones that are not in wanted
        training_indexes.append( other_indexes )
        #
    ##
    Reconstructions_shuffled=[]
    for It in range(iterations):
        #####
        Recons_dfs_not_shared=[]
        for not_shared in list_wm_scans2:
            training_data =   np.mean(testing_activity[:, tr_st:tr_end, :], axis=1) ## son los mismos siempre, pero puede haber time dependence!
            testing_data= testing_activity[:, not_shared, :]   
            reconstrction_sh=[]
            #kf = KFold(shuffle=True, n_splits=n_slpits);
            #kf.get_n_splits(testing_data);
            ###
            for train_index, test_index in zip(training_indexes, testing_indexes):
                X_train, X_test = training_data[train_index], testing_data[test_index]
                y_train, y_test = training_angles[train_index], testing_angles[test_index]
                ## train
                WM2, Inter2 = Weights_matrix_LM(X_train, y_train)
                WM_t2 = WM2.transpose()
                ##
                testing_beh = testing_behaviour.iloc[test_index, :] 
                ###### Get the responded ones
                if decode_item == 'T_alone':
                    testing_data2 = testing_data[test_index][testing_beh[decode_item] == testing_beh['T'] ] 
                    testing_beh = testing_beh[testing_beh[decode_item] == testing_beh['T'] ]                     
                if decode_item == 'dist_alone':
                    testing_data2 = testing_data[test_index][testing_beh[decode_item] == testing_beh['Dist'] ] 
                    testing_beh = testing_beh[testing_beh[decode_item] == testing_beh['Dist'] ]                     
                ###
                if len(testing_beh)>0:
                    y_test = np.array(testing_beh[:][decode_item])
                    y_test = np.array([random.choice([0, 90, 180, 270]) for i in range(len(y_test))])
                    X_test = testing_data2                 
                    ## test
                    rep_x = Representation(testing_data=X_test, testing_angles=y_test, Weights=WM2, Weights_t=WM_t2, ref_angle=180, plot=False, intercept=Inter2)
                    reconstrction_sh.append(rep_x)
                else:
                    print('Missing matching criteria trials in some runs')
            ###
            reconstrction_sh = pd.concat(reconstrction_sh, axis=1) ##una al lado de la otra, de lo mismo, ahora un mean manteniendo indice
            reconstrction_sh_mean = reconstrction_sh.mean(axis = 1) #solo queda una columna con el mean de cada channel 
            Recons_dfs_not_shared.append(reconstrction_sh_mean)
        ####
        Reconstruction_not_shared = pd.concat(Recons_dfs_not_shared, axis=1)
        Reconstruction_not_shared.columns =  [str(i * TR) for i in list_wm_scans2 ] 
        ####
        ####
        ####
        #### Run the ones WITH shared information: train and test in each TR of the tr_st:tr_end
        Recons_dfs_shared=[]
        for shared_TR in trs_shared:
            testing_data= testing_activity[:, shared_TR, :]            
            reconstrction_sh=[]
            #kf = KFold(shuffle=True, n_splits=n_slpits);
            #kf.get_n_splits(testing_data);
            #
            for train_index, test_index in zip(training_indexes, testing_indexes):
                X_train, X_test = testing_data[train_index], testing_data[test_index]
                y_train, y_test = training_angles[train_index], testing_angles[test_index]
                ## train
                WM2, Inter2 = Weights_matrix_LM(X_train, y_train)
                WM_t2 = WM2.transpose()
                ##
                testing_beh = testing_behaviour.iloc[test_index, :]
                ###### Get the responded ones
                if decode_item == 'T_alone':
                    testing_data2 = testing_data[test_index][testing_beh[decode_item] == testing_beh['T'] ] 
                    testing_beh = testing_beh[testing_beh[decode_item] == testing_beh['T'] ]                     
                if decode_item == 'dist_alone':
                    testing_data2 = testing_data[test_index][testing_beh[decode_item] == testing_beh['Dist'] ] 
                    testing_beh = testing_beh[testing_beh[decode_item] == testing_beh['Dist'] ]                     
                ### 
                if len(testing_beh)>0:
                    y_test = np.array(testing_beh[:][decode_item])
                    y_test = np.array([random.choice([0, 90, 180, 270]) for i in range(len(y_test))])
                    X_test = testing_data2                 
                    ## test
                    rep_x = Representation(testing_data=X_test, testing_angles=y_test, Weights=WM2, Weights_t=WM_t2, ref_angle=180, plot=False, intercept=Inter2)
                    reconstrction_sh.append(rep_x)
                else:
                    print('Missing matching criteria trials in some runs')
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
        ####
        ####
        Reconstructions_shuffled.append(Reconstruction)

    #####
    df_shuffle=[]
    for i in range(len(Reconstructions_shuffled)):
        n = Reconstructions_shuffled[i].iloc[ref_angle*2, :] #around the ref_angle (x2 beacuse now we have 720 instead of 360)
        n = n.reset_index()
        n.columns = ['times', 'decoding']
        n['decoding'] = [sum(Reconstructions_shuffled[i].iloc[:, ts] * f2(ref_angle)) for ts in range(len(n))] #population vector method (scalar product)
        n['times']=n['times'].astype(float)
        n['region'] = region
        n['subject'] = subject
        n['condition'] = condition
        df_shuffle.append(n) #save thhis
    
    ##
    df_shuffle = pd.concat(df_shuffle)    #same shape as the decosing of the signal
    return df_shuffle
    


