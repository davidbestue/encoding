# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:05:33 2019

@author: David
"""

import sys, os
path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 
sys.path.insert(1, path_tools)
from tools import *


#### Antes de hacer el decoding, deberíamos restarle el shuffle 

#### Lo suyo sería que de cada trial hacer 
####          - decode de T responded (guardar rotated NT1) (alguno será repetido)
####          - decode de T2 (guardar rotated NT2)
####          - decode de T3 (guardar rotated NT3)
####          - label de que target es el isolated
####          - order, delay 






####################################################################
####################################################################  Shuffle, get the mean (720, TR)
####################################################################
####################################################################
## Es distinto a lo de antes porque en los shuffle hacia todo el proceso de "pop vector multiplicando por la sum". Similar pero guardaba cosaas distintas
####################################################################




def decoding_angle_sh_pvector(testing_data, testing_angles, Weights, Weights_t, ref_angle=180, intercept=False):
    ## Make the data parallelizable
    n_trials_test = len(testing_data) #number trials
    data_prall = []
    for i in range(n_trials_test):
        data_prall.append(testing_data[i, :])
    ###
    ###
    numcores = multiprocessing.cpu_count()
    Channel_all_trials_rolled = Parallel(n_jobs = numcores)(delayed(trial_rep)(Signal, angle_trial, Weights, Weights_t, ref=ref_angle, intercept_ = intercept)  for Signal, angle_trial in zip( data_prall, testing_angles))    ####
    Channel_all_trials_rolled = np.array(Channel_all_trials_rolled)
    Channel_all_trials_rolled = pd.DataFrame(Channel_all_trials_rolled.mean(axis=0) ) 
    ### return the mean of all the trials aligned (I will use this for the shuffle). I would work for normal trials
    return Channel_all_trials_rolled
    





def Representation_angle_runsout_shuff(training_activity, training_behaviour, testing_activity, testing_behaviour, 
    decode_item, training_item, tr_st, tr_end, iterations, ref_angle=180):
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
    testing_angles = np.array(testing_behaviour[decode_item])    
    #####
    Reconstructions_shuffled=[]
    for It in range(iterations):
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
        ####
        Reconstruction = pd.concat(Recons_trs, axis=1)
        Reconstruction.columns =  [str(i * TR) for i in list_wm_scans2 ] 
        Reconstructions_shuffled.append(Reconstruction)
    ####
    #####
    df_shuffle =  pd.concat(Reconstructions_shuffled).mean(level=0)  ###dimensions (720, TRs) (mean shuffle!) (average of shuffled reconstructions)
    ##
    return df_shuffle















####################################################################
####################################################################  Decode the angle
####################################################################
















def decoding_angles_pvector( testing_behaviour, testing_data, df_shuffle, specific_tr_shuffle, Weights, Weights_t, ref_angle=180, intercept=False):
    ## Make the data parallelizable
    n_trials_test = len(testing_data) #number trials
    data_prall = []
    for i in range(n_trials_test):
        data_prall.append(testing_data[i, :])
    ###
    ###
    numcores = multiprocessing.cpu_count()
    ###
    ###
    distractor_labels = ['Dist',  'Dist_NT1',  'Dist_NT2']
    ###
    ###
    frames=[]
    #
    for idx_tar, Dec_item in enumerate(['T', 'NT1', 'NT2']):
        #
        testing_angles = np.array(testing_behaviour[Dec_item])   
        testing_distractors = np.array(testing_behaviour[distractor_labels[idx_tar]])
        #
        corresp_isol = list(np.array(testing_behaviour[Dec_item] == testing_behaviour['T_alone']) )
        corresp_isol_dist = list(np.array(testing_behaviour[distractor_labels[idx_tar]] == testing_behaviour['dist_alone']) )
        list_label_target = [Dec_item for i in range(len(testing_behaviour))]
        list_label_distr = [distractor_labels[idx_tar] for i in range(len(testing_behaviour))]
        new_indexes = list(testing_behaviour['new_index'].values)  
        #        
        Channel_all_trials_rolled = Parallel(n_jobs = numcores)(delayed(trial_rep)(Signal, angle_trial, Weights, Weights_t, ref=ref_angle, intercept_ = intercept)  for Signal, angle_trial in zip( data_prall, testing_angles))    ####
        Channel_all_trials_rolled = np.array(Channel_all_trials_rolled)
        n_trials_ = len(Channel_all_trials_rolled)
        ### 
        ### substract the shuffle and just get positive data (rest to 0) ##chequear que esto funcione...
        Channel_all_trials_substracted = Channel_all_trials_rolled - df_shuffle[str(specific_tr_shuffle*TR) ].values
        Channel_all_trials_substracted[Channel_all_trials_substracted<0]=0
        ###
        ###
        distance_to_ref =  testing_angles - ref_angle
        targ_180 = testing_angles - distance_to_ref
        ##
        ##
        decoded_angle=[]
        for i in range(n_trials_):
            Trial_reconstruction = Channel_all_trials_substracted[i,:]
            _135_ = ref_angle*2 - 45*2 
            _225_ = ref_angle*2 + 45*2 
            Trial_135_225 = Trial_reconstruction[_135_:_225_]
            N=len(Trial_135_225)
            R = []
            angles = np.radians(np.linspace(135,224,180) ) 
            R=np.dot(Trial_135_225,np.exp(1j*angles)) / N
            angle = np.angle(R)
            if angle < 0:
                angle +=2*np.pi 
            #
            angle_degrees = np.degrees(angle)
            if np.all(Trial_135_225==0)==True: ### if all the values are negative here, the decoded is np.nan (will not count as 0)
                angle_degrees=np.nan 
            #
            decoded_angle.append(angle_degrees)
        ###
        ###
        decoded_angle=np.array(decoded_angle)
        #### Rotation of distractor associated
        distractors_angles = np.array(testing_behaviour[distractor_labels[idx_tar]]) 
        #
        dist_180_ = testing_distractors - distance_to_ref
        dist_180 = []
        for n_dist in dist_180_:
            if n_dist<0:
                n_ = n_dist+360
                dist_180.append(n_)
            elif n_dist>360:
                n_ = n_dist-360
                dist_180.append(n_)
            else:
                dist_180.append(n_dist)
        ##
        dist_180 = np.array(dist_180)
        #
        df_dec = pd.DataFrame({'decoded_angle':decoded_angle, 'target_centered':targ_180, 
                                'label_target':list_label_target, 'corresp_isolated':corresp_isol,
                                'distractor_centered':dist_180, 'corresp_isolated_distractor':corresp_isol_dist,
                                'label_distractor':list_label_distr, 'new_index':new_indexes })
        #
        frames.append(df_dec)
    #
    df_dec = pd.concat(frames)
    #
    return df_dec
    







###################

def Representation_angle_runsout(training_activity, training_behaviour, testing_activity, testing_behaviour, decode_item, training_item, tr_st, tr_end, df_shuffle):
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
    testing_angles = np.array(testing_behaviour[decode_item])    
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
            y_train, y_test = training_angles[train_index], testing_angles[test_index]
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
        reconstrction_ = pd.concat(reconstrction_) #
        #####
        ##### Ahora tienes en por cada trial, tantos decoders como sessiones. Hacer un mean de eso. De tal manera que de cada new index solo queden 3 valores (T, NT1, NT2)
        #####
        for Idx in reconstrction_.new_index.unique():
            for Dec_item in ['T', 'NT1', 'NT2']:
                df_x = reconstrction_.loc[(reconstrction_['new_index']==Idx) &  (reconstrction_['label_target']==Idx)]
                decoded_angle = df_x.decoded_angle.mean() ###this ignores the Nans. It is the same as np.nanmean(df_x.decoded_angle.values) 
                target_centered = df_x.targ_180.iloc[0]
                label_target = df_x.label_target.iloc[0]
                corresp_isolated = df_x.corresp_isolated.iloc[0]
                distractor_centered = df_x.distractor_centered.iloc[0]
                corresp_isolated_distractor = df_x.corresp_isolated_distractor.iloc[0]
                label_distractor = df_x.label_distractor.iloc[0]
                new_index = df_x.new_index.iloc[0]
                TR_ = str(df_x.TR_.iloc[0] * TR)
                #
                Recons_trs.append([decoded_angle, target_centered, label_target, corresp_isolated, distractor_centered, corresp_isolated_distractor, 
                                             label_distractor, new_index, TR_])
            #
        #
    #
    ####
    Reconstruction = pd.DataFrame(Recons_trs)
    Reconstruction.columns =  ['decoded_angle', 'target_centered', 'label_target', 'corresp_isolated', 'distractor_centered', 'corresp_isolated_distractor', 
                                'label_distractor', 'new_index', 'TR']
    #
    return Reconstruction



















training_activity=training_activity
training_behaviour=training_behaviour
testing_activity=testing_activity
testing_behaviour=testing_behaviour
decode_item=decoding_thing
training_item=training_item
tr_st=tr_st
tr_end=tr_end
df_shuffle=shuff




list_wm_scans= range(nscans_wm)  
list_wm_scans2 = list_wm_scans
####
####
####
#### Run the ones WITHOUT shared information the same way
#testing_behaviour = testing_behaviour.reset_index()
#training_behaviour = training_behaviour.reset_index()
training_angles = np.array(training_behaviour[training_item])   
testing_angles = np.array(testing_behaviour[decode_item])    
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
    y_train, y_test = training_angles[train_index], testing_angles[test_index]
    ## train
    WM2, Inter2 = Weights_matrix_LM(X_train, y_train)
    WM_t2 = WM2.transpose()