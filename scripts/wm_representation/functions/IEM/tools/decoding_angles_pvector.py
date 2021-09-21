# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:05:33 2019

@author: David
"""

import sys, os
path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 
sys.path.insert(1, path_tools)
from tools import *






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
    


