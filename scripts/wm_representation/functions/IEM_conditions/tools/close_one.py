
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:05:33 2019

@author: David
"""


def close_one(behaviour):
    targets_distractors = behaviour[['T', 'NT1', 'NT2', 'Dist', 'Dist_NT1', 'Dist_NT2']]
    ###
    target_close_one=[]
    distractor_close_one=[]
    ###
    for i in range(len(targets_distractors)):
        err1 = abs(err_deg(targets_distractors['T'].iloc[i], targets_distractors['Dist'].iloc[i]))
        err2 =abs(err_deg(targets_distractors['NT1'].iloc[i], targets_distractors['Dist_NT1'].iloc[i]))
        err3 =abs(err_deg(targets_distractors['NT2'].iloc[i], targets_distractors['Dist_NT2'].iloc[i]))
        ############
        options_t = ['T', 'NT1', 'NT2']
        options_d = ['Dist', 'Dist_NT1', 'Dist_NT2']
        #
        erros_dist = [err1, err2, err3]
        pos_min_err = np.where( np.array(erros_dist)==min(erros_dist))[0][0]
        #
        target_close_one.append( targets_distractors[options_t[pos_min_err]].iloc[i] )
        distractor_close_one.append( targets_distractors[options_d[pos_min_err]].iloc[i] )
        ###
    ###
    behaviour['T_close'] = target_close_one
    behaviour['dist_close'] = distractor_close_one
    #
    return behaviour
