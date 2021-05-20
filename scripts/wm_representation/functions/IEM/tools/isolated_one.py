
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:05:33 2019

@author: David
"""

import sys, os
path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 
sys.path.insert(1, path_tools)
from tools import *


def isolated_one(behaviour):
    targets_distractors = behaviour[['T', 'NT1', 'NT2', 'Dist', 'Dist_NT1', 'Dist_NT2']]
    ###
    targets_distractors['q_t1'] = [get_quad(targets_distractors.iloc[i]['T']) for i in range(len(targets_distractors))]
    targets_distractors['q_nt1'] = [get_quad(targets_distractors.iloc[i]['NT1']) for i in range(len(targets_distractors))]
    targets_distractors['q_nt2'] = [get_quad(targets_distractors.iloc[i]['NT2']) for i in range(len(targets_distractors))]
    targets_distractors['q_dist'] = [get_quad(targets_distractors.iloc[i]['Dist']) for i in range(len(targets_distractors))]
    targets_distractors['q_dist1'] = [get_quad(targets_distractors.iloc[i]['Dist_NT1']) for i in range(len(targets_distractors))]
    targets_distractors['q_dist2'] = [get_quad(targets_distractors.iloc[i]['Dist_NT2']) for i in range(len(targets_distractors))]
    ##
    targets_alone_quadrant=[]
    distractor_alone_quadrant=[]
    ###
    for i in range(len(targets_distractors)):
        targets_quadrants = [targets_distractors['q_t1'].iloc[i], targets_distractors['q_nt1'].iloc[i], targets_distractors['q_nt2'].iloc[i]]                                                                                 
        distractors_quadrants = [targets_distractors['q_dist'].iloc[i], targets_distractors['q_dist1'].iloc[i], targets_distractors['q_dist2'].iloc[i]] 
        ##################
        ################## get target alone
        ##################
        if targets_quadrants[0] not in distractors_quadrants:
            targets_alone_quadrant.append(targets_distractors.iloc[i]['T'])
        elif targets_quadrants[1] not in distractors_quadrants:
            targets_alone_quadrant.append(targets_distractors.iloc[i]['NT1'])
        elif targets_quadrants[2] not in distractors_quadrants:
            targets_alone_quadrant.append(targets_distractors.iloc[i]['NT2'])
        else:
            print('Error distribution stimuli')
        ##################
        ################## get distractor alone
        ##################
        if distractors_quadrants[0] not in targets_quadrants:
            distractor_alone_quadrant.append(targets_distractors.iloc[i]['Dist'])
        elif distractors_quadrants[1] not in targets_quadrants:
            distractor_alone_quadrant.append(targets_distractors.iloc[i]['Dist_NT1'])
        elif distractors_quadrants[2] not in targets_quadrants:
            distractor_alone_quadrant.append(targets_distractors.iloc[i]['Dist_NT2'])
        else:
            print('Error distribution stimuli')
    ###
    ###
    behaviour['T_alone'] = targets_alone_quadrant
    behaviour['dist_alone'] = distractor_alone_quadrant
    #
    return behaviour

