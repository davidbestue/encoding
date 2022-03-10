
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:05:33 2019

@author: David
"""

import sys, os
path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 
sys.path.insert(1, path_tools)
from tools import *


def get_free_quadrant(behaviour):
    targets_distractors = behaviour[['T', 'NT1', 'NT2']]
    ###
    targets_distractors['q_t1'] = [get_quad(targets_distractors.iloc[i]['T']) for i in range(len(targets_distractors))]
    targets_distractors['q_nt1'] = [get_quad(targets_distractors.iloc[i]['NT1']) for i in range(len(targets_distractors))]
    targets_distractors['q_nt2'] = [get_quad(targets_distractors.iloc[i]['NT2']) for i in range(len(targets_distractors))]
    ##
    quadrants_free=[]
    ###
    for i in range(len(targets_distractors)):
        qa = targets_distractors.q_t1.iloc[i]
        qb = targets_distractors.q_nt1.iloc[i]
        qc = targets_distractors.q_nt2.iloc[i]
        quadrant_free = list(set([1,2,3,4]) - set([qa,qb,qc]))[0]
        quadrants_free.append(quadrant_free)
    ###
    ###
    behaviour['quadrants_free'] = quadrants_free
    #
    return behaviour

