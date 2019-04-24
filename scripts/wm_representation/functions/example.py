# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:32:56 2019

@author: David Bestue
"""

from basic_functions import *
from fake_data_generator import *
from Weigths_matrix import *
from Representation import *
import random


n_trials_train=800
training_angles = np.array([ random.randint(0,359) for i in range(n_trials_train)])
training_data = fake_data(training_angles) 

##
WM = Weights_matrix( training_data, training_angles )
##

n_trials_test=600
testing_angles = np.array([ random.randint(0,359) for i in range(n_trials_test)])
testing_data = fake_data(testing_angles) 
#random.shuffle(testing_angles)

Representation(WM, testing_data, testing_angles)





