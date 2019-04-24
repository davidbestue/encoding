# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:32:56 2019

@author: David
"""

from fake_data_generator import *
from Weigths_matrix import *
import random


n_trials=300
training_angles = np.array([ random.randint(0,359) for i in range(n_trials)])
training_data = fake_data(training_angles)

#
WM = Weights_matrix( training_data, training_angles )
WM