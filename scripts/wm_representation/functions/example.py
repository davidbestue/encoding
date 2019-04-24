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
from joblib import Parallel, delayed
import multiprocessing


n_trials_train=400
training_angles = np.array([ random.randint(0,359) for i in range(n_trials_train)])
training_data = fake_data(training_angles) 

##
WM = Weights_matrix( training_data, training_angles )
##

n_trials_test=300
testing_angles = np.array([ random.randint(0,359) for i in range(n_trials_test)])
testing_data = fake_data(testing_angles)
#random.shuffle(testing_angles)




#### paralel!
numcores = multiprocessing.cpu_count()
alltuns = Parallel(n_jobs = numcores)(delayed(Representation)(WM, testing_data, testing_angles) )


#Representation(WM, testing_data, testing_angles)





