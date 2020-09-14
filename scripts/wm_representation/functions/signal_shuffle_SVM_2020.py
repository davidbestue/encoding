# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:24:32 2019

@author: David Bestue
"""
from model_functions import *
from Weights_matrixs import *
from Representation import *
from process_encoding import *
from process_wm import *
from data_to_use import *
from bootstrap_functions import *
from leave_one_out import *
from joblib import Parallel, delayed
import multiprocessing
import time
import random
from sklearn import svm

#
numcores = multiprocessing.cpu_count() - 10


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

xx, yy = np.meshgrid(np.linspace(-3, 3, 500),
                     np.linspace(-3, 3, 500))
np.random.seed(0)
X = np.random.randn(300, 2)
Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0) * 1

#a=[1,2,3,4]
#Y=np.array([random.choice(a) for i in range(300)] )# elegir de a n veces


#################################################################
################################################################# Desde train and test
#################################################################

X_train = np.random.randn(300, 1)
Y_train = Y = np.random.randint(0,3, 300) 

X_test = np.random.randn(300, 1)
Y_test = Y = np.random.randint(0,3, 300) 

# fit the model
clf = svm.NuSVC(gamma='auto')
clf.fit(X_train, Y_train)

prediction = clf.predict(X_test)
decoding = np.mean(Y_test==prediction)  ### npercentage of correct classifications

#
shuffl_dec = []

for i in range(4):
    Y_shuffl = Y = np.random.randint(0,3, 300) 
    decoding = np.mean(Y_shuffl==prediction) 
    shuffl_dec.append(decoding)



######################################################################
###################################################################### Desde leave one out
###################################################################### 


### error in the fit???'' hoooow????

# plot the decision function for each datapoint on the grid
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
           origin='lower', cmap=plt.cm.PuOr_r)
contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                       linestyles='dashed')
plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired,
            edgecolors='k')
plt.xticks(())
plt.yticks(())
plt.axis([-3, 3, -3, 3])
plt.show(block=False)


