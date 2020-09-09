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
Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

# fit the model
clf = svm.NuSVC(gamma='auto')
clf.fit(X, Y)

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
plt.show()



#### Randomizaciones varias

np.random.randint(low=2, high=5, size=(300,3)) # una array con min y max de dimensiones (size)
np.random.randn(300, 2) #una array de estas dimensiones de distrib normal

import random
a=[1,3,4,5,67]
[random.choice(a) for i in range(10)] # elegir de a n veces 
np.array(random.sample(a, len(a)) ) #mezclar a


a = df.col.values # nezclar a como antes, pero si es una columna de un dataframe
shuff= np.array(random.sample(a, len(a)) )
df['a_shuff'] = a





final=[]

n_animales=20
n_dias=7

for animal in range(n_animales):
    animalito=[] #cada animal será una lista. Dentro de esta lista habra tantas listas como dias (20)
    for dia in range(n_dias):
        numeritos=[1,2,3,4] #numeros del 1 al 4
        random.shuffle(numeritos) #los mezclo
        tres_ =  numeritos[:3] #pillo los tres primeros, alterantiva: [numeritos.pop(0) for i in range(3) ] 
        animalito.append(tres_) #los guardo "dentro" del animal
    ##
    final.append(animalito) #cuando he hecho todos los dias, guardo el animal en "final"


