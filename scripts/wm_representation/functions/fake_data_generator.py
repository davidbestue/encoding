# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:13:12 2019

@author: David Bestue
"""

#### generator of fake data
import numpy as np

def fake_data(angle):
    #n_vox=100    
    ang_rad = np.radians(angle) 
    #
    vxls_1_30 = [np.cos(ang_rad) + np.random.normal(0, 1)  for i in range(30)] #30 with cos signal
    vxls_30_50 = [np.sin(ang_rad) + np.random.normal(0, 1)  for i in range(20)] #20 with sin signal
    vxls_50_100 = [np.random.normal(0, 1)  for i in range(50)] #50 with noise
    #
    vxls_1_100 = vxls_1_30 + vxls_30_50 + vxls_50_100
    
    return vxls_1_100
    
    