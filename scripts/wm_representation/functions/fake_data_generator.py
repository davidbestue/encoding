# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:13:12 2019

@author: David Bestue
"""

#### generator of fake data

def fake_data(angles):
    #n_vox=100   
    # input list of angles
    data=[]   
    
    for angle in range(len(angles)):
        ang_rad = np.radians(angles[angle]) 
        #
        vxls_1_30 = [np.cos(ang_rad) + np.random.normal(0, 1)  for i in range(30)] #30 with cos signal
        vxls_30_50 = [np.sin(ang_rad) + np.random.normal(0, 1)  for i in range(20)] #20 with sin signal
        vxls_50_100 = [np.random.normal(0, 1)  for i in range(50)] #50 with noise
        #
        vxls_1_100 = vxls_1_30 + vxls_30_50 + vxls_50_100
        data.append( vxls_1_100 )
    
    data_r=np.array(data)
    return data_r
    
    