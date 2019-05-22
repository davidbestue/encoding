# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:13:12 2019

@author: David Bestue
"""

#### generator of fake data
import numpy as np

def fake_data(angles):
    #n_vox=100   
    # input list of angles
    data=[]   
    
    for angle in range(len(angles)):
        ang_rad = np.radians(angles[angle]) 
        #
        vxls_1_30 = [np.cos(ang_rad) + np.random.normal(0, 0.001)  for i in range(30)] #30 with cos signal
        vxls_30_50 = [np.sin(ang_rad) + np.random.normal(0, 0.001)  for i in range(20)] #20 with sin signal
        vxls_50_100 = [np.random.normal(0, 0.001)  for i in range(50)] #50 with noise
        #
        vxls_1_100 = vxls_1_30 + vxls_30_50 + vxls_50_100
        data.append( vxls_1_100 )
    
    data_r=np.array(data)
    return data_r
    




def fake_data_3(T, NT1, NT2):
    #n_vox=100   
    # input list of angles
    data_3=[]
    data_3.append(fake_data(T))
    data_3.append(fake_data(NT1))
    data_3.append(fake_data(NT2))
    
    data=[]
    for i in range(len(T)):
        data.append(  data_3[0][i] +  data_3[1][i] +  data_3[2][i] )
    
    
    data_r=np.array(data)
    return data_r



#t=[20, 100]
#nt1=[320, 32]
#nt2=[150, 350]
#fake_data_3(t, nt1, nt2)