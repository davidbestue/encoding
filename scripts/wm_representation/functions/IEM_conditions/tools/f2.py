# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:05:33 2019

@author: David
"""

def f2(position_target):
    list_r=[]
    for channel in pos_channels2:
        R = round(circ_dist(position_target, channel), 3)
        list_r.append(R)
    
    f_list=[]
    for r in list_r:
        if r<adjusted_size_contant:
            f = ( 0.5 + 0.5*np.cos(r*np.pi/adjusted_size_contant) )**7
            f=round(f,3)
            f_list.append(f)
        else:
            f = 0
            f_list.append(f)
    
    return f_list
