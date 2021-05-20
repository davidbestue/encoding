
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:05:33 2019

@author: David
"""

import sys, os
path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 
sys.path.insert(1, path_tools)
from tools import *


def ch2vrep3(channel):
    #Input the channel activity
    #Return the visual respresentation of this channel activity
    ###
    #It multiplies each channel by its corresponding f function --> 36 values
    #It sums all the 36 values of the 36 channels  --> 36 values (a way to smooth)
    #Equivalent to the population vector
    all_basis_functions=[]
    for pos, ch_value in enumerate(pos_channels):
        a = channel[pos]*np.array( f2(ch_value) )
        #a= sum(a)
        all_basis_functions.append(a)
        #all_basis_functions.append(channel[pos]*array( f2(ch_value)  ))
    
    
    vrep=sum(all_basis_functions)
    return vrep
