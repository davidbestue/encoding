# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:05:33 2019

@author: David
"""

import sys, os
path_tools = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 
sys.path.insert(1, path_tools)
from tools import *


def Pos_IEM2( channel_reconstruction ):
    # make it all positive
    # negative values are positive in somwhere else
    all_channels = list(np.arange(0, len(channel_reconstruction), 1))
    half_ = len(all_channels)/2
    new_ = np.zeros((len(all_channels)))
    #
    POS_channel_reconstruction = list(channel_reconstruction)
    #
    all_channels=list(all_channels)
    #
    #
    for ch in all_channels:
        value_ch = POS_channel_reconstruction[ch]
        if value_ch>0:
            new_[ch] = new_[ch] + value_ch
        if value_ch<0:            
            if ch<half_:
                new_[int(ch+half_)] = new_[int(ch+half_)] - value_ch
            if ch>=half_:
                new_[int(ch-half_)] = new_[int(ch-half_)] - value_ch
    ###
    return new_



