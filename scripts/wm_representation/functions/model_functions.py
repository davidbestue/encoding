# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:28:22 2019

@author: David Bestue
"""


import numpy as np
import pandas as pd

#Generate the positions of the channels (there will be 14)
sep_channels=10
adjusted_size_contant = 48.519

pos_channels = np.arange(sep_channels/2,360,sep_channels)
pos_channels = [round(pos_channels[i],3) for i in range(0, len(pos_channels))]

pos_channels2 = np.arange(0,360,0.5)
pos_channels2 = [round(pos_channels2[i],3) for i in range(0, len(pos_channels2))]


def circ_dist(a1,a2):
    ## Returns the minimal distance in angles between to angles 
    op1=abs(a2-a1)
    angs=[a1,a2]
    op2=min(angs)+(360-max(angs))
    options=[op1,op2]
    return min(options)



def f(position_target):
    #I want to return a list of the activity of each channel in front of a stim at any location
    #
    #The f function imput is the distance from the position to the channel. That is why first we need to
    #get a distance from the locaion to each channel
    #
    #First i calculate the distance in degrees from the location to each channel
    #Once I have the distance value, I use the same formula as Sprague to extract a value of f for each channel.
    #colculate the r : the circular distance between target position and each channel
    list_r=[]
    for channel in pos_channels:
        R = round(circ_dist(position_target, channel), 3)
        list_r.append(R)
    
    #I need the adjusted because the r is not in visual angles, it is in degrees
    #I calculate the f for those inside the spread of the maximum, farther, it is 0
    f_list=[]
    for r in list_r:
        if r<adjusted_size_contant:
            #f = ( 0.5 + 0.5*cos(r*pi/adjusted_size_contant) )
            f = ( 0.5 + 0.5*np.cos(r*np.pi/adjusted_size_contant) )**7
            f=round(f,3)
            f_list.append(f)
        else:
            f = 0
            f_list.append(f)
    
    
    #Return the list   
    return f_list



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



def ch2vrep3_int(channel):
    #Input the channel activity
    #Return the visual respresentation of this channel activity
    ###
    #It multiplies each channel by its corresponding f function --> 36 values
    #It sums all the 36 values of the 36 channels  --> 36 values (a way to smooth)
    #Equivalent to the population vector
    all_basis_functions=[]
    channel_pos = channel[1:]
    for pos, ch_value in enumerate(pos_channels):
        a = channel[0] + channel_pos[pos]*np.array( f2(ch_value) )
        #a= sum(a)
        all_basis_functions.append(a)
        #all_basis_functions.append(channel[pos]*array( f2(ch_value)  ))
    
    
    vrep=sum(all_basis_functions)
    return vrep



def posch1_to_posch2(ch_1):
    return np.where(np.array(pos_channels2) == pos_channels[ch_1])[0][0]


