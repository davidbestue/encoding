# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:05:33 2019

@author: David
"""


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
