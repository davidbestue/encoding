
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:05:33 2019

@author: David
"""



def get_quad(degree):
    if degree <= 90:
        angle = 1
    else:
        if degree <= 180:
            angle = 2
        else:
            if degree <= 270:
                angle = 3
            else:
                if degree < 360:
                    angle = 4
    ###
    return angle
