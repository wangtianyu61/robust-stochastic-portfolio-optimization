# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 21:46:04 2019

@author: wangtianyu6162
"""

#show_return.py is for output the during one rolling period
import numpy as np
def show_return(test_return,weight):
    (num_of_sample,num) = test_return.shape
    return_list = np.zeros(num_of_sample)

    
    for i in range(num_of_sample):
        return_list[i] = np.dot(test_return[i],weight)#the next time point
        weight = np.multiply(test_return[i]/100 + 1, weight)
        weight = weight/np.sum(weight) #normalization
    return [return_list,weight]
    