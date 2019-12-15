# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 12:00:23 2019

@author: wangtianyu6162
"""
import numpy as np
import matplotlib.pyplot as plt

def Markowitz(train_return_mean,train_return_covar,test_return):
    sigma_inv = np.linalg.inv(train_return_covar)
    relative_weight = sigma_inv.dot(train_return_mean)
    abs_weight = np.array(relative_weight/np.sum(relative_weight))[0]
    #print(abs_weight)
    
    (num_of_sample,port_num) = test_return.shape     


    return_Markowitz = np.array(test_return.dot(abs_weight))
    #change the form to plot

    #basic info about the return list
    #print(return_Markowitz)

    print(num_of_sample)
    method_name = "Makowitz policy portfolios"
    print("==================\nMethod:",method_name)
    print("The number of test sample is ",num_of_sample)
    print("The mean and standard deviation of the return is ")
    print(return_Markowitz.mean(),return_Markowitz.std())
    print("==================")
    return [abs_weight, return_Markowitz]