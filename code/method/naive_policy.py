# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 11:17:11 2019

@author: wangtianyu6162
"""

#naive_policy.py is meant to give the performance 
#give each portfolio the same weight:1/N
import numpy as np
import matplotlib.pyplot as plt

from method.show_return import *
def naive_policy(test_return):
    #the definition of the policy
    (num_of_sample,port_num) = test_return.shape
    
    portfolio_weight =  np.ones(port_num)/port_num
    #print(portfolio_weight)



    #print(date)
    return_naivepolicy = np.array(test_return.dot(portfolio_weight))
   

    #basic info about the return list
    #print(return_naivepolicy)
    
    method_name = "naive policy 1/N portfolios"
    print("==================\nMethod:",method_name)
    print("The number of test sample is ",num_of_sample)
    print("The mean and standard deviation of the return is ")
    print(return_naivepolicy.mean(),return_naivepolicy.std())
    print("==================")
    return [portfolio_weight,return_naivepolicy]

def naive_policy_tran(test_return, weight_pre, tran_cost_p):
    (num_of_sample,port_num) = test_return.shape
    
    portfolio_weight =  np.ones(port_num)/port_num
    #print(portfolio_weight)
    tran_cost = 0
    for i in range(port_num):
        tran_cost = tran_cost + tran_cost_p*abs(portfolio_weight[i] - weight_pre[i])

    #print(date)

        
    [return_naivepolicy, portfolio_weight] = show_return(test_return, portfolio_weight)
    
    return [portfolio_weight, return_naivepolicy*(1 - tran_cost)]
