# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 17:15:50 2019

@author: wangtianyu6162
"""

#SAA_CVaR.py also sets a benchmark for robust optimization

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gurobipy import *
from method.show_return import *
def SAA_CVaR(train_return,test_return,epsilon):

    m = Model('SAA_CVaR')

    #Create the Variables
    (train_num,port_num) = train_return.shape
    #print(train_return.iloc[0])

    return_weight = pd.Series(m.addVars(train_num)) # - mu*x - v
    weight = pd.Series(m.addVars(port_num))
    #print(weight)
    v = m.addVar(name = 'v')

    #Set the objective
    obj = v + return_weight.sum()/(epsilon*train_num)
    m.setObjective(obj, GRB.MINIMIZE)

    #Set the constraints
    m.addConstrs((weight[i] >= 0  
                  for i in range(port_num)),'nonnegative for weight')
    m.addConstrs((return_weight[i] >= 0
                  for i in range(train_num)),'nonnegative for positive expectation')
    m.addConstrs((return_weight[i] >= -np.dot(train_return.iloc[i],weight) - v
                  for i in range(train_num)),'c0')
    m.addConstr(weight.sum() == 1,'budget')

    #Solve the Linear Optimization Problem
    m.setParam('OutputFlag',0)
    m.optimize()

    #Retrieve the weight
    #print("The optimal weight portfolio from the SAA method is :")
    weight = [v.x for v in weight]
    #print(weight) 

    (num_of_sample,port_num) = test_return.shape
    

    return_SAACVaR =  np.array(test_return.dot(weight))
    #print(return_SAACVaR)

    method_name = "SAA_CVaR portfolios"
    print("==================\nMethod:",method_name)
    print("The number of test sample is ",num_of_sample)
    print("The mean and standard deviation of the return is ")
    return_SAA_CVaR =  np.array(test_return.dot(weight))
    print(return_SAA_CVaR.mean(),return_SAA_CVaR.std())
    print("==================")
    
    return [weight,return_SAACVaR]

def SAA_CVaR_tran(train_return,test_return,epsilon,weight_pre,tran_cost_p):

    m = Model('SAA_CVaR')

    #Create the Variables
    (train_num,port_num) = train_return.shape
    #print(train_return.iloc[0])

    return_weight = pd.Series(m.addVars(train_num)) # - mu*x - v
    weight = pd.Series(m.addVars(port_num))
    weight_dif = pd.Series(m.addVars(port_num))
    #print(weight)
    v = m.addVar(name = 'v')

    #Set the objective
    obj = v + return_weight.sum()/(epsilon*train_num)
    m.setObjective(obj, GRB.MINIMIZE)

    #Set the constraints
    m.addConstrs((weight[i] >= 0  
                  for i in range(port_num)),'nonnegative for weight')
    m.addConstrs((return_weight[i] >= 0
                  for i in range(train_num)),'nonnegative for positive expectation')
    m.addConstrs((return_weight[i] >= -np.dot(train_return.iloc[i],weight) + tran_cost_p*np.sum(weight_dif) - v
                  for i in range(train_num)),'c0')
    m.addConstrs((weight[i] - weight_pre[i] <= weight_dif[i]
                for i in range(port_num)),'abs1')
    m.addConstrs((weight_pre[i] - weight[i] <= weight_dif[i]
                for i in range(port_num)),'abs2')
    m.addConstr(weight.sum() == 1,'budget')

    #Solve the Linear Optimization Problem
    m.setParam('OutputFlag',0)
    m.optimize()

    #Retrieve the weight
    #print("The optimal weight portfolio from the SAA method is :")
    weight = [v.x for v in weight]
    #print(weight) 

    (num_of_sample,port_num) = test_return.shape
    

    
    tran_cost = 0
    for i in range(port_num):
        tran_cost = tran_cost + tran_cost_p*abs(weight[i] - weight_pre[i])
    [return_SAACVaR, weight] = show_return(test_return,weight)    

    return [weight,return_SAACVaR*(1 - tran_cost)]