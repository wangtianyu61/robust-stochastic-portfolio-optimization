# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 20:30:21 2019

@author: wangtianyu6162
"""

# Markowitz_revised.py is to adjust the superiority of Markowitz model
# to add the constraint in Markowitz Model that x>=0 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gurobipy import *
from method.show_return import *
def Markowitz_revised(train_return_mean,train_return_covar,test_return,risk_aversion):
#Create a Model
    m = Model("Markowitz")

    # Create variables
    #print(column_name)
    mu = pd.Series(train_return_mean.tolist())

    #print(type(mu))
    S = pd.DataFrame(train_return_covar,index = None)
    #print(type(S))

    (num_of_sample,num) = test_return.shape

    weight = pd.Series(m.addVars(num))
    #print(type(weight))

    #Set the objective:
    obj = risk_aversion*(S.dot(weight).dot(weight))/2 - np.dot(mu,weight)
    m.setObjective(obj, GRB.MINIMIZE)
    
    #Set the constraint:
    m.addConstr(weight.sum() == 1,'budget')
    m.addConstrs((weight[i] >= 0  
                  for i in range(num)),'nonnegative')

    #Solve the Optimization Problem
    m.setParam('OutputFlag',0)
    m.optimize()

    #Retrieve the weight
    #print("The optimal weight portfolio from the Revised Markowitz Model is :")
    weight = [v.x for v in weight]
    #print(weight)
    
    method_name = "Markowitz revised portfolios"
    print("==================\nMethod:",method_name)
    print("The number of test sample is ",num_of_sample)
    print("The mean and standard deviation of the return is ")
    return_Markowitz_revised =  np.array(test_return.dot(weight))
    print(return_Markowitz_revised.mean(),return_Markowitz_revised.std())
    print("==================")
    
    return [weight,return_Markowitz_revised]

def Markowitz_revised_tran(train_return_mean,train_return_covar,test_return,risk_aversion,weight_pre,tran_cost_p):
#Create a Model
    m = Model("Markowitz")

    # Create variables
    #print(column_name)
    mu = pd.Series(train_return_mean.tolist())

    #print(type(mu))
    S = pd.DataFrame(train_return_covar,index = None)
    #print(type(S))

    (num_of_sample,num) = test_return.shape

    weight = pd.Series(m.addVars(num))
    weight_dif = pd.Series(m.addVars(num)) #abs(x(i,t+1)-x(i,t))
    #print(type(weight))

    #Set the objective:
    obj = risk_aversion*(S.dot(weight).dot(weight))/2 + tran_cost_p*np.sum(weight_dif)- np.dot(mu,weight)
    m.setObjective(obj, GRB.MINIMIZE)
    #print(type(weight),type(weight_pre))
    #Set the constraint:
    m.addConstr(weight.sum() == 1,'budget')
    m.addConstrs((weight[i] >= 0  
                  for i in range(num)),'nonnegative')
    m.addConstrs((weight[i] - weight_pre[i] <= weight_dif[i]
                for i in range(num)),'abs1')
    m.addConstrs((weight_pre[i] - weight[i] <= weight_dif[i]
                for i in range(num)),'abs2')

    #Solve the Optimization Problem
    m.setParam('OutputFlag',0)
    m.optimize()

    #Retrieve the weight
    #print("The optimal weight portfolio from the Revised Markowitz Model is :")
    weight = [v.x for v in weight]
    #print(weight)

    
    tran_cost = 0
    for i in range(num):
        tran_cost = tran_cost + tran_cost_p*abs(weight[i] - weight_pre[i])
        
    [return_Markowitz_revised, weight] = show_return(test_return, weight)    
    return [weight,return_Markowitz_revised*(1 - tran_cost)]