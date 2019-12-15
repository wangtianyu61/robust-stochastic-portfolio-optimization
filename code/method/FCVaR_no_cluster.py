# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 14:11:17 2019

@author: wangtianyu6162
"""

#FCVaR_no_cluster.py is meant for robust CVaR with no clusters
#by minimizing the worst-case CVaR and evaluate the performance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gurobipy import *
from method.show_return import *
def FCVaR_no_cluster(train_return_mean,train_return_covar,test_return,epsilon):
# Create a new model
    m = Model("Popescu_Bound_no_clusters")

    # Create variables
    #print(column_name)
    mu = pd.Series(train_return_mean.tolist())

    #print(type(mu))
    S = pd.DataFrame(train_return_covar,index = None)
    #print(type(S))

    (num_of_sample, port_num) = test_return.shape 

    weight = pd.Series(m.addVars(port_num))
    
    #print(type(weight))

    return_weight = m.addVar(name = 'return_weight')#mu*x+v
    covar = m.addVar(name = 'covar')#sqrt((mu*x+v)^2+x'sigmax

    ## Set the objective: 
    obj = (covar - return_weight)/(2*epsilon) + return_weight - np.dot(mu,weight)
    m.setObjective(obj, GRB.MINIMIZE)

    # Set the constraints:
    m.addConstr(S.dot(weight).dot(weight) + return_weight*return_weight <= covar*covar, "c0")
    m.addConstr(weight.sum() == 1,'budget')
    m.addConstrs((weight[i] >= 0  
                  for i in range(port_num)),'nonnegative')
    m.addConstr(return_weight >= 0,'PD1')
    m.addConstr(covar >= 0,'PD2')

    #Solve the Optimization Problem
    m.setParam('OutputFlag',0)
    m.optimize()

    #Retrieve the weight
    #print("The optimal weight portfolio from the Popescu Bound without clusters is :")
    weight = [v.x for v in weight]
    #print(weight)





    test_return = np.matrix(test_return)

    return_Popescu2007 =  np.array(test_return.dot(weight))[0] 
    #change the form to plot

    #basic info about the return list
    #print(return_Popescu2007)
    method_name = "Popescu 2007 portfolios"
    print("==================\nMethod:",method_name)
    print("The number of test sample is ",num_of_sample)
    print("The mean and standard deviation of the return is ")
    
    print(return_Popescu2007.mean(),return_Popescu2007.std())
    print("==================")
    
    return [weight,return_Popescu2007]

def FCVaR_no_cluster_tran(train_return_mean,train_return_covar,test_return,epsilon,weight_pre,tran_cost_p):
# Create a new model
    m = Model("Popescu_Bound_no_clusters")

    # Create variables
    #print(column_name)
    mu = pd.Series(train_return_mean.tolist())

    #print(type(mu))
    S = pd.DataFrame(train_return_covar,index = None)
    #print(type(S))

    (num_of_sample, port_num) = test_return.shape 

    weight = pd.Series(m.addVars(port_num))
    weight_dif = pd.Series(m.addVars(port_num))
    #print(type(weight))

    return_weight = m.addVar(name = 'return_weight')#mu*x+v
    covar = m.addVar(name = 'covar')#sqrt((mu*x+v)^2+x'sigmax

    ## Set the objective: 
    obj = (covar - return_weight + tran_cost_p*np.sum(weight_dif))/(2*epsilon) + return_weight - np.dot(mu,weight)
    m.setObjective(obj, GRB.MINIMIZE)
    
    # Set the constraints:
    m.addConstr(S.dot(weight).dot(weight) + (return_weight - tran_cost_p*np.sum(weight_dif))*(return_weight - tran_cost_p*np.sum(weight_dif)) <= covar*covar, "c0")
    m.addConstr(weight.sum() == 1,'budget')
    m.addConstrs((weight[i] >= 0  
                  for i in range(port_num)),'nonnegative')
    m.addConstrs((weight[i] - weight_pre[i] <= weight_dif[i]    
                for i in range(port_num)),'abs1')
    m.addConstrs((weight_pre[i] - weight[i] <= weight_dif[i]
                for i in range(port_num)),'abs2')
    m.addConstr(return_weight >= 0,'PD1')
    m.addConstr(covar >= 0,'PD2')

    #Solve the Optimization Problem
    m.setParam('OutputFlag',0)
    m.optimize()

    #Retrieve the weight
    #print("The optimal weight portfolio from the Popescu Bound without clusters is :")
    weight = [v.x for v in weight]
    #print(weight)

    #test_return = np.matrix(test_return)


    tran_cost = 0
    for i in range(port_num):
        tran_cost = tran_cost + tran_cost_p*abs(weight[i] - weight_pre[i])

    [return_list,weight] = show_return(test_return,weight)

    return [weight,return_list*(1 - tran_cost)]
