# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 19:16:49 2019

@author: wangtianyu6162
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from gurobipy import *
from method.show_return import *

def FCVaR_cluster(cluster_number, cluster_freq, mean_info, cov_info,test_return,epsilon):
#data preprocessing
    (num_of_sample, port_num) = test_return.shape
    print(port_num)
    covar = list()
    for i in range(cluster_number):
        covar.append(cov_info.iloc[port_num*i:port_num*(i+1)])

    #print(covar[0])

    # Create a new model
    m = Model("Popescu_Bound_clusters")

    # Create variables


    weight = pd.Series(m.addVars(port_num))
    #print(type(weight))

    return_weight = m.addVars(cluster_number, name = 'return_weight')#mu*x+v
    covar_mean = m.addVars(cluster_number,name = 'covar')#sqrt((mu*x+v)^2+x'sigmax
    
    ## Set the objective: 
    obj = 0 
    for i in range(cluster_number):
        obj = obj + cluster_freq[i]*((covar_mean[i] - return_weight[i])/(2*epsilon) + return_weight[i] - np.dot(mean_info.iloc[i],weight))
    m.setObjective(obj, GRB.MINIMIZE)

    # Set the constraints:
    m.addConstrs((np.dot(covar[i],weight).dot(weight) + return_weight[i]*return_weight[i] <= covar_mean[i]*covar_mean[i]
                for i in range(cluster_number)), "c0")
    m.addConstr(weight.sum() == 1,'budget')
    m.addConstrs((weight[i] >= 0  
                  for i in range(port_num)),'nonnegative')
    m.addConstrs((return_weight[i] >= 0
                  for i in range(cluster_number)),'PD1')
    m.addConstrs((covar_mean[i] >= 0
                  for i in range(cluster_number)),'PD2')

    #Solve the Optimization Problem
    m.setParam('OutputFlag',0)
    m.optimize()

    #Retrieve the weight
    #print("The optimal weight portfolio from the Popescu Bound with clusters is :")
    weight = [v.x for v in weight]
    #print(weight)

    print(num_of_sample)
    return_PopescuCluster =  np.array(test_return.dot(weight))
    #print(return_SAACVaR)
    
    method_name = "Popescu " + str(cluster_number) + " cluster portfolios"
    print("==================\nMethod:",method_name)
    print("The number of test sample is ",num_of_sample)
    print("The mean and standard deviation of the return is ")
    
    print(return_PopescuCluster.mean(),return_PopescuCluster.std())
    print("==================")
    
    return [weight,return_PopescuCluster]

def FCVaR_cluster_tran(cluster_number, cluster_freq, mean_info, cov_info,test_return,epsilon,weight_pre,tran_cost_p):
# Create a new model
#data preprocessing
    (num_of_sample, port_num) = test_return.shape
    print(port_num)
    covar = list()
    for i in range(cluster_number):
        covar.append(cov_info.iloc[port_num*i:port_num*(i+1)])

    #print(covar[0])

    # Create a new model
    m = Model("Popescu_Bound_clusters")

    # Create variables


    weight = pd.Series(m.addVars(port_num))
    #print(type(weight))
    weight_dif = pd.Series(m.addVars(port_num))
    
    return_weight = m.addVars(cluster_number, name = 'return_weight')#mu*x+v
    covar_mean = m.addVars(cluster_number,name = 'covar')#sqrt((mu*x+v)^2+x'sigmax
    
    ## Set the objective: 
    obj = 0 
    for i in range(cluster_number):
        obj = obj + cluster_freq[i]*((covar_mean[i] - return_weight[i] + tran_cost_p*np.sum(weight_dif))/(2*epsilon) + return_weight[i] - np.dot(mean_info.iloc[i],weight))
    m.setObjective(obj, GRB.MINIMIZE)

    # Set the constraints:
    m.addConstrs((np.dot(covar[i],weight).dot(weight) + (return_weight[i] - tran_cost_p*np.sum(weight_dif))*(return_weight[i] - tran_cost_p*np.sum(weight_dif))<= covar_mean[i]*covar_mean[i]
                for i in range(cluster_number)), "c0")
    m.addConstr(weight.sum() == 1,'budget')
    m.addConstrs((weight[i] >= 0  
                  for i in range(port_num)),'nonnegative')
    m.addConstrs((weight[i] - weight_pre[i] <= weight_dif[i]
                for i in range(port_num)),'abs1')
    m.addConstrs((weight_pre[i] - weight[i] <= weight_dif[i]
                for i in range(port_num)),'abs2')
    m.addConstrs((return_weight[i] >= 0
                  for i in range(cluster_number)),'PD1')
    m.addConstrs((covar_mean[i] >= 0
                  for i in range(cluster_number)),'PD2')

    #Solve the Optimization Problem
    m.setParam('OutputFlag',0)
    m.optimize()

    #Retrieve the weight
    #print("The optimal weight portfolio from the Popescu Bound with clusters is :")
    weight = [v.x for v in weight]
    tran_cost = 0
    for i in range(port_num):
        tran_cost = tran_cost + tran_cost_p*abs(weight[i] - weight_pre[i])
    print(tran_cost)
    [return_PopescuCluster,weight] = show_return(test_return,weight)
    return [weight, return_PopescuCluster*(1 - tran_cost)]
    