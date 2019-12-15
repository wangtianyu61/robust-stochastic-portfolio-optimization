# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 15:25:34 2019

@author: wangtianyu6162
"""
import pandas as pd
import numpy as np

from gurobipy import *
# adjust min CVaR based on target rate via bi search
def sub_FCVaR_cluster(cluster_number, cluster_freq, mean_info, cov_info,test_return,epsilon):
 #data preprocessing
    (num_of_sample, port_num) = test_return.shape
    covar = list()
    for i in range(cluster_number):
        covar.append(cov_info.iloc[port_num*i:port_num*(i+1)])
    #print(covar[0])
    # Create a new model
    m = Model("Popescu_Bound_no_clusters")
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
    weight = [v.x for v in weight]
    #print(weight)
    return [m.ObjVal, weight]

def FCVaR_cluster_bs(max_iter,cluster_number, cluster_freq, mean_info, cov_info,test_return,target_rate):
    
    (num_of_sample, port_num) = test_return.shape
    lower_ep = 0.01
    upper_ep = 100
    error = 0.001
    best_weight = 0
    #for count in range(max_iter):
    ## lower bound
    [a,w1] = sub_FCVaR_cluster(cluster_number, cluster_freq, mean_info, cov_info,test_return,lower_ep) 
    ## upper bound
    [b,w2] = sub_FCVaR_cluster(cluster_number, cluster_freq, mean_info, cov_info,test_return,upper_ep) 
    print(a,b)
    if b > target_rate:
        print("We cannot realize that minimum target rate!")
        return ["error","error"]
    else:
        for count in range(max_iter):
            x = (lower_ep + upper_ep)/2 
            [mid,w3] = sub_FCVaR_cluster(cluster_number, cluster_freq, mean_info, cov_info,test_return,x) 
            if mid > target_rate + error:
                lower_ep = x
            elif mid < target_rate - error:
                upper_ep = x
            else:
                break
        best_weight = w3
        method_name = "Popescu + " + str(cluster_number) + " clusters" + " best measure"
        print("==================\nMethod:",method_name)
        print("the final epsilon is ",x)
        print("The number of test sample is ",num_of_sample)
        return_bestrial=  np.array(test_return.dot(best_weight))
        print("The mean and standard deviation of the return is ")
        print(return_bestrial.mean(),return_bestrial.std())
        print("==================")
        return [best_weight,return_bestrial]
        
        
