# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 18:50:04 2019

@author: wangtianyu6162
"""

#test_result.py is meant for outputing the results of return and criterion for comparison
#it is also for visualization
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

from CVaR_parameter import risk_aversion,epsilon,target_rate

def plt_return (method_name, return_policy):
    #visualize the return distributions in the test_sample
    return_list = list(return_policy)
    #change the form to plot
    num_of_sample = len(return_list)
    print("The number of test sample in the plot",num_of_sample)
    plt.plot(range(num_of_sample),return_list)
    #if we use the data to illustrate, it is not easy to see clearly.
    plt.xlabel("testing time")
    plt.ylabel("actual return")
    plt.title(method_name)
    plt.show()

def plt_return_tran (tran_cost_p,method_list,return_list): 

    plt.figure(figsize = (10,5))
    for i in range(len(return_list)):
        base = 1
        return_val = list()
        return_val.append(base)
        for j in range(len(return_list[i])):            
            base = base*(1 + return_list[i][j]/100)
            return_val.append(base)
        plt.plot(range(j + 2),return_val,label = method_list[i])
        #if we use the date to illustrate, it is not easy to see clearly.
        
        plt.xlabel("test time")
        plt.ylabel("real return")
        plt.legend()
        plt.title("Return under different policies (p = " + str(tran_cost_p) + ")")
    
    plt.show()
            
def output_head(csv_name, data_parameter):
## the head of one output csv
#read simulation data into the csv

    csvFile = open(csv_name,'a',newline = '')
    writer = csv.writer(csvFile)
    stat_info = ["method","number of sample","test_return_mean","test_return_std","Sharper ratio","Loss_Probability","Conditional Loss","VaR","CVaR"]
    data_parameter.extend(stat_info)
    writer.writerow(data_parameter)
    csvFile.close()

def just_output_CVaR (return_policy):
    VaR = - np.percentile(return_policy,100*epsilon)
    num_of_sample = int(len(return_policy))
    count_CVaR = 0
    CVaR = np.zeros(num_of_sample)
    for i in range(num_of_sample):
        if return_policy[i] < - VaR:
            count_CVaR = count_CVaR + 1
            CVaR[i] = - return_policy[i]
    return CVaR.mean()

def output_return (csv_name, data_parameter, method_name, return_policy):
## the result of one specific policy return
    csvFile = open(csv_name,'a',newline = '')
    writer = csv.writer(csvFile)
    num_of_sample = int(len(return_policy))
    re_mean = return_policy.mean()
    re_std = return_policy.std()
    
    #several measures of statistics of risks
    Sharper_ratio = re_mean / re_std
    VaR = - np.percentile(return_policy,100*epsilon)
    CL = np.zeros(num_of_sample)
    CVaR = np.zeros(num_of_sample)
    count_CL = 0
    count_CVaR = 0
    for i in range(num_of_sample):
        if return_policy[i]< 0:
            count_CL = count_CL + 1
            CL[i] = - return_policy[i]
        if return_policy[i]< - VaR:
            count_CVaR = count_CVaR + 1
            CVaR[i] = - return_policy[i]
    #Normalize CL and CVaR. 
    #Here we just set the benchmark as risk_free_rate
    #in case not sufficient test samples            
    loss_probability = count_CL / num_of_sample 
    
    CL = CL / loss_probability
    
    stat_info = [method_name,num_of_sample,re_mean,re_std, Sharper_ratio, loss_probability, CL.mean(), VaR, CVaR.mean()]
    writer.writerow(data_parameter + stat_info)
    csvFile.close()    

def output_tail(csv_name):
    csvFile = open(csv_name,'a',newline = '')
    writer = csv.writer(csvFile)
    writer.writerow([""])
    csvFile.close()
    