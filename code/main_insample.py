# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 16:25:02 2019

@author: wangt
"""

#basic module
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import os

from gurobipy import *

from main_head import *
#------------------------------------------------#
#this two shows the visualization of the different return policies
method_list = list()
return_list = list()

#choose the dataset and the file path
portfolio_number = 2


filepath = "../factor model/" + "xjbgg" + ".csv"

#print(filepath)

#select part of the data


rolling_day = 1

data_head = ["filepath","rolling_day","tran_cost_p"]
data_parameter = [filepath,rolling_day,tran_cost_p]
#for writing to csv

csv_name = "result_self_sim/rolling_xjbgg_insample" + "_v3.csv"


output_head(csv_name, data_head)

df_select = pd.read_csv(filepath)
print(df_select.head())
column_name = df_select.columns
#we let the dataset to generate its index

df_select = df_select[0:300]
df_train = df_select[0:120]
num_of_train = len(df_train)
#print(num_of_train)
num_of_sample = len(df_select) - num_of_train


#filepath_factor = "../factor model/Fan2008_"  + "factor.csv"
#df_factor = pd.read_csv(filepath_factor)

test_return = np.array(df_train)
return_array = np.zeros(num_of_train)
weight = np.zeros(portfolio_number)
method_name = "1/N policy"
[weight, return_array] = naive_policy_tran(test_return, weight, tran_cost_p)
output_return(csv_name, data_parameter, method_name, return_array)
method_list.append(method_name)
return_list.append(return_array)

return_array = np.zeros(num_of_train)
method_name = "Markowitz (gamma = 0.5)"
train_return_mean = np.array(df_train.mean())
train_return_covar = np.matrix(df_train.cov())
[weight, return_array] = Markowitz_revised_tran(train_return_mean,train_return_covar,test_return,risk_aversion,weight,tran_cost_p)
output_return(csv_name, data_parameter, method_name, return_array)  
method_list.append(method_name)
return_list.append(return_array)


return_array = np.zeros(num_of_train)
method_name = "CVaR(SAA)"
train_return = np.array(df_train)
[weight, return_array] = SAA_CVaR_tran(pd.DataFrame(train_return),test_return,epsilon,weight,tran_cost_p)
output_return(csv_name, data_parameter, method_name, return_array)
method_list.append(method_name)
return_list.append(return_array)


return_array = np.zeros(num_of_train)
method_name = "F-CVaR"
[weight, return_array] = FCVaR_no_cluster_tran(train_return_mean,train_return_covar,test_return,epsilon,weight,tran_cost_p)
output_return(csv_name, data_parameter, method_name, return_array)
method_list.append(method_name)
return_list.append(return_array)


for cluster_number in [2,3,4]:
    return_array = np.zeros(num_of_train)
    method_name = "F-CVaR (" + str(cluster_number) + " cls, return)"
    [cluster_freq, mean_info, cov_info] = return_cluster(train_return, column_name,test_return,cluster_number)     
    [weight, return_array] = FCVaR_cluster_tran(cluster_number, cluster_freq, mean_info, cov_info,test_return,epsilon,np.array(weight),tran_cost_p)        
    output_return(csv_name, data_parameter, method_name, return_array)
    method_list.append(method_name)
    return_list.append(return_array)


plt_return_tran (tran_cost_p,method_list,return_list)
##
##print(return_list[0][0])
#output_tail(csv_name)
#output_return(csv_name, data_parameter, method_name, return_array - rfr_data)
csvFile.close()
os.remove(filepath)