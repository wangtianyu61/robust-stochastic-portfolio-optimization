# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 19:43:31 2019

@author: wangt
"""

#basic module
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

from gurobipy import *

from main_head import *
#------------------------------------------------#
#this two shows the visualization of the different return policies
method_list = list()
return_list = list()

#choose the dataset and the file path
portfolio_number = 30


filepath = "../factor model/" + "Fan2008" + ".csv"

#print(filepath)

#select part of the data


rolling_day = 30

data_head = ["filepath","rolling_day","tran_cost_p"]
data_parameter = [filepath,rolling_day,tran_cost_p]
#for writing to csv

csv_name = "result_self_sim/rolling_Fan" + "_v1.csv"



output_head(csv_name, data_head)

df_select = pd.read_csv(filepath)
#we let the dataset to generate its index

df_train = df_select[0:840]
num_of_train = len(df_train)
#print(num_of_train)
num_of_sample = len(df_select) - num_of_train


filepath_factor = "../factor model/Fan2008_"  + "factor.csv"

df_factor = pd.read_csv(filepath_factor)

column_name = list()
for i in range(portfolio_number):
    column_name.append("S" + str(i+1))


return_array = np.zeros(num_of_sample)
method_name = "1/N policy"
i = 0
weight = np.zeros(portfolio_number)
while i < num_of_sample:
    if i + num_of_train + rolling_day < len(df_select):       
        test_return = np.array(df_select[i + num_of_train : i + num_of_train + rolling_day])
    else:
        test_return = np.array(df_select[i + num_of_train : len(df_select)])
    #[weight, return_array[i:i + rolling_day]] = naive_policy(test_return)
    [weight, return_array[i:i + rolling_day]] = naive_policy_tran(test_return, weight, tran_cost_p)

    i = i + rolling_day
#plt_return(method_name,return_policy)
#print(return_array)
output_return(csv_name, data_parameter, method_name, return_array)
method_list.append(method_name)
return_list.append(return_array)
print("Finish naive policy!")

return_array = np.zeros(num_of_sample)
method_name = "Markowitz (gamma = 0.5)"
i = 0
weight = np.zeros(portfolio_number)
while i < num_of_sample:
    train_return = df_select[i: i + num_of_train]
    #print(train_return.head())
    if i + num_of_train + rolling_day < len(df_select):       
        test_return = np.array(df_select[i + num_of_train : i + num_of_train + rolling_day])
    else:
        test_return = np.array(df_select[i + num_of_train : len(df_select)])
    #print(test_return)
    train_return_mean = np.array(train_return.mean())
    train_return_covar = np.matrix(train_return.cov())

    [weight, return_array[i:i + rolling_day]] = Markowitz_revised_tran(train_return_mean,train_return_covar,test_return,risk_aversion,weight,tran_cost_p)

    i = i + rolling_day

output_return(csv_name, data_parameter, method_name, return_array)  
method_list.append(method_name)
return_list.append(return_array)
#
print("Finish Markowitz policy!")


return_array = np.zeros(num_of_sample)
method_name = "CVaR(SAA)"
i = 0
weight = np.zeros(portfolio_number)
while i < num_of_sample:
    train_return = df_select[i: i + num_of_train]
    if i + num_of_train + rolling_day < len(df_select):       
        test_return = np.array(df_select[i + num_of_train : i + num_of_train + rolling_day])
    else:
        test_return = np.array(df_select[i + num_of_train : len(df_select)])

    [weight, return_array[i:i + rolling_day]] = SAA_CVaR_tran(train_return,test_return,epsilon,weight,tran_cost_p)

    #print(weight_pre)
    i = i + rolling_day
output_return(csv_name, data_parameter, method_name, return_array)
method_list.append(method_name)
return_list.append(return_array)
#
print("Finish SAA policy!")

return_array = np.zeros(num_of_sample)
method_name = "F-CVaR"
weight = np.zeros(portfolio_number)
i = 0
while i < num_of_sample:
    train_return = df_select[i: i + num_of_train]
    if i + num_of_train + rolling_day < len(df_select):       
        test_return = np.array(df_select[i + num_of_train : i + num_of_train + rolling_day])
    else:
        test_return = np.array(df_select[i + num_of_train : len(df_select)])
    train_return_mean = np.array(train_return.mean())
    train_return_covar = np.matrix(train_return.cov())

    [weight, return_array[i:i + rolling_day]] = FCVaR_no_cluster_tran(train_return_mean,train_return_covar,test_return,epsilon,weight,tran_cost_p)

    #print(weight_pre)
    i = i + rolling_day
    
output_return(csv_name, data_parameter, method_name, return_array)
method_list.append(method_name)
return_list.append(return_array)
print("Finish Popescu(2007) Policy!")


for cluster_number in [2,3,4]:
    return_array = np.zeros(num_of_sample)
    method_name = "F-CVaR (" + str(cluster_number) + " cls, 3factor)"
    i = 0
    weight = np.zeros(portfolio_number)
    while i < num_of_sample:
        train_return = df_select[i: i + num_of_train]
        if i + num_of_train + rolling_day < len(df_select):       
            test_return = np.array(df_select[i + num_of_train : i + num_of_train + rolling_day])
        else:
            test_return = np.array(df_select[i + num_of_train : len(df_select)])
        factor_data = df_factor[i: i + num_of_train]
#        test_return = np.array(df_select[i + num_of_train : i + num_of_train + 1])
        [cluster_freq, mean_info, cov_info] = factor_cluster(train_return, 
                factor_data,column_name,test_return,cluster_number)
     
        [weight, return_array[i:i + rolling_day]] = FCVaR_cluster_tran(cluster_number, cluster_freq, mean_info, cov_info,test_return,epsilon,np.array(weight),tran_cost_p)

#print(weight_pre)
        i = i + rolling_day
    output_return(csv_name, data_parameter, method_name, return_array)
    method_list.append(method_name)
    return_list.append(return_array)
    
plt_return_tran (tran_cost_p,method_list,return_list)
##
##print(return_list[0][0])
#output_tail(csv_name)
#output_return(csv_name, data_parameter, method_name, return_array - rfr_data)