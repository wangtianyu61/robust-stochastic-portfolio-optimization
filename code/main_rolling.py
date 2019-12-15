# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 16:09:52 2019

@author: wangtianyu6162
"""

#main_rolling.py is for the whole function to input given datasets and split

#basic module
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

from gurobipy import *

from main_head import *
#------------------------------------------------#

#choose the dataset and the file path
portfolio_number = 10
#different kinds of datasets (10/17/30)

freq = "Daily"
#Daily/Weekly/Monthly

value = ""
#NONE represents average value weighted return
#_eq represents average equal weighted return

filepath = "../factor model/" + str(portfolio_number) + "_Industry_Portfolios_" + freq + value + ".csv"
data_name = str(portfolio_number) + freq + value +""

#print(filepath)

#select part of the data
start_time = 20120101
end_time = 20180101
train_test_split = 20161231

rolling_day = 1

data_head = ["filepath","start_time","train_test_split","end_time"]
data_parameter = [filepath,start_time,train_test_split,end_time]
#for writing to csv

csv_name = "result/rolling_result_ " + data_name + "_vXXXX.csv"
output_head(csv_name, data_head)

df = pd.read_csv(filepath)
#we let the dataset to generate its index

df_select = df[(df['Date']<end_time)&(df['Date']>=start_time)]
df_train = df_select[(df_select['Date']<= train_test_split)]
num_of_train = len(df_train)
#print(num_of_train)
num_of_sample = len(df_select) - num_of_train


#date = list(df_select[(df_select['Date']<train_test_split)])
#print(date)
column_name = list(df_select.columns)
#print(column_name)

column_name.remove('Date')

df_select = df_select[column_name]


three_factor = ['Mkt-RF','SMB','HML']
filepath_factor = "../factor model/F_F_Research_Data_Factors_" + freq + ".csv"

df_factor = pd.read_csv(filepath_factor)

df_factor = df_factor[(df_factor['Date']<end_time)&(df_factor['Date']>=start_time)]


df_factor = df_factor[three_factor]

five_factor = ['Mkt-RF','SMB','HML','RMW','CMA']
filepath_factor = "../factor model/F_F_Research_Data_5Factors_" + freq + ".csv"

df_five_factor = pd.read_csv(filepath_factor)

df_five_factor = df_five_factor[(df_five_factor['Date']<end_time)&(df_five_factor['Date']>=start_time)]
df_five_factor = df_five_factor[five_factor]

rfr_data = risk_free_rate(freq, train_test_split, end_time)



return_array = np.zeros(num_of_sample)
method_name = "naive policy 1/N portfolios"
i = 0
while i < num_of_sample:
    if i + num_of_train + rolling_day < len(df_select):       
        test_return = np.array(df_select[i + num_of_train : i + num_of_train + rolling_day])
    else:
        test_return = np.array(df_select[i + num_of_train : len(df_select)])
    [weight, return_array[i:i + rolling_day]] = naive_policy(test_return)
    i = i + rolling_day
#plt_return(method_name,return_policy)
#print(return_array)
output_return(csv_name, data_parameter, method_name, return_array - rfr_data)
output_tail(csv_name)

return_array = np.zeros(num_of_sample)
for risk_aversion in [0.1,0.5,0.9]:
    method_name = "Markowitz_revised gamma = "+ str(risk_aversion)
    i = 0
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
        [weight, return_array[i:i + rolling_day]] = Markowitz_revised(train_return_mean,train_return_covar,test_return,risk_aversion)
##plt_return(method_name,return_policy)
        i = i + rolling_day
        #print(weight)
    output_return(csv_name, data_parameter, method_name, return_array - rfr_data)
#
output_tail(csv_name)
##
return_array = np.zeros(num_of_sample)
method_name = "SAA_CVaR"
i = 0
while i < num_of_sample:
    train_return = df_select[i: i + num_of_train]
    if i + num_of_train + rolling_day < len(df_select):       
        test_return = np.array(df_select[i + num_of_train : i + num_of_train + rolling_day])
    else:
        test_return = np.array(df_select[i + num_of_train : len(df_select)])
    [weight, return_array[i:i + rolling_day]] = SAA_CVaR(train_return,test_return,epsilon)
#    #plt_return(method_name,return_policy)
    i = i + rolling_day
output_return(csv_name, data_parameter, method_name, return_array - rfr_data)
##
#
return_array = np.zeros(num_of_sample)
method_name = "Popescu 2007"
i = 0
while i < num_of_sample:
    train_return = df_select[i: i + num_of_train]
    if i + num_of_train + rolling_day < len(df_select):       
        test_return = np.array(df_select[i + num_of_train : i + num_of_train + rolling_day])
    else:
        test_return = np.array(df_select[i + num_of_train : len(df_select)])
    train_return_mean = np.array(train_return.mean())
    train_return_covar = np.matrix(train_return.cov())
    [weight, return_array[i:i + rolling_day]] = FCVaR_no_cluster(train_return_mean,train_return_covar,test_return,epsilon)
##plt_return(method_name,return_policy)
    i = i + rolling_day
output_return(csv_name, data_parameter, method_name, return_array - rfr_data)
#
output_tail(csv_name)
#
#
#
for cluster_number in [2,3,4]:
    return_array = np.zeros(num_of_sample)
    method_name = "Popescu " + str(cluster_number) + " cluster portfolios (3factor)"
    i = 0
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
        [weight, return_array[i:i + rolling_day]] = FCVaR_cluster(cluster_number, cluster_freq, mean_info, cov_info, test_return, epsilon)
        i = i + rolling_day
    output_return(csv_name, data_parameter, method_name, return_array - rfr_data)
#
output_tail(csv_name)

for cluster_number in [2,3,4]:
    return_array = np.zeros(num_of_sample)
    method_name = "Popescu " + str(cluster_number) + " cluster portfolios (5factor)"
    i = 0
    while i < num_of_sample:
        train_return = df_select[i: i + num_of_train]
        if i + num_of_train + rolling_day < len(df_select):       
            test_return = np.array(df_select[i + num_of_train : i + num_of_train + rolling_day])
        else:
            test_return = np.array(df_select[i + num_of_train : len(df_select)])
        factor_data = df_five_factor[i: i + num_of_train]
#        test_return = np.array(df_select[i + num_of_train : i + num_of_train + 1])
        [cluster_freq, mean_info, cov_info] = factor_cluster(train_return, 
                factor_data,column_name,test_return,cluster_number)
        [weight, return_array[i:i + rolling_day]] = FCVaR_cluster(cluster_number, cluster_freq, mean_info, cov_info, test_return, epsilon)
        i = i + rolling_day
    output_return(csv_name, data_parameter, method_name, return_array - rfr_data)
    
output_tail(csv_name)   
#
for cluster_number in [2,3,4]:
    return_array = np.zeros(num_of_sample)
    method_name = "Popescu " + str(cluster_number) + " cluster portfolios (return)"
    i = 0
    while i < num_of_sample:
        train_return = df_select[i: i + num_of_train]
        if i + num_of_train + rolling_day < len(df_select):       
            test_return = np.array(df_select[i + num_of_train : i + num_of_train + rolling_day])
        else:
            test_return = np.array(df_select[i + num_of_train : len(df_select)])
#
        [cluster_freq, mean_info, cov_info] = return_cluster(train_return,
                column_name,test_return,cluster_number)
        [weight, return_array[i:i + rolling_day]] = FCVaR_cluster(cluster_number, cluster_freq, mean_info, cov_info,test_return, epsilon)
        i = i + rolling_day
    output_return(csv_name, data_parameter, method_name, return_array - rfr_data)
#
#
output_tail(csv_name)
#print(num_of_train)
#output_tail(csv_name)
#mean_info = df_select[column_name].rolling(window = num_of_train).mean().tail(num_of_sample + 1)
#
#print(mean_info)
#
#cov_info = df_select[column_name].rolling(window = num_of_train).cov().tail(num_of_sample + 1)
#print(cov_info)
