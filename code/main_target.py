# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:25:07 2019

@author: wangtianyu6162
"""

#how to beat 1/N in the test
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
start_time = 20090101
end_time = 20150101
train_test_split = 20131231

rolling_day = 5

data_head = ["filepath","start_time","train_test_split","end_time"]
data_parameter = [filepath,start_time,train_test_split,end_time]
#for writing to csv

csv_name = "result/rolling_result_ " + data_name + "_vXXX.csv"
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
    target_rate = just_output_CVaR (return_array - rfr_data)
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
    #target_rate = just_output_CVaR (return_array - rfr_data)
#
output_tail(csv_name)

cluster_number = 3
method_name = "Popescu " + "3 cluster portfolios (3factor) tau = "+ str(target_rate)
return_array = np.zeros(num_of_sample)
i = 0
while i < num_of_sample:
    train_return = df_select[i: i + num_of_train]
        #print(train_return.head())
    if i + num_of_train + rolling_day < len(df_select):       
        test_return = np.array(df_select[i + num_of_train : i + num_of_train + rolling_day])
    else:
        test_return = np.array(df_select[i + num_of_train : len(df_select)])
    factor_data = df_factor[i: i + num_of_train]
#   test_return = np.array(df_select[i + num_of_train : i + num_of_train + 1])
    [cluster_freq, mean_info, cov_info] = factor_cluster(train_return, 
        factor_data,column_name,test_return,cluster_number)
    
    [weight, return_array[i:i + rolling_day]] = FCVaR_cluster_bs(50, cluster_number, cluster_freq, mean_info, cov_info, test_return, target_rate)
    i = i + rolling_day

if weight != "error":
    output_return(csv_name, data_parameter, method_name, return_array - rfr_data)
