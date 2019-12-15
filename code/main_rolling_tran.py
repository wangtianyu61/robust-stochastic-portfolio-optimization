# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 14:08:38 2019

@author: wangtianyu6162
"""

#main_rolling_tran.py is for the whole function to input given datasets and split taken account of transaction costs
 
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

rolling_day = 5

data_head = ["filepath","start_time","train_test_split","end_time"]
data_parameter = [filepath,start_time,train_test_split,end_time]
#for writing to csv

csv_name = "result_tran/rolling_tran_ " + data_name + "_v1.csv"

#for specific parameters
csvFile = open(csv_name,'a',newline = '')
writer = csv.writer(csvFile)
writer.writerow(["p = " + str(tran_cost_p)])
csvFile.close()

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

#df_five_factor = pd.read_csv(filepath_factor)
#
#df_five_factor = df_five_factor[(df_five_factor['Date']<end_time)&(df_five_factor['Date']>=start_time)]
#df_five_factor = df_five_factor[five_factor]

rfr_data = risk_free_rate(freq, train_test_split, end_time)



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
output_return(csv_name, data_parameter, method_name, return_array - rfr_data)
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

output_return(csv_name, data_parameter, method_name, return_array - rfr_data)  
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
output_return(csv_name, data_parameter, method_name, return_array - rfr_data)
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
    
output_return(csv_name, data_parameter, method_name, return_array - rfr_data)
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
    output_return(csv_name, data_parameter, method_name, return_array - rfr_data)
    method_list.append(method_name)
    return_list.append(return_array)
    
plt_return_tran (tran_cost_p,method_list,return_list)
##
##print(return_list[0][0])
#output_tail(csv_name)
#output_return(csv_name, data_parameter, method_name, return_array - rfr_data)