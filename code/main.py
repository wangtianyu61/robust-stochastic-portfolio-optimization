# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 14:36:52 2019

@author: wangtianyu6162
"""

#main.py is for the whole function to input given datasets and split

# basic modules
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

from gurobipy import *

from main_head import *
#------------------------------------------------#


#choose the dataset and the file path
portfolio_number = str(10)
#different kinds of datasets (10/17/30)

freq = "Daily"
#Daily/Weekly/Monthly

value = ""
#NONE represents average value weighted return
#_eq represents average equal weighted return

filepath = "../factor model/" + portfolio_number + "_Industry_Portfolios_" + freq + value + ".csv"
data_name = portfolio_number + freq + value +"_vs"

#print(filepath)

#select part of the data
start_time = 20100101
end_time = 20150701
train_test_split = 20141231

data_head = ["filepath","start_time","train_test_split","end_time"]
data_parameter = [filepath,start_time,train_test_split,end_time]
#for writing to csv

csv_name = "result/result_ " + data_name + ".csv"
output_head(csv_name, data_head)
#
[df_train,df_test,column_name] = data_input(filepath,start_time,end_time,train_test_split)
#plot_return(df_train, start_time, train_test_split, column_name)

[train_return,test_return,train_return_mean,train_return_covar] = data_stat(df_train,df_test,column_name)
rfr_data = risk_free_rate(freq, train_test_split, end_time)
##print(rfr_data)
#
#
## use different methods to compare
method_name = "naive policy 1/N portfolios"
[weight, return_policy] = naive_policy(test_return)
#plt_return(method_name,return_policy)
output_return(csv_name, data_parameter, method_name, return_policy - rfr_data)
output_tail(csv_name)
#
#method_name = "Markowitz"
#[weight, return_policy] = Markowitz(train_return_mean,train_return_covar,test_return)
#plt_return(method_name,return_policy)
#output_return(csv_name, data_parameter, method_name, return_policy - rfr_data)

for risk_aversion in [0.1,0.3,0.5,0.7,0.9]:
    method_name = "Markowitz_revised gamma = "+ str(risk_aversion)
    [weight, return_policy] = Markowitz_revised(train_return_mean,train_return_covar,test_return,risk_aversion)
#plt_return(method_name,return_policy)
    output_return(csv_name, data_parameter, method_name, return_policy - rfr_data)

output_tail(csv_name)

method_name = "SAA_CVaR"
[weight, return_policy] = SAA_CVaR(train_return,test_return,epsilon)
#plt_return(method_name,return_policy)
output_return(csv_name, data_parameter, method_name, return_policy - rfr_data)
#
method_name = "Popescu 2007"
[weight, return_policy] = FCVaR_no_cluster(train_return_mean,train_return_covar,test_return,epsilon)
#plt_return(method_name,return_policy)
output_return(csv_name, data_parameter, method_name, return_policy - rfr_data)
#
output_tail(csv_name)
for cluster_number in [2,3,4,5]:
    ## three-factor model
    factor_data = three_factor_load(freq,start_time,train_test_split)
    [cluster_freq, mean_info, cov_info] = factor_cluster(df_train, 
                factor_data,column_name,test_return,cluster_number)
 
    method_name = "Popescu " + str(cluster_number) + " cluster portfolios (3factor)"
    [weight, return_policy] = FCVaR_cluster(cluster_number, cluster_freq, mean_info, cov_info,test_return, epsilon)
    #plt_return(method_name, return_policy)
    output_return(csv_name, data_parameter, method_name, return_policy - rfr_data)
    
    method_name = "Popescu " + str(cluster_number) + " cluster portfolios (3factor) bs"
    [weight,return_policy] = FCVaR_cluster_bs(20,cluster_number, cluster_freq, mean_info, cov_info,test_return,target_rate)
    #plt_return(method_name,return_policy)
    output_return(csv_name, data_parameter, method_name, return_policy - rfr_data)

output_tail(csv_name)

for cluster_number in [2,3,4,5]:
    ## five-factor model
    factor_data = five_factor_load(freq,start_time,train_test_split)
    [cluster_freq, mean_info, cov_info] = factor_cluster(df_train, 
                factor_data,column_name,test_return,cluster_number)

    method_name = "Popescu " + str(cluster_number) + " cluster portfolios (5factor)"
    [weight, return_policy] = FCVaR_cluster(cluster_number, cluster_freq, mean_info, cov_info,test_return, epsilon)
    #plt_return(method_name, return_policy)
    output_return(csv_name, data_parameter, method_name, return_policy - rfr_data)

    method_name = "Popescu " + str(cluster_number) + " cluster portfolios (5factor) bs"
    [weight,return_policy] = FCVaR_cluster_bs(20,cluster_number, cluster_freq, mean_info, cov_info,test_return,target_rate)
    #plt_return(method_name,return_policy)
    output_return(csv_name, data_parameter, method_name, return_policy - rfr_data)

output_tail(csv_name)

for cluster_number in [2,3,4,5]:
    [cluster_freq, mean_info, cov_info] = return_cluster(df_train,
            column_name,test_return,cluster_number)

    method_name = "Popescu " + str(cluster_number) + " cluster portfolios (return)"
    [weight, return_policy] = FCVaR_cluster(cluster_number, cluster_freq, mean_info, cov_info,test_return, epsilon)
    #plt_return(method_name, return_policy)
    output_return(csv_name, data_parameter, method_name, return_policy - rfr_data)

    method_name = "Popescu " + str(cluster_number) + " cluster portfolios (return) bs"
    [weight,return_policy] = FCVaR_cluster_bs(20,cluster_number, cluster_freq, mean_info, cov_info,test_return,target_rate)
    #plt_return(method_name,return_policy)
    output_return(csv_name, data_parameter, method_name, return_policy - rfr_data)

output_tail(csv_name)


#sensitivity analysis for target rate in three factor bs
#cluster_number = 3
#for target_rate in [0.5, 1, 1.5, 2, 2.5, 3, 3.5]:
#    factor_data = three_factor_load(freq,start_time,train_test_split)
#    [cluster_freq, mean_info, cov_info] = factor_cluster(df_train, 
#                factor_data,column_name,test_return,cluster_number)
#    method_name = "Popescu " + "3 cluster portfolios (3factor) tau = "+ str(target_rate)
#    [weight, return_policy] = FCVaR_cluster_bs(20, cluster_number, cluster_freq, mean_info, cov_info, test_return, target_rate)
#    if weight != "error":
#        output_return(csv_name, data_parameter, method_name, return_policy - rfr_data)
#
#output_tail(csv_name)
#for target_rate in [0.5, 1, 1.5, 2, 2.5, 3, 3.5]:
#    factor_data = five_factor_load(freq,start_time,train_test_split)
#    [cluster_freq, mean_info, cov_info] = factor_cluster(df_train, 
#                factor_data,column_name,test_return,cluster_number)
#    method_name = "Popescu " + "3 cluster portfolios (5factor) tau = "+ str(target_rate)
#    [weight, return_policy] = FCVaR_cluster_bs(20, cluster_number, cluster_freq, mean_info, cov_info, test_return, target_rate)
#    if weight != "error":
#        output_return(csv_name, data_parameter, method_name, return_policy - rfr_data)

output_tail(csv_name)
output_tail(csv_name)