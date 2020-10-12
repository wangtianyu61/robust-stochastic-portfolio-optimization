#use for SAA-CVaR in real datasets
#Fama-French factor model
#basic module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import time

from gurobipy import *

from main_head import *
#------------------------------------------------#
#this two shows the visualization of the different return policies

method_list = list()
return_list = list()
method_time = list()

#choose the dataset and the file path
portfolio_number = 6
#different kinds of datasets (6*/10/17/30/48)
freq = "Daily"
#Daily/Weekly/Monthly

value = ""
#NONE represents average value weighted return
#_eq represents average equal weighted return

#select part of the data
start_time = 20000101
end_time = 20190101
train_test_split = 20000131
#------------------------------------------------#



#data input
Input_csv = Input(portfolio_number, freq, value, start_time, end_time, train_test_split)
[data_head, data_parameter, csv_name] = Input_csv.parameter_output()
[df_select, df_train] = Input_csv.data_load()

#df_factor = Input_csv.three_factor_load()

threshold = [0.025*i for i in list(range(15))]

return_list = list()
max_weight0 = list()
for value in threshold:
    
    saa_CVaR = SAA_CVaR(df_select, df_train, rolling_day, portfolio_number, 'SAA_CVaR', False, value)
    saa_CVaR.rolling(shortsale_sign)  
    [method_list, return_list] = saa_CVaR.finish_flag(method_list, return_list)
    max_weight0.append(np.mean(saa_CVaR.weight_opt))
#print(return_list)
[mean_saa, cvar_saa] = empirical_mean_cvar(return_list)


plt.figure(figsize = (10, 6), dpi = 100)
plt.plot(cvar_saa, mean_saa, label = 'SAA-CVaR', marker = '*')
plt.xlabel("empirical out-of-sample CVaR")
plt.ylabel("empirical out-of-sample average return")
plt.legend()

return_list = []
naive = naive_strategy(df_select, df_train, rolling_day, portfolio_number, "naive policy")
naive.rolling()
[method_list, return_list] = naive.finish_flag(method_list, return_list)
[mean_naive, cvar_naive] = empirical_mean_cvar(return_list)
#return_list = list()
#max_weight1 = list()
#for value in threshold:
#    fcvar = F_CVaR(df_select, df_train, rolling_day, portfolio_number, 'F_CVaR', False, value)
#    fcvar.rolling(shortsale_sign)
#    [method_list, return_list] = fcvar.finish_flag(method_list, return_list)
#    max_weight1.append(np.mean(fcvar.weight_opt))
#    
#[mean_fcvar, cvar_fcvar] = empirical_mean_cvar(return_list)
#
#plt.figure(figsize = (10, 6), dpi = 100)
#plt.plot(cvar_fcvar, mean_fcvar, label = 'FCVaR', marker = '*')
#plt.xlabel("empirical out-of-sample CVaR")
#plt.ylabel("empirical out-of-sample average return")
#plt.legend()

#return_list = list()
#max_weight2 = list()
#method_name = "FCVaR (" + str(2) + " cls,)"
#for value in threshold:
#    fcvar_cluster = FCVaR_cluster(df_select, df_train, rolling_day, portfolio_number, 0, 0, 2, method_name, False, value)
#    fcvar_cluster.rolling(shortsale_sign)
#    [method_list, return_list] = fcvar_cluster.finish_flag(method_list, return_list)
#    max_weight2.append(np.mean(fcvar_cluster.weight_opt))
#[mean_fcvar2, cvar_fcvar2] = empirical_mean_cvar(return_list)
#plt.figure(figsize = (10, 6), dpi = 100)
#plt.plot(cvar_fcvar2, mean_fcvar2, label = 'FCVaR-2 cluster', marker = '*')
#plt.xlabel("empirical out-of-sample CVaR")
#plt.ylabel("empirical out-of-sample average return")
#plt.legend()
#

