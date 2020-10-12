# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 21:52:15 2020

@author: wangt
"""

#main_rolling_tran.py is for the whole function to input given datasets and split taken account of transaction costs

#basic module
import numpy as np
import pandas as pd
import csv
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
portfolio_number = 10
#different kinds of datasets (10/17/30)

freq = "Monthly"
#Daily/Weekly/Monthly

value = ""
#NONE represents average value weighted return
#_eq represents average equal weighted return

filepath = "../factor model/" + str(portfolio_number) + "_Industry_Portfolios_" + freq + value + ".csv"
data_name = str(portfolio_number) + freq + value +""

#print(filepath)

#select part of the data
start_time = 197812
end_time = 201901
train_test_split = 198811


#data input
Input_csv = Input(portfolio_number, freq, value, start_time, end_time, train_test_split)
[data_head, data_parameter, csv_name] = Input_csv.parameter_output()
[df_select, df_train] = Input_csv.data_load()
df_factor = Input_csv.three_factor_load()
#df_five_factor = Input_csv.five_factor_load()

if sharpe_ratio_open == False:
    rfr_data = 0
else:
    rfr_data = Input_csv.risk_free_rate()


#best_cluster_number = cross_validation(df_train, 0.5, portfolio_number, df_factor, 0, shortsale_sign)
for cluster_number in [1,2,3,4]:
    if cluster_number >1:
        start = time.process_time()
        method_name = "FCVaR (" + str(cluster_number) + " cls, return)"
        fcvar_cluster = FCVaR_cluster(df_select, df_train, rolling_day, portfolio_number, df_factor, 1, cluster_number, method_name)
        fcvar_cluster.rolling(shortsale_sign)
        
        [method_list, return_list] = fcvar_cluster.finish_flag(method_list, return_list)
        method_time.append(float(time.process_time() - start))
        Output_csv = output(csv_name, data_head, data_parameter, fcvar_cluster.return_array)
        Output_csv.return_info(fcvar_cluster, rfr_data)
    else:
        start = time.process_time()
        fcvar = F_CVaR(df_select, df_train, rolling_day, portfolio_number, 'F_CVaR')
        fcvar.rolling(shortsale_sign)
        
        [method_list, return_list] = fcvar.finish_flag(method_list, return_list)
        method_time.append(float(time.process_time() - start))
        Output_csv = output(csv_name, data_head, data_parameter, fcvar.return_array)
        Output_csv.head()
        Output_csv.return_info(fcvar, rfr_data)
        
    start = time.process_time()
    method_name = "MVP (" + str(cluster_number) + " cls, return)"
    mvp = MVP(df_select, df_train, rolling_day, portfolio_number, df_factor, method_name)
    mvp.rolling(shortsale_sign, 0, cluster_number)
    [method_list, return_list] = mvp.finish_flag(method_list, return_list)
    method_time.append(float(time.process_time() - start))
    Output_csv.return_info(mvp, rfr_data)
    Output_csv.pvalue(mvp.return_array)

    start = time.process_time()
    method_name = "Markowitz (" + str(cluster_number) + " cls, return)"
    markowitz = Markowitz(risk_aversion, df_select, df_train, rolling_day, portfolio_number, df_factor, method_name)
    markowitz.rolling(shortsale_sign, 0, cluster_number)
    [method_list, return_list] = markowitz.finish_flag(method_list, return_list)
    method_time.append(float(time.process_time() - start))
    Output_csv.return_info(markowitz, rfr_data)
    Output_csv.pvalue(markowitz.return_array)

    