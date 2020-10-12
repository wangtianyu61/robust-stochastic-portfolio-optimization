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
import os
import time
from scipy import stats
from gurobipy import *

from main_head import *
#------------------------------------------------#
#this two shows the visualization of the different return policies
method_list = list()
return_list = list()
method_time = list()

#choose the dataset and the file path
portfolio_number = 2


filepath = "result_self_sim/" + "xjbgg" + ".csv"

#print(filepath)

#select part of the data


rolling_day = 1

shortsale_sign = 0

data_head = ["filepath","rolling_day","tran_cost_p"]
data_parameter = [filepath,rolling_day,tran_cost_p]
#for writing to csv

csv_name = "result_self_sim/rolling_xjbg" + "_v4.csv"
output_head(csv_name, data_head)

df_select = pd.read_csv(filepath)

#we let the dataset to generate its index

df_select = df_select[0:240]
df_train = df_select[0:120]
num_of_train = len(df_train)
#print(num_of_train)
num_of_sample = len(df_select) - num_of_train


#filepath_factor = "../factor model/Fan2008_"  + "factor.csv"
#df_factor = pd.read_csv(filepath_factor)
start = time.process_time()
naive = naive_strategy(df_select, df_train, rolling_day, portfolio_number, "naive policy")
naive.rolling()
output_return(csv_name, data_parameter, naive.method_name, naive.return_array, naive.turnover)
[method_list, return_list] = naive.finish_flag(method_list, return_list)
method_time.append(float(time.process_time() - start))
base_return = naive.return_array


start = time.process_time()
mvp = MVP(df_select, df_train, rolling_day, portfolio_number, 'MVP')
mvp.rolling(shortsale_sign)
output_return(csv_name, data_parameter, mvp.method_name, mvp.return_array, mvp.turnover)
output_pvalue(csv_name, data_parameter, base_return, mvp.return_array)  
[method_list, return_list] = mvp.finish_flag(method_list, return_list)
method_time.append(float(time.process_time() - start))

start = time.process_time()
markowitz = Markowitz(df_select, df_train, rolling_day, portfolio_number, df_factor, 'Markowitz')
markowitz.rolling(shortsale_sign, 0, 1)
output_return(csv_name, data_parameter, markowitz.method_name, markowitz.return_array, markowitz.turnover)
output_pvalue(csv_name, data_parameter, base_return, markowitz.return_array)  
[method_list, return_list] = markowitz.finish_flag(method_list, return_list)
method_time.append(float(time.process_time() - start))


start = time.process_time()
saa_CVaR = SAA_CVaR(df_select, df_train, rolling_day, portfolio_number, 'SAA_CVaR')
saa_CVaR.rolling(shortsale_sign)
output_return(csv_name, data_parameter, saa_CVaR.method_name, saa_CVaR.return_array, saa_CVaR.turnover)
output_pvalue(csv_name, data_parameter, base_return, saa_CVaR.return_array)  
[method_list, return_list] = saa_CVaR.finish_flag(method_list, return_list)
method_time.append(float(time.process_time() - start))



start = time.process_time()
fcvar = F_CVaR(df_select, df_train, rolling_day, portfolio_number, 'F_CVaR')
fcvar.rolling(shortsale_sign)
output_return(csv_name, data_parameter, fcvar.method_name, fcvar.return_array, fcvar.turnover)
output_pvalue(csv_name, data_parameter, base_return, fcvar.return_array)  
[method_list, return_list] = fcvar.finish_flag(method_list, return_list)
method_time.append(float(time.process_time() - start))


for cluster_number in [2,3,4]:
    start = time.process_time()
    method_name = "FCVaR (" + str(cluster_number) + " cls, return)"
    fcvar_cluster = FCVaR_cluster(df_select, df_train, rolling_day, portfolio_number, 0, 0, cluster_number, method_name)
    fcvar_cluster.rolling(shortsale_sign)
    output_return(csv_name, data_parameter, fcvar_cluster.method_name, fcvar_cluster.return_array, fcvar_cluster.turnover)
    output_pvalue(csv_name, data_parameter, base_return, fcvar_cluster.return_array)  
    [method_list, return_list] = fcvar_cluster.finish_flag(method_list, return_list)
    method_time.append(float(time.process_time() - start))


##print(return_list[0][0])
#output_tail(csv_name)
#output_return(csv_name, data_parameter, method_name, return_array - rfr_data)

os.remove(filepath)