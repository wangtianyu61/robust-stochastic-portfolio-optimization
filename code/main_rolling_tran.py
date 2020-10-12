# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 14:08:38 2019

@author: wangtianyu6162
"""

#main_rolling_tran.py is for the whole function to input given datasets and split taken account of transaction costs
 
#basic module
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
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
#different kinds of datasets (6*/10/17/30/48)
freq = "Monthly"
#Daily/Weekly/Monthly

value = ""
#NONE represents average value weighted return
#_eq represents average equal weighted return

#select part of the data
start_time = 197812
end_time = 201901
train_test_split = 198811
#------------------------------------------------#



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

a = list()
     

#put the general benchmark for p_value = 1 here
# naive policy, put every equal weight into every asset
#start = time.process_time()
naive = naive_strategy(df_select, df_train, rolling_day, portfolio_number, "naive policy")
naive.rolling()
[method_list, return_list] = naive.finish_flag(method_list, return_list)
#method_time.append(float(time.process_time() - start))
#print('down')
## output the head and the first policy here
Output_csv = output(csv_name, data_head, data_parameter, naive.return_array)
Output_csv.head()
Output_csv.return_info(naive, rfr_data)


   
#Output_csv.return_info(vw, rfr_data)
#Output_csv.pvalue(return_array)

# value weighted policy, put weight which is linear with the mean return of every asset
#start = time.process_time()
#vw = vw_strategy(df_select, df_train, rolling_day, portfolio_number, "vw_strategy")
#vw.rolling()
#[method_list, return_list] = vw.finish_flag(method_list, return_list)
#method_time.append(float(time.process_time() - start))
#
#Output_csv.return_info(vw, rfr_data)
#Output_csv.pvalue(vw.return_array)
#
#minimum variance policy
#start = time.process_time()
#mvp = MVP(df_select, df_train, rolling_day, portfolio_number, df_factor, "MVP-U")
#mvp.rolling(1, 0, 1)
#[method_list, return_list] = mvp.finish_flag(method_list, return_list)
#method_time.append(float(time.process_time() - start))
#
#Output_csv.return_info(mvp, rfr_data)
#Output_csv.pvalue(mvp.return_array)
#
## minimum variance policy non-robust version
#for cluster_number in [1]:
#    mvp = MVP(df_select, df_train, rolling_day, portfolio_number, df_factor, "MVP-C " + str(cluster_number) + " Return")
#    #the fourth parameter means the robust level, 1 indicates we consider the robust case.
#    mvp.rolling(0, 0, cluster_number, 0)
#    [method_list, return_list] = mvp.finish_flag(method_list, return_list)
#
#    Output_csv.return_info(mvp, rfr_data)
#    Output_csv.pvalue(mvp.return_array)
#for cluster_number in [2]:
#    mvp = MVP(df_select, df_train, rolling_day, portfolio_number, df_factor, "MVP-C " + str(cluster_number) + " Robust Return")
#    #the fourth parameter means the robust level, 1 indicates we consider the robust case.
#    mvp.rolling(0, 0, cluster_number, 1)
#    [method_list, return_list] = mvp.finish_flag(method_list, return_list)
#
#    Output_csv.return_info(mvp, rfr_data)
#    Output_csv.pvalue(mvp.return_array)
###
### mean-var model p1
#for risk_aversion in [0.02]:
#    start = time.process_time()
#    method_name = "Markowitz (gamma =" + str(risk_aversion*50) + " C)"
#    markowitz = Markowitz(risk_aversion, df_select, df_train, rolling_day, portfolio_number, df_factor, method_name)
#    markowitz.rolling(0, 0, 1)
#    [method_list, return_list] = markowitz.finish_flag(method_list, return_list)
#    method_time.append(float(time.process_time() - start))
##    
#    Output_csv.return_info(markowitz, rfr_data)
#    Output_csv.pvalue(markowitz.return_array)
##
#for risk_aversion in [0.02]:
#    start = time.process_time()
#    method_name = "Markowitz (gamma =" + str(risk_aversion*50) + " U)"
#    markowitz = Markowitz(risk_aversion, df_select, df_train, rolling_day, portfolio_number, df_factor, method_name)
#    markowitz.rolling(1, 0, 1)
#    [method_list, return_list] = markowitz.finish_flag(method_list, return_list)
#    method_time.append(float(time.process_time() - start))
##    
#    Output_csv.return_info(markowitz, rfr_data)
#    Output_csv.pvalue(markowitz.return_array)
##    
##    
# mean-var model p2 non-robust version
#risk_aversion = 1
#for cluster_number in [1]:
#    method_name = "Markowitz-C (lambda =" + str(risk_aversion*50) + " Return)"
#    markowitz = Markowitz(risk_aversion, df_select, df_train, rolling_day, portfolio_number, df_factor, method_name)
#    #the fourth parameter means the robust level, 1 indicates we consider the robust case.
#    markowitz.rolling(0, 0, cluster_number, 0) 
#    Output_csv.return_info(markowitz, rfr_data)
#    Output_csv.pvalue(markowitz.return_array)

#for cluster_number in [1, 2]:
#    method_name = "Markowitz-C (lambda =" + str(risk_aversion*50) + " Robust Return)"
#    markowitz = Markowitz(risk_aversion, df_select, df_train, rolling_day, portfolio_number, df_factor, method_name)
#    #the fourth parameter means the robust level, 1 indicates we consider the robust case.
#    markowitz.rolling(0, 0, cluster_number, 1) 
#    Output_csv.return_info(markowitz, rfr_data)
#    Output_csv.pvalue(markowitz.return_array)
##
#for risk_aversion in [0.1]:
#    start = time.process_time()
#    method_name = "Markowitz (lambda =" + str(risk_aversion*50) + " U)"
#    markowitz = Markowitz(risk_aversion, df_select, df_train, rolling_day, portfolio_number, df_factor, method_name)
#    markowitz.rolling(1, 0, 1)
#    [method_list, return_list] = markowitz.finish_flag(method_list, return_list)
#    method_time.append(float(time.process_time() - start))
#    
#    Output_csv.return_info(markowitz, rfr_data)
#    Output_csv.pvalue(markowitz.return_array)
##
#SAA-CVaR model
#start = time.process_time()
#saa_CVaR = SAA_CVaR(df_select, df_train, rolling_day, portfolio_number, 'SAA_CVaR')
#saa_CVaR.rolling(shortsale_sign)  
#[method_list, return_list] = saa_CVaR.finish_flag(method_list, return_list)
##method_time.append(float(time.process_time() - start))
#
#Output_csv.return_info(saa_CVaR, rfr_data)
#Output_csv.pvalue(saa_CVaR.return_array)
###
#### FCVaR model
#fcvar = F_CVaR(df_select, df_train, rolling_day, portfolio_number, 'F_CVaR')
#fcvar.rolling(shortsale_sign)
#[method_list, return_list] = fcvar.finish_flag(method_list, return_list)
##method_time.append(float(time.process_time() - start))
#
#Output_csv.return_info(fcvar, rfr_data)
#Output_csv.pvalue(fcvar.return_array)

#fcvar cluster with return
#for cluster_number in [2]:
#     method_name = "FCVaR_F (" + str(cluster_number) + " cls, Return SR Approx.)"
#     fcvar_cluster = FCluster_approximate(df_select, df_train, rolling_day, portfolio_number, df_factor, 0, cluster_number, method_name)
#     print(shortsale_sign)
#     fcvar_cluster.rolling(shortsale_sign)
#     Output_csv.return_info(fcvar_cluster, rfr_data)
#     Output_csv.pvalue(fcvar_cluster.return_array)     
#
#for cluster_number in [2]:
#     method_name = "FCVaR (" + str(cluster_number) + " cls, Return Adj)"
#     fcvar_cluster = FCluster_framework(df_select, df_train, rolling_day, portfolio_number, df_factor, 0, cluster_number, method_name)
#     fcvar_cluster.rolling(shortsale_sign, 1)
#     [method_list, return_list] = fcvar_cluster.finish_flag(method_list, return_list)
#     Output_csv.return_info(fcvar_cluster, rfr_data)
#     Output_csv.pvalue(fcvar_cluster.return_array)   
#
#     method_name = "FCVaR (" + str(cluster_number) + " cls, Return No Adj)"
#     fcvar_cluster = FCluster_framework(df_select, df_train, rolling_day, portfolio_number, df_factor, 0, cluster_number, method_name)
#     fcvar_cluster.rolling(shortsale_sign, 0)
#     Output_csv.return_info(fcvar_cluster, rfr_data)
#     Output_csv.pvalue(fcvar_cluster.return_array)     
#
#for cluster_number in [2]:
#    method_name = "FCVaR (" + str(cluster_number) + " cls, Return Basic)"
#    fcvar_cluster = FCVaR_cluster(df_select, df_train, rolling_day, portfolio_number, df_factor, 0, cluster_number, method_name)
#    fcvar_cluster.rolling(shortsale_sign)
#    [method_list, return_list] = fcvar_cluster.finish_flag(method_list, return_list)
#    Output_csv.return_info(fcvar_cluster, rfr_data)
#    Output_csv.pvalue(fcvar_cluster.return_array)     
#    
#for cluster_number in [2, 3]:
#     method_name = "FCVaR_F (" + str(cluster_number) + " cls, 3Factor SR Approx.)"
#     fcvar_cluster = FCluster_approximate(df_select, df_train, rolling_day, portfolio_number, df_factor, 1, cluster_number, method_name)
#     print(shortsale_sign)
#     fcvar_cluster.rolling(shortsale_sign)
#     Output_csv.return_info(fcvar_cluster, rfr_data)
#     Output_csv.pvalue(fcvar_cluster.return_array)     

#for cluster_number in [3]:
#     method_name = "FCVaR (" + str(cluster_number) + " cls, 3Factor Adj)"
#     fcvar_cluster = FCluster_framework(df_select, df_train, rolling_day, portfolio_number, df_factor, 1, cluster_number, method_name)
#     fcvar_cluster.rolling(shortsale_sign, 1)
#     [method_list, return_list] = fcvar_cluster.finish_flag(method_list, return_list)
#     Output_csv.return_info(fcvar_cluster, rfr_data)
#     Output_csv.pvalue(fcvar_cluster.return_array)   
#
#     method_name = "FCVaR (" + str(cluster_number) + " cls, 3Factor No Adj)"
#     fcvar_cluster = FCluster_framework(df_select, df_train, rolling_day, portfolio_number, df_factor, 1, cluster_number, method_name)
#     fcvar_cluster.rolling(shortsale_sign, 0)
#     Output_csv.return_info(fcvar_cluster, rfr_data)
#     Output_csv.pvalue(fcvar_cluster.return_array)     
#
#for cluster_number in [2, 3]:
#     method_name = "FCVaR (" + str(cluster_number) + " cls, 3Factor Basic)"
#     fcvar_cluster = FCVaR_cluster(df_select, df_train, rolling_day, portfolio_number, df_factor, 1, cluster_number, method_name)
#     fcvar_cluster.rolling(shortsale_sign)
#     [method_list, return_list] = fcvar_cluster.finish_flag(method_list, return_list)
#     Output_csv.return_info(fcvar_cluster, rfr_data)
#     Output_csv.pvalue(fcvar_cluster.return_array)     

##fcvar cluster with return
#for cluster_number in [2,3,4]:
#    start = time.process_time()
#    method_name = "FCVaR (" + str(cluster_number) + " cls, return)"
#    fcvar_cluster = FCVaR_cluster(df_select, df_train, rolling_day, portfolio_number, df_factor, 0, cluster_number, method_name)
#    fcvar_cluster.rolling(shortsale_sign)
##    [method_list, return_list] = fcvar_cluster.finish_flag(method_list, return_list)
##    method_time.append(float(time.process_time() - start))
#    
#    Output_csv.return_info(fcvar_cluster, rfr_data)
#    Output_csv.pvalue(fcvar_cluster.return_array)

  
 
##fcvar cluster with 3-factor
for cluster_number in [2, 3, 4]:
    start = time.process_time()
    method_name = "FCVaR (" + str(cluster_number) + " cls, 3 factor)"
    fcvar_cluster = FCVaR_cluster(df_select, df_train, rolling_day, portfolio_number, df_factor, 1, cluster_number, method_name)
    fcvar_cluster.rolling(shortsale_sign)
#    [method_list, return_list] = fcvar_cluster.finish_flag(method_list, return_list)
#    method_time.append(float(time.process_time() - start))
    
    Output_csv.return_info(fcvar_cluster, rfr_data)
    Output_csv.pvalue(fcvar_cluster.return_array)

#fcvar side cluster with 3-factor info
#for cluster_number in [2,3]:
#    start = time.process_time()
#    method_name = "FCVaR (" + str(cluster_number) + " cls, 3 side-factor)"
#    fcvar_side_cluster = FCVaR_side_cluster(df_select, df_train, rolling_day, portfolio_number, df_factor, cluster_number, method_name)
#    fcvar_side_cluster.rolling(shortsale_sign)
#    [method_list, return_list] = fcvar_side_cluster.finish_flag(method_list, return_list)
#    method_time.append(float(time.process_time() - start))
#    
#    Output_csv.return_info(fcvar_side_cluster, rfr_data)
#    Output_csv.pvalue(fcvar_side_cluster.return_array)

#print_return(method_list, return_list)
##
#Output_csv.tail()  
#Visualization: draw the plot of different policies    
#plt_return_tran(method_list,return_list)

#print(method_time)
    
####print(return_list[0][0])
###output_tail(csv_name)
#return_to_matlab(method_list, return_list)