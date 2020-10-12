# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 11:59:47 2020

@author: wangtianyu6162
"""

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import pyplot
import matplotlib.pyplot as plt
import time

from gurobipy import *

from main_head import *

method_list = list()
return_list = list()
method_time = list()

choice = 0
# simulation parameter
train_number = 50
test_number = 50000
if choice == 0:
    portfolio_number = 4
    asset_mean = (0.061166, 0.109547, 0.090358, 0.040923)
    asset_cov = np.array([[0.018632, 0.020056, 0.020646, 0.015213],
                          [0.020056, 0.034507, 0.027412, 0.020652],
                          [0.020646, 0.027412, 0.048680, 0.021663],
                          [0.015213, 0.020652, 0.021663, 0.018791]])

    df_select = pd.DataFrame(np.random.multivariate_normal(asset_mean, asset_cov, train_number + test_number), columns = ['A', 'B', 'C', 'D'])
    df_train = df_select[0: train_number]

elif choice == 1:
    portfolio_number = 30
    df_select = pd.read_csv("../factor model/Fan2008.csv")
    df_select = df_select[0: train_number + test_number]
    df_train = df_select[0: train_number]

threshold = [0.005*i for i in list(range(21))]

#return_list = list()
#max_weight1 = list()
#for value in threshold:
#    fcvar = F_CVaR(df_select, df_train, rolling_day, portfolio_number, 'F_CVaR', False, value)
#    fcvar.rolling(shortsale_sign)
#    [method_list, return_list] = fcvar.finish_flag(method_list, return_list)
#    max_weight1.append(np.mean(fcvar.weight_opt))
#    
#[mean_fcvar, cvar_fcvar] = empirical_mean_cvar(return_list)

return_list = list()
max_weight0 = list()
for value in threshold:
    saa_CVaR = SAA_CVaR(df_select, df_train, rolling_day, portfolio_number, 'SAA_CVaR', False, value)
    saa_CVaR.rolling(shortsale_sign)  
    [method_list, return_list] = saa_CVaR.finish_flag(method_list, return_list)
    max_weight0.append(np.mean(saa_CVaR.weight_opt))
    
[mean_saa, cvar_saa] = empirical_mean_cvar(return_list)

#return_list = list()
#max_weight2 = list()
#method_name = "FCVaR (" + str(2) + " cls,)"
#for value in threshold:
#    fcvar_cluster = FCVaR_cluster(df_select, df_train, rolling_day, portfolio_number, 0, 0, 2, method_name, False, value)
#    fcvar_cluster.rolling(shortsale_sign)
#    [method_list, return_list] = fcvar_cluster.finish_flag(method_list, return_list)
#    max_weight2.append(np.mean(fcvar_cluster.weight_opt))
#[mean_fcvar2, cvar_fcvar2] = empirical_mean_cvar(return_list)
#
#return_list = list()
#max_weight3 = list()
#method_name = "FCVaR (" + str(3) + " cls,)"
#for value in threshold:
#    fcvar_cluster = FCVaR_cluster(df_select, df_train, rolling_day, portfolio_number, 0, 0, 3, method_name, False, value)
#    fcvar_cluster.rolling(shortsale_sign)
#    [method_list, return_list] = fcvar_cluster.finish_flag(method_list, return_list)
#    max_weight3.append(np.mean(fcvar_cluster.weight_opt))
#[mean_fcvar3, cvar_fcvar3] = empirical_mean_cvar(return_list)
#
#return_list = list()
#max_weight4 = list()
#method_name = "FCVaR (" + str(4) + " cls,)"
#for value in threshold:
#    fcvar_cluster = FCVaR_cluster(df_select, df_train, rolling_day, portfolio_number, 0, 0, 4, method_name, False, value)
#    fcvar_cluster.rolling(shortsale_sign)
#    [method_list, return_list] = fcvar_cluster.finish_flag(method_list, return_list)
#    max_weight4.append(np.mean(fcvar_cluster.weight_opt))
#[mean_fcvar4, cvar_fcvar4] = empirical_mean_cvar(return_list)

#draw the plot for cvar
#times = [1 + value for value in threshold]
#plt.figure(figsize = (10, 6), dpi = 100)
#plt.plot(times, cvar_saa, label = 'SAA-CVaR', marker = 'o')
#plt.plot(times, cvar_fcvar, label = 'worst-case CVaR', marker = 'o')
#plt.plot(times, cvar_fcvar2, label = 'FCVaR 2 cluster', marker = '*')
#plt.plot(times, cvar_fcvar3, label = 'FCVaR 3 cluster', marker = '*')
#plt.plot(times, cvar_fcvar4, label = 'FCVaR 4 cluster', marker = '*')
#plt.legend()
##pyplot.yticks([0.05, 0.06, 0.07, 0.08])
#plt.xlabel('Target normalized by t / t_min')
#plt.ylabel('empirical out-of-sample cvar')
#plt.show()
#plt.savefig("dao method1.png")
#
##draw the plot for mean
#times = [1 + value for value in threshold]
#plt.figure(figsize = (10, 6), dpi = 100)
#plt.plot(times, mean_saa, label = 'SAA-CVaR', marker = 'o')
#plt.plot(times, mean_fcvar, label = 'worst-case CVaR', marker = 'o')
#plt.plot(times, mean_fcvar2, label = 'FCVaR 2 cluster', marker = '*')
#plt.plot(times, mean_fcvar3, label = 'FCVaR 3 cluster', marker = '*')
#plt.plot(times, mean_fcvar4, label = 'FCVaR 4 cluster', marker = '*')
#plt.legend()
##pyplot.yticks([0.05, 0.06, 0.07, 0.08])
#plt.xlabel('Target normalized by t / t_min')
#plt.ylabel('empirical out-of-sample average return')
#plt.show()
#plt.savefig("dao method1.png")
#
##draw the plot for infty norm
#times = [1 + value for value in threshold]
#plt.figure(figsize = (10, 6), dpi = 100)
#plt.plot(times, max_weight0, label = 'SAA-CVaR', marker = 'o')
#plt.plot(times, max_weight1, label = 'worst-case CVaR', marker = 'o')
#plt.plot(times, max_weight2, label = 'FCVaR 2 cluster', marker = '*')
#plt.plot(times, max_weight3, label = 'FCVaR 3 cluster', marker = '*')
#plt.plot(times, max_weight4, label = 'FCVaR 4 cluster', marker = '*')
#plt.legend()
##pyplot.yticks([0.05, 0.06, 0.07, 0.08])
#plt.xlabel('Target normalized by t / t_min')
#plt.ylabel('max weight among portfolios')
#plt.show()
#plt.savefig("dao method2.png")

#the mean-cvar efficient frontier
plt.figure(figsize = (10, 6), dpi = 100)
plt.plot(cvar_saa, mean_saa, label = 'SAA-CVaR', marker = '*')
plt.xlabel("empirical out-of-sample CVaR")
plt.ylabel("empirical out-of-sample average return")