# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 10:31:49 2020

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
test_number = 500
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
    
threshold = [0.05*i for i in list(range(1,20))]
for value in threshold:
    saa_CVaR = SAA_CVaR(df_select, df_train, rolling_day, portfolio_number, 'SAA_CVaR', value)
    saa_CVaR.rolling(shortsale_sign)  
    [method_list, return_list] = saa_CVaR.finish_flag(method_list, return_list)

[mean1, cvar1] = empirical_mean_cvar(return_list)


return_list = list()
for value in threshold:
    fcvar = F_CVaR(df_select, df_train, rolling_day, portfolio_number, 'F_CVaR', value)
    fcvar.rolling(shortsale_sign)
    [method_list, return_list] = fcvar.finish_flag(method_list, return_list)

[mean2, cvar2] = empirical_mean_cvar(return_list)
#
return_list = list()
method_name = "FCVaR (" + str(2) + " cls,)"
for value in threshold:
    fcvar_cluster = FCVaR_cluster(df_select, df_train, rolling_day, portfolio_number, 0, 0, 2, method_name, value)
    fcvar_cluster.rolling(shortsale_sign)
    [method_list, return_list] = fcvar_cluster.finish_flag(method_list, return_list)
[mean3, cvar3] = empirical_mean_cvar(return_list)
#
return_list = list()
method_name = "FCVaR (" + str(3) + " cls,)"
for value in threshold:
    fcvar_cluster = FCVaR_cluster(df_select, df_train, rolling_day, portfolio_number, 0, 0, 3, method_name, value)
    fcvar_cluster.rolling(shortsale_sign)
    [method_list, return_list] = fcvar_cluster.finish_flag(method_list, return_list)
[mean4, cvar4] = empirical_mean_cvar(return_list)
#
#return_list = list()
#method_name = "FCVaR (" + str(4) + " cls,)"
#for value in threshold:
#    fcvar_cluster = FCVaR_cluster(df_select, df_train, rolling_day, portfolio_number, 0, 0, 4, method_name, value)
#    fcvar_cluster.rolling(shortsale_sign)
#    [method_list, return_list] = fcvar_cluster.finish_flag(method_list, return_list)
#[mean5, cvar5] = empirical_mean_cvar(return_list)
#
#
plt.plot(cvar1, mean1, label = 'SAA-CVaR', marker = 'o')
plt.plot(cvar2, mean2, label = 'worst-case CVaR', marker = 'o')
plt.plot(cvar3, mean3, label = 'FCVaR 2 cluster', marker = 'o')
plt.plot(cvar4, mean4, label = 'FCVaR 3 cluster', marker = 'o')
#plt.plot(cvar5, mean5, label = 'FCVaR 4 cluster', marker = 'o')
plt.legend()

#pyplot.yticks([0.05, 0.06, 0.07, 0.08])
plt.xlabel('cvar')
plt.ylabel('mean')

plt.show()
#print(mean2, cvar2)
#print(mean3, cvar3)