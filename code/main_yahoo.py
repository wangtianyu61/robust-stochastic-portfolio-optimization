# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 10:00:58 2020

@author: wangtianyu6162
"""

#basic module
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from scipy import stats

from gurobipy import *

from main_head import *

portfolio_number = 20
rfr_data = 0
out_csv = '../yahoo finance/2012_2019_ye2010.csv'
read_csv = '../yahoo finance/merged.csv'

method_list = []
return_list = []
data_head = ["filepath","start_time","train_test_split","end_time"]
data_parameter = [read_csv, '2012-01-01',str(rolling_day) + ' days','2019-12-31']

df_select = pd.read_csv(read_csv, index_col = 0)
df_select = df_select[0:300]
df_factor = df_select
df_train = df_select[0:30]

#naive policy
naive = naive_strategy(df_select, df_train, rolling_day, portfolio_number, "naive policy")
naive.rolling()
[method_list, return_list] = naive.finish_flag(method_list, return_list)

Output_csv = output(out_csv, data_head, data_parameter, naive.return_array)
Output_csv.head()
Output_csv.return_info(naive)

#mininum variance unconstrained 
mvp = MVP(df_select, df_train, rolling_day, portfolio_number, df_factor, "MVP-U")
mvp.rolling(1, 0, 1)
[method_list, return_list] = mvp.finish_flag(method_list, return_list)
Output_csv.return_info(mvp)
Output_csv.pvalue(mvp.return_array)

#minimum variance constrained
mvp = MVP(df_select, df_train, rolling_day, portfolio_number, df_factor, "MVP-C")
mvp.rolling(0, 0, 1)
[method_list, return_list] = mvp.finish_flag(method_list, return_list)
Output_csv.return_info(mvp, rfr_data)
Output_csv.pvalue(mvp.return_array)

#SP MODEL
saa_CVaR = SAA_CVaR(df_select, df_train, rolling_day, portfolio_number, 'SAA_CVaR')
saa_CVaR.rolling(shortsale_sign)  
[method_list, return_list] = saa_CVaR.finish_flag(method_list, return_list)
Output_csv.return_info(saa_CVaR)
Output_csv.pvalue(saa_CVaR.return_array)

#worst-case CVaR model
fcvar = F_CVaR(df_select, df_train, rolling_day, portfolio_number, 'F_CVaR')
fcvar.rolling(shortsale_sign)
[method_list, return_list] = fcvar.finish_flag(method_list, return_list)
Output_csv.return_info(fcvar)
Output_csv.pvalue(fcvar.return_array)

#worst-case CVaR with clusters
for cluster_number in [2, 3, 4, 5]:
     method_name = "FCVaR (" + str(cluster_number) + " cls, return)"
     fcvar_cluster = FCVaR_cluster(df_select, df_train, rolling_day, portfolio_number, df_factor, 0, cluster_number, method_name) 
     fcvar_cluster.rolling(shortsale_sign)
     [method_list, return_list] = fcvar_cluster.finish_flag(method_list, return_list)
     Output_csv.return_info(fcvar_cluster, rfr_data)
     Output_csv.pvalue(fcvar_cluster.return_array)

plt_return_tran(method_list,return_list)

