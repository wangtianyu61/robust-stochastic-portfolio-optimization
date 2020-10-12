# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:53:08 2019

@author: wangt
"""

# basic modules
import numpy as np
import pandas as pd
import random
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

#select part of the data
start_time = 20120101
end_time = 20180101
train_test_split = 20161231

[df_train,df_test,column_name] = data_input(filepath,start_time,end_time,train_test_split)
[train_return,test_return,train_return_mean,train_return_covar] = data_stat(df_train,df_test,column_name)
#we delete the risk_free_rate here to make more sense

cluster_number = 3
[cluster_freq, mean_info, cov_info] = return_cluster(df_train,column_name,test_return,cluster_number)
#these parameters above will function as the parameter of assumed distribution

[cluster_mean, cluster_covariance] = cluster_convert(cluster_number, int(portfolio_number), mean_info, cov_info)


#the parameters controlling whether clustering or not; below the probability of threshold is not clusters
sim_data_name = "Wang " + str(threshold)
csv_name = "../factor model/" + sim_data_name + ".csv"
csvFile = open(csv_name,'a',newline = '')
writer = csv.writer(csvFile)
writer.writerow(column_name)
for j in range(750 + 360):
    [return_mean, return_covariance] = choose_random(threshold, train_return_mean, train_return_covar,
                                        cluster_freq, cluster_mean, cluster_covariance)
    return_data = np.random.multivariate_normal(return_mean,return_covariance)
    writer.writerow(return_data)
    
csvFile.close()
#    