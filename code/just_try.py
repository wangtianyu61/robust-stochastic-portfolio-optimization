# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 00:14:59 2019

@author: wangtianyu6162
"""

#just thinking and try
# basic modules
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from gurobipy import *

from main_head import *
#------------------------------------------------#

#choose the dataset and the file path
portfolio_number = str(17)
#different kinds of datasets (10/17/30)

freq = "Daily"
#Daily/Weekly/Monthly

value = ""
#NONE represents average value weighted return
#_eq represents average equal weighted return

filepath = "../factor model/" + portfolio_number + "_Industry_Portfolios_" + freq + value + ".csv"
data_name = portfolio_number + freq + value +"_v5"

#print(filepath)

#select part of the data
start_time = 20120101
end_time = 20170701
train_test_split = 20161231

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

for PCA_component in [2,3,4,5,6,7,8,9]:
    print("When the component is ",PCA_component)
    pca_return = PCA(n_components = PCA_component)
    new_train_return = pca_return.fit_transform(train_return)
    #plt.scatter(new_train_return_1,new_train_return_2)
    cluster_number = 3
    [cluster_freq, mean_info, cov_info] = factor_cluster(df_train,new_train_return,column_name,test_return,cluster_number)
    print(mean_info)
    print("====================\n")
    method_name = "PCA to " + str(PCA_component) + "D, 3 clusters"
    [weight, return_policy] = FCVaR_cluster(cluster_number, cluster_freq, mean_info, cov_info,test_return, epsilon)
    #plt_return(method_name, return_policy)
    output_return(csv_name, data_parameter, method_name, return_policy - rfr_data)
    
#plt.figure(figsize=(10,6))
#plt.xlabel('dim1')
#plt.ylabel('dim2')
#for i in range(0,len(y)): 
#    if(y[i]==0): 
#        plt.plot(new_train_return[i][0],new_train_return[i][1],"*r") 
#    elif(y[i]==1): 
#        plt.plot(new_train_return[i][0],new_train_return[i][1],"sy") 
#    elif(y[i]==2): 
#        plt.plot(new_train_return[i][0],new_train_return[i][1],"pb")
#plt.show()