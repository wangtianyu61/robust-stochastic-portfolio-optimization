# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 16:28:34 2019

@author: wangtianyu6162
"""
# factor_cluster
## to import data of three factors and thus use the factor_cluster to classify the data
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

def risk_free_rate(freq, train_test_split, end_time):
    filepath_factor = "../factor model/F_F_Research_Data_Factors_" + freq + ".csv"
    #this is the csv path and freq is the same as data_input.py for convenience
    df_rfr = pd.read_csv(filepath_factor)
    df_rfr = df_rfr[((df_rfr['Date'])>train_test_split)&((df_rfr['Date'])<end_time)]
    
    rfr_data = np.array(df_rfr['RF'])
    return rfr_data

def cluster_output(df_train, column_name, cluster_number):
    dataname = "../cluster_tag/trainset_" + str(len(column_name)) + "_" + str(cluster_number) + " clusters (factor).csv" 
    df_train.to_csv(dataname)
    
def three_factor_load(freq,start_time,train_test_split):
    three_factor = ['Mkt-RF','SMB','HML']
    filepath_factor = "../factor model/F_F_Research_Data_Factors_" + freq + ".csv"
    #this is the csv path and freq is the same as data_input.py for convenience

    #print(filepath_factor)
    #read and data and generate its own index
    df_factor = pd.read_csv(filepath_factor)
    #df_factor.info()
    #print(df_factor.head())

    df_factor = df_factor[(df_factor['Date']<=train_test_split)&(df_factor['Date']>=start_time)]
    #print(df_factor.head())
    
    factor_data = np.array(df_factor[three_factor])
    #plt.scatter(list(df_factor['Mkt-RF']),list(df_factor['HML']),s = 10)
    #choose the data to classify

    return factor_data

def five_factor_load(freq,start_time,train_test_split):
    five_factor = ['Mkt-RF','SMB','HML','RMW','CMA']
    filepath_factor = "../factor model/F_F_Research_Data_5Factors_" + freq + ".csv"
    #this is the csv path and freq is the same as data_input.py for convenience

    #print(filepath_factor)
    #read and data and generate its own index
    df_factor = pd.read_csv(filepath_factor)
    #df_factor.info()
    #print(df_factor.head())

    df_factor = df_factor[(df_factor['Date']<=train_test_split)&(df_factor['Date']>=start_time)]
    #print(df_factor.head())
    
    factor_data = np.array(df_factor[five_factor])
    #plt.scatter(list(df_factor['Mkt-RF']),list(df_factor['HML']),s = 10)
    #choose the data to classify

    return factor_data

def factor_cluster(df_train,factor_data,column_name,test_return,cluster_number):

    
    #the process and result of the clustering
    cluster_freq = np.zeros(cluster_number)

    clf = KMeans(n_clusters = cluster_number)
    clf = clf.fit(factor_data)

    #print(clf.cluster_centers_)
    df_train['tag_cluster'] = clf.labels_
    #print(df_train)
    
    #output the information of clustering with three-factor 
    #cluster_output(df_train, column_name, cluster_number)

    #get the information of each cluster
    grouped = df_train[column_name].groupby(df_train['tag_cluster'])

    ## frequence
    countall = len(df_train)
    counter = grouped.count()
    for index in range(cluster_number):
        cluster_freq[index] = counter.iloc[index,0]/countall
        #print(cluster_freq)

    ## mean and covariance
    mean_info = grouped.mean()
    #print(mean_info)
    #print(type(mean_info.iloc[0]))
    cov_info = grouped.cov()
    #print(type(cov_info.iloc[0:10]))
    return [cluster_freq, mean_info, cov_info]
