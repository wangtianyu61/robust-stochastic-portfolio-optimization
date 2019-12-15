# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 10:11:47 2019

@author: wangtianyu6162
"""

#cluster.py is for return clustering
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def cluster_output(df_train, column_name, cluster_number):
    dataname = "../cluster_tag/trainset_" + str(len(column_name)) + "_" + str(cluster_number) + " clusters.csv" 
    df_train.to_csv(dataname)
    

def return_cluster(df_train,column_name,test_return,cluster_number):
    
    cluster_freq = np.zeros(cluster_number)


    portfolio_data = np.array(df_train[column_name])
    #choose the data to classify

    #the process and the result of the clustering 
    clf = KMeans(n_clusters = cluster_number)
    clf = clf.fit(portfolio_data)

    #print(clf.cluster_centers_)

    #tag the label in the original datasets
    df_train['tag_cluster'] = clf.labels_
    
    #some visualization information about the clustering
    cluster_output(df_train, column_name, cluster_number)
    #print(df_train)



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