# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 12:08:21 2020

@author: wangt
"""
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from CVaR_parameter import *

class strategy:
    rolling_day = 1
    turnover = 0
    method_name = "strategy"
    def __init__(self, df_select, df_train, rolling_day, portfolio_number):
        self.df_select = df_select
        self.df_train = df_train
        self.rolling_day = rolling_day
        self.portfolio_number = portfolio_number
        self.weight = np.zeros(self.portfolio_number)
        num_of_sample = len(df_select) - len(df_train)
        self.return_array = np.zeros(num_of_sample)
        self.column_name = list(df_select.columns)
        if 'Date' in self.column_name:
            self.column_name.remove('Date')
    
    def show_return(self, test_return, weight):
        (num_of_sample,num) = test_return.shape
        return_list = np.zeros(num_of_sample)
        for i in range(num_of_sample):
            return_list[i] = np.dot(test_return[i],weight)#the next time point
            weight = np.multiply(test_return[i]/100 + 1, weight)
            weight = weight/np.sum(weight) #normalization
        return [return_list,weight]
    
    def finish_flag(self, method_list, return_list):
        method_list.append(self.method_name)
        return_list.append(self.return_array)
        print("Finish "+ self.method_name + " policy!")
        return [method_list, return_list]
    
class strategy_cluster:
    def factor_return_cluster(self, df_train, factor_data, column_name, cluster_number):
        if cluster_type == "KMeans":
            clf = KMeans(n_clusters = cluster_number, random_state = 0, algorithm = 'auto')
            clf = clf.fit(factor_data)
            factor_data['tag_cluster'] = clf.labels_
#        #print(clf.cluster_centers_)
            df_train['tag_cluster'] = clf.labels_
            
        # GMM Algorithms
        elif cluster_type == "GMM":
            gmm = GaussianMixture(n_components = cluster_number, random_state = 0)
            gmm.fit(factor_data)
            
            factor_data['tag_cluster'] = gmm.predict(factor_data)
            df_train['tag_cluster'] = factor_data['tag_cluster']
            
        
        grouped = df_train.groupby(df_train['tag_cluster'])
        factor_group = factor_data.groupby(factor_data['tag_cluster'])
        ## frequence
        ## mean and covariance
        mean_info = grouped.mean()
        cov_info = grouped.cov()
        factor_center = factor_group.mean()
        return [factor_center, mean_info, cov_info]
    
    # compute the hmm model frequency
    ## hmm learning
    def hmm_fit(self, df_train, side_info):
        pass
    #compute the frequency of the transition matrix
    def hmm_state_compute(self, df_train, cluster_number):
        #the initial transition matrix
        transition_matrix = np.zeros((cluster_number, cluster_number))
        state = np.zeros(cluster_number)
        train_tag_cluster = list(df_train['tag_cluster'])
        #compute the frequency 
        for i in range(len(df_train) - 1):
            transition_matrix[train_tag_cluster[i]][train_tag_cluster[i + 1]] += 1
            state[train_tag_cluster[i]] += 1
            
        #normalization
        for i in range(cluster_number):
            transition_matrix[i] = transition_matrix[i]/state[i]
        print(transition_matrix)
        #just the previous month beforehand
        last_state = train_tag_cluster[len(df_train) - 1]
        return transition_matrix[last_state]
    
    def factor_cluster(self,df_train,factor_data,column_name,cluster_number, hmm_state_estimate = False):
    #the process and result of the clustering
        cluster_freq = np.zeros(cluster_number)
        
        # KMeans Algorithms
        if cluster_type == "KMeans":
            clf = KMeans(n_clusters = cluster_number, random_state = 0, algorithm = 'auto')
            clf = clf.fit(factor_data)
#           
#        #print(clf.cluster_centers_)
            df_train['tag_cluster'] = clf.labels_
            
        # GMM Algorithms
        elif cluster_type == "GMM":
            gmm = GaussianMixture(n_components = cluster_number, random_state = 0)
            gmm.fit(factor_data)
        
            df_train['tag_cluster'] = gmm.predict(factor_data)
        
        grouped = df_train[column_name].groupby(df_train['tag_cluster'])
        
        ## frequence
        countall = len(df_train)
        counter = grouped.count()
        #print(counter)
        for index in range(cluster_number):
            cluster_freq[index] = counter.iloc[index,0]/countall

        ## mean and covariance
        mean_info = grouped.mean()
        cov_info = grouped.cov()
        
        #keep the same as the original model
        if hmm_state_estimate == False:
            return [cluster_freq, mean_info, cov_info]
        #change to a new estimate
        else:
            #frequency estimate for the transition matrix
            cluster_freq = self.hmm_state_compute(df_train, cluster_number)
            return [cluster_freq, mean_info, cov_info]
    
    def return_cluster(self,df_train,column_name,cluster_number, hmm_state_estimate = False):
    
        cluster_freq = np.zeros(cluster_number)
        portfolio_data = np.array(df_train[column_name])
        #choose the data to classify
        #the process and the result of the clustering 
        # KMeans Algorithms
        if cluster_type == "KMeans":
            clf = KMeans(n_clusters = cluster_number, random_state = 0, algorithm = 'auto')
            clf = clf.fit(df_train)
#        
#        
#        #print(clf.cluster_centers_)
            df_train['tag_cluster'] = clf.labels_
        
        # GMM Algorithms
        elif cluster_type == "GMM":
            gmm = GaussianMixture(n_components = cluster_number, random_state = 0)
            gmm.fit(df_train)
        
            df_train['tag_cluster'] = gmm.predict(df_train)
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
        
        
        #keep the same as the original model
        if hmm_state_estimate == False:
            return [cluster_freq, mean_info, cov_info]
        #change to a new estimate
        else:
            #frequency estimate for the transition matrix
            cluster_freq = self.hmm_state_compute(df_train, cluster_number)
            return [cluster_freq, mean_info, cov_info]
    
    def hmm_train(self, train_return, time, regime_num):
        #revise the train return into different dimensions: cluster_number * cls_index * portfolio_number
        train_return_regime = [[] for i in range(regime_num)]
        df_seq = pd.read_csv("../factor model/HMFSS.csv")
        remodel = hmm.GaussianHMM(n_components = regime_num, covariance_type = "full", n_iter = 100)
        dfseq = np.array(list(df_seq['seq'])).reshape(-1, 1)
        #hmm_data = np.array(df[df.columns[1:]])
        remodel.fit(dfseq)
        time_state = np.array(list(remodel.predict(dfseq))[time:time + 120])
        
        #120 is the size of rolling window
        for tid in range(len(time_state)):
            #add each time into the regime
            
            train_return_regime[time_state[tid]].append(train_return[tid])
        
        #give an illustration of how many time points in each regime
        #print(time_state, np.sum(time_state[time_state == 0]), np.sum(time_state[time_state == 1]))
        num_time_in_cluster = [len(time_state[time_state == i]) for i in range(regime_num)]
        
        #shows the in the last training data point
        cluster_freq = remodel.transmat_[time_state[time + 120 - 1]]
        
        return [np.array(num_time_in_cluster), cluster_freq, train_return_regime] 

#    def cluster_output(self,df_train, column_name, cluster_number, cluster_sign):
#        if cluster_sign == 0:
#            dataname = "../cluster_tag/trainset_" + str(len(column_name)) + "_" + str(cluster_number) + " clusters.csv" 
#        elif cluster_sign == 1:
#            dataname = "../cluster_tag/trainset_" + str(len(column_name)) + "_" + str(cluster_number) + " clusters (3 factor).csv"
#        else:
#            dataname = "../cluster_tag/trainset_" + str(len(column_name)) + "_" + str(cluster_number) + " clusters (5 factor).csv"
#        df_train.to_csv(dataname) 
#    
#
#    def cluster_convert(self,cluster_number, portfolio_number, mean_info, cov_info):
#        cluster_mean = list()
#        cluster_covariance = list()
#        for i in range(cluster_number):
#            cluster_mean.append(tuple(np.array(mean_info)[i]))
#            cluster_covariance.append(np.array(cov_info)[portfolio_number*i: portfolio_number*(i+1)])
#        return [cluster_mean, cluster_covariance]
        