# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 22:20:01 2020

@author: wangtianyu6162
"""

import pandas as pd
import numpy as np
import math
from CVaR_parameter import *
from gurobipy import *
from method.strategy import *
from method.support import *

class FCluster_approximate(strategy, strategy_cluster, resample):
    portfolio_number = 10
    weight = np.zeros(portfolio_number)
    method_name = "strategy"
    def __init__(self, df_select, df_train, rolling_day, portfolio_number, df_factor, cluster_sign, cluster_number, method_name):
        strategy.__init__(self, df_select, df_train, rolling_day, portfolio_number)
        self.df_factor = df_factor
        self.method_name = method_name
        self.cluster_sign = cluster_sign
        #cluster_sign means choose which method to cluster
        #if cluster_sign = 1, use factor to build clusters
        #else if cluster_sign = 0, just use return to build clusters
        self.cluster_number = cluster_number
   
    def optimize(self,train_return_mean,train_return_covar,test_return, epsilon, weight_pre, tran_cost_p, shortsale_sign): 
        m = Model("FCVaR_frame")
        
        # Create variables
        #print(column_name)
        mu = pd.Series(train_return_mean.tolist())
        S = pd.DataFrame(train_return_covar,index = None)
        (num_of_sample,num) = test_return.shape

        if shortsale_sign == 0:
            weight = pd.Series(m.addVars(num))
        else:
            weight = pd.Series(m.addVars(num, lb = -GRB.INFINITY))
        covar_bound = m.addVar(lb = 0)
        weight_dif = pd.Series(m.addVars(num)) #abs(x(i,t+1)-x(i,t))
        print(self.gamma1, self.gamma2)
        #Set the objective :
        obj = (math.sqrt((1-epsilon)*self.gamma2/epsilon) + math.sqrt(self.gamma1))*covar_bound + tran_cost_p*np.sum(weight_dif)- np.dot(mu,weight)
        m.setObjective(obj, GRB.MINIMIZE)
        #print(type(weight),type(weight_pre))
        #Set the constraint:
        m.addConstr(weight.sum() == 1,'budget')
        m.addConstr(S.dot(weight).dot(weight) <= covar_bound*covar_bound)
        m.addConstrs((weight[i] - weight_pre[i] <= weight_dif[i]
                    for i in range(num)),'abs1')
        m.addConstrs((weight_pre[i] - weight[i] <= weight_dif[i]
                    for i in range(num)),'abs2')

        #Solve the Optimization Problem
        m.setParam('OutputFlag',0)
        m.optimize()

        #Retrieve the weight
        #print("The optimal weight portfolio from the Revised Markowitz Model is :")
        weight = [v.x for v in weight]
        #print(weight)

    
        tran_cost = 0
        for i in range(num):
            tran_cost = tran_cost + tran_cost_p*abs(weight[i] - weight_pre[i]) 
        self.turnover = self.turnover + np.sum(abs(weight - weight_pre))    

        [return_F, weight] = self.show_return(test_return, weight) 

        return [weight, return_F*(1 - tran_cost)]
    
    def optimize_cluster(self, cluster_freq, mean_info, cov_info,test_return, epsilon, weight_pre,tran_cost_p, shortsale_sign):
        (num_of_sample, port_num) = test_return.shape
        covar = list()
        for i in range(self.cluster_number):
            covar.append(cov_info.iloc[port_num*i:port_num*(i+1)])


        # Create a new model
        m = Model("FCVaR_frame_clusters")

        # Create variables


        if shortsale_sign == 0:
            weight = pd.Series(m.addVars(port_num))
        else:
            weight = pd.Series(m.addVars(port_num, lb = -GRB.INFINITY))
        #print(type(weight))
        weight_dif = pd.Series(m.addVars(port_num))
        
        covar_bound = pd.Series(m.addVars(self.cluster_number))
        
        #robust approximate version
        self.gamma1 = np.array([0.1/math.sqrt(cluster_freq[i]) for i in range(self.cluster_number)])
        self.gamma2 = np.array([1 + 2/math.sqrt(cluster_freq[i]) for i in range(self.cluster_number)])
        
        #non-robust version
        #self.gamma1 = np.zeros(self.cluster_number)
        #self.gamma2 = np.ones(self.cluster_number)
        
        print(self.gamma1, self.gamma2)
        ## Set the objectinve: 
        obj = 0 
        for i in range(self.cluster_number):
            obj = obj + cluster_freq[i]*((math.sqrt((1-epsilon)*self.gamma2[i]/epsilon) + math.sqrt(self.gamma1[i]))*covar_bound[i] + 
                                    tran_cost_p*np.sum(weight_dif)- np.dot(mean_info.iloc[i],weight))
            
        m.setObjective(obj, GRB.MINIMIZE)

        # Set the constraints:
        
        m.addConstr(weight.sum() == 1,'budget')
        m.addConstrs((np.dot(covar[i], weight).dot(weight) <= covar_bound[i]*covar_bound[i]
                    for i in range(self.cluster_number)),'first0')
        m.addConstrs((weight[i] - weight_pre[i] <= weight_dif[i]
                    for i in range(port_num)),'abs1')
        m.addConstrs((weight_pre[i] - weight[i] <= weight_dif[i]
                    for i in range(port_num)),'abs2')
       
        #Solve the Optimization Problem
        m.setParam('OutputFlag',0)
        m.optimize()
        
        #Retrieve the weight
        #print("The optimal weight portfolio from the Popescu Bound with clusters is :")
        weight = [v.x for v in weight]
        
        tran_cost = 0
        for i in range(port_num):
            tran_cost = tran_cost + tran_cost_p*abs(weight[i] - weight_pre[i])
            #print(tran_cost)  
        self.turnover = self.turnover + np.sum(abs(weight - weight_pre))  

        [return_FCluster,weight] = self.show_return(test_return,weight)
        
        return [weight, return_FCluster*(1 - tran_cost)]
    
    def rolling(self, shortsale_sign):
        i = 0
        num_of_sample = len(self.df_select)
        num_of_train = len(self.df_train)
        while i < num_of_sample - num_of_train:   
            train_return = self.df_select[i: i + num_of_train]
            if i + num_of_train + self.rolling_day < len(self.df_select):       
                test_return = np.array(self.df_select[i + num_of_train : i + num_of_train + self.rolling_day])
            else:
                test_return = np.array(self.df_select[i + num_of_train : len(self.df_select)])
            if self.cluster_number == 1:     
            #no clusters
                train_return_mean = np.array(train_return.mean())
                train_return_covar = np.cov(train_return, rowvar = False, ddof = 1)
                self.robust_level(train_return, train_return_mean, train_return_covar)
                [self.weight, self.return_array[i:i + self.rolling_day]] = self.optimize(train_return_mean,train_return_covar,
                                                                                        test_return, epsilon, self.weight,tran_cost_p, shortsale_sign)
            
            else:
                if self.cluster_sign == 1:
                    #use factor to cluster
                    factor_data = self.df_factor[i: i + num_of_train]
                    [cluster_freq, mean_info, cov_info] = self.factor_cluster(train_return, factor_data, self.column_name,self.cluster_number)
                else:
                    #use return to cluster
                    [cluster_freq, mean_info, cov_info] = self.return_cluster(train_return,self.column_name,self.cluster_number)
            
                cov_info = cov_info.fillna(0)
                #self.robust_level_cluster(train_return, cluster_freq, mean_info, cov_info)
                [self.weight, self.return_array[i:i + self.rolling_day]] = self.optimize_cluster(cluster_freq, mean_info, cov_info, test_return,
                                                                                        epsilon, self.weight, tran_cost_p, shortsale_sign)
            self.weight = np.array(self.weight)
            i = i + self.rolling_day

