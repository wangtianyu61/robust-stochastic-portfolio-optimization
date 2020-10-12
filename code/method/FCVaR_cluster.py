# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 19:16:49 2019

@author: wangtianyu6162
"""
import pandas as pd
import numpy as np
import math
from CVaR_parameter import *
from gurobipy import *
from method.strategy import *

class FCVaR_cluster(strategy, strategy_cluster):
    portfolio_number = 10
    weight = np.zeros(portfolio_number)
    method_name = "strategy"
    #adj_level means the type of robustness optimization
    def __init__(self, df_select, df_train, rolling_day, portfolio_number, df_factor, cluster_sign, cluster_number, method_name, mean_target = False, adj_level = False, hmm_state_estimate = False):
        strategy.__init__(self, df_select, df_train, rolling_day, portfolio_number)
        self.df_factor = df_factor
        self.method_name = method_name
        self.cluster_sign = cluster_sign
        #cluster_sign means choose which method to cluster
        #if cluster_sign = 1, use factor to build clusters
        #else if cluster_sign = 0, just use return to build clusters
        self.cluster_number = cluster_number
        self.mean_target = mean_target
        self.adj_level = adj_level
        self.hmm_state_estimate = hmm_state_estimate
        if type(self.adj_level) != bool:
            self.weight_opt = []
            print('yes')
    
    def optimize(self, cluster_number, cluster_freq, mean_info, cov_info,test_return,epsilon,weight_pre,tran_cost_p, shortsale_sign, train_return_mean):
        # Create a new model
        #data preprocessing    
        (num_of_sample, port_num) = test_return.shape
        covar = list()
        for i in range(cluster_number):
            covar.append(cov_info.iloc[port_num*i:port_num*(i+1)])

        m = Model("Popescu_Bound_clusters")
        v = m.addVar(name = 'v', lb = -GRB.INFINITY, ub = GRB.INFINITY)
        # Create variables
        if shortsale_sign == 0:
            weight = pd.Series(m.addVars(port_num))
        else:
            weight = pd.Series(m.addVars(port_num, lb = -GRB.INFINITY))
        
        #print(type(weight))
        weight_dif = pd.Series(m.addVars(port_num))
        
        covar_mean = m.addVars(cluster_number, lb = 0, name = 'covar')#sqrt((mu*x+v)^2+x'sigmax

        ## Set the objective: 
        obj = v
        for i in range(cluster_number):
            obj = obj + cluster_freq[i]*(covar_mean[i] + tran_cost_p*np.sum(weight_dif)- np.dot(mean_info.iloc[i], weight) - v)/(2*epsilon) 
        # Set the constraints:
        m.addConstrs((np.dot(covar[i], weight).dot(weight) + (np.dot(mean_info.iloc[i],weight) + v - tran_cost_p*np.sum(weight_dif))*(np.dot(mean_info.iloc[i],weight) + v - tran_cost_p*np.sum(weight_dif)) <= covar_mean[i]*covar_mean[i]
                        for i in range(cluster_number)), "c0")

        m.addConstr(weight.sum() == 1,'budget')

        m.addConstrs((weight[i] - weight_pre[i] <= weight_dif[i]
                    for i in range(port_num)),'abs1')
        m.addConstrs((weight_pre[i] - weight[i] <= weight_dif[i]
                    for i in range(port_num)),'abs2')
        
        
        if self.mean_target != False:
            max_value = max(np.array(train_return_mean))
            m.addConstr(np.dot(train_return_mean, weight) >= (1 - self.mean_target*np.sign(max_value))*max_value, "to draw mean-cvar frontier")
    
        m.setObjective(obj, GRB.MINIMIZE)
        #Solve the Optimization Problem
        m.setParam('OutputFlag',0)
        m.optimize()            
        self.base_line = m.objVal#Retrieve the weight
        #print("The optimal weight portfolio from the Popescu Bound with clusters is :")
        weight = [v.x for v in weight]
#        for i in range(cluster_number):
#            print(cluster_freq[i]*((0 - np.dot(mean_info.iloc[i],weight)-v)/(2*epsilon))) 
# 
#        print(covar_mean)

        tran_cost = tran_cost_p*np.sum(abs(weight - weight_pre))
        self.turnover = self.turnover + np.sum(abs(weight - weight_pre))   
        
        [return_PopescuCluster,weight] = self.show_return(test_return, weight)
        return [weight, return_PopescuCluster*(1 - tran_cost)]
    
    def optimize_adj(self, cluster_number, cluster_freq, mean_info, cov_info,test_return,epsilon,weight_pre,tran_cost_p, shortsale_sign, train_return_mean):
        # Create a new model
        #data preprocessing    
        (num_of_sample, port_num) = test_return.shape
        covar = list()
        for i in range(cluster_number):
            covar.append(cov_info.iloc[port_num*i:port_num*(i+1)])

        m = Model("Popescu_Bound_clusters")
        v = m.addVar(name = 'v', lb = -GRB.INFINITY, ub = GRB.INFINITY)
        # Create variables
        if shortsale_sign == 0:
            weight = pd.Series(m.addVars(port_num))
        else:
            weight = pd.Series(m.addVars(port_num, lb = -GRB.INFINITY))
        
        #print(type(weight))
        weight_dif = pd.Series(m.addVars(port_num))
        
        covar_mean = m.addVars(cluster_number, lb = 0, name = 'covar')#sqrt((mu*x+v)^2+x'sigmax
        k = m.addVar(name = 'k', lb = 0, ub = 1)
        
        ## Set the constraint: 
        obj = v
        for i in range(cluster_number):
            obj = obj + cluster_freq[i]*(covar_mean[i] + tran_cost_p*np.sum(weight_dif)- np.dot(mean_info.iloc[i], weight) - v)/(2*epsilon) 
        
        m.addConstr(obj <= (1 + self.adj_level*np.sign(self.base_line))*self.base_line)
        m.addConstrs((np.dot(covar[i], weight).dot(weight) + (np.dot(mean_info.iloc[i],weight) + v - tran_cost_p*np.sum(weight_dif))*(np.dot(mean_info.iloc[i],weight) + v - tran_cost_p*np.sum(weight_dif)) <= covar_mean[i]*covar_mean[i]
                        for i in range(cluster_number)), "c0")

        m.addConstr(weight.sum() == 1,'budget')

        m.addConstrs((weight[i] - weight_pre[i] <= weight_dif[i]
                    for i in range(port_num)),'abs1')
        m.addConstrs((weight_pre[i] - weight[i] <= weight_dif[i]
                    for i in range(port_num)),'abs2')
        
        
        if self.mean_target != False:
            max_value = max(np.array(train_return_mean))
            m.addConstr(np.dot(train_return_mean, weight) >= (1 - self.mean_target*np.sign(max_value))*max_value, "to draw mean-cvar frontier")
    
        m.setObjective(k, GRB.MINIMIZE)
        #Solve the Optimization Problem
        m.addConstrs((k >= weight[i] for i in range(port_num)), 'budget1')
        if shortsale_sign != 0:
            m.addConstrs((k >= -weight[i] for i in range(port_num)), 'budget2')
        m.setParam('OutputFlag',0)
        m.optimize()
        self.weight_opt.append(m.objVal)         
        #print("The optimal weight portfolio from the Popescu Bound with clusters is :")
        weight = [v.x for v in weight]


        tran_cost = tran_cost_p*np.sum(abs(weight - weight_pre))
        self.turnover = self.turnover + np.sum(abs(weight - weight_pre))   
        
        [return_PopescuCluster,weight] = self.show_return(test_return, weight)
        return [weight, return_PopescuCluster*(1 - tran_cost)]
    
    def rolling(self, shortsale_sign):
        i = 0
        num_of_sample = len(self.df_select)
        num_of_train = len(self.df_train)
        pre_info = 0
        while i < num_of_sample - num_of_train:
            
            train_return = self.df_select[i: i + num_of_train]
            train_return_mean = np.array(train_return.mean())
            if i + num_of_train + self.rolling_day < len(self.df_select):       
                test_return = np.array(self.df_select[i + num_of_train : i + num_of_train + self.rolling_day])
            else:
                test_return = np.array(self.df_select[i + num_of_train : len(self.df_select)])
            
            if self.cluster_sign == 1:
                factor_data = self.df_factor[i: i + num_of_train]
                #normal case or with hmm estimate
                [cluster_freq, mean_info, cov_info] = self.factor_cluster(train_return, factor_data, self.column_name,self.cluster_number, self.hmm_state_estimate)
                
            else:
                [cluster_freq, mean_info, cov_info] = self.return_cluster(train_return,self.column_name,self.cluster_number, self.hmm_state_compute)
            
            cov_info = cov_info.fillna(0)
            
            [self.weight, self.return_array[i:i + self.rolling_day]] = self.optimize(self.cluster_number, cluster_freq, mean_info, cov_info, test_return,
                                                                                        epsilon, self.weight, tran_cost_p, shortsale_sign, train_return_mean)
            if type(self.adj_level) != bool:
                [self.weight, self.return_array[i:i + self.rolling_day]] = self.optimize_adj(self.cluster_number, cluster_freq, mean_info, cov_info, test_return,
                                                                                        epsilon, self.weight, tran_cost_p, shortsale_sign, train_return_mean)
            
            self.weight = np.array(self.weight)
            i = i + self.rolling_day
            
    