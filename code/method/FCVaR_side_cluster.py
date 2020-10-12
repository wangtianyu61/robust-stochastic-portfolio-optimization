# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 17:42:49 2020

@author: wangt
"""

import pandas as pd
import numpy as np
import math


from CVaR_parameter import *
from gurobipy import *
from method.strategy import *

class FCVaR_side_cluster(strategy, strategy_cluster):
    portfolio_number = 10
    weight = np.zeros(portfolio_number)
    method_name = "strategy"
    def __init__(self, df_select, df_train, rolling_day, portfolio_number, df_factor, cluster_number, method_name):
        strategy.__init__(self, df_select, df_train, rolling_day, portfolio_number)
        self.df_factor = df_factor
        self.method_name = method_name
        self.cluster_number = cluster_number
    
    def indicate(self, factor_center, indicate_factor, mean_info, cov_info):
        dist = np.zeros(self.cluster_number)
        
        for i in range(self.cluster_number):
            dif = np.array(factor_center.iloc[i]) - np.array(indicate_factor)
            dist[i] = np.linalg.norm(dif, ord = 2)
            #the default value of distance metrics is 2
        index = np.where(dist == np.min(dist))[0][0]
        return [np.array(mean_info.iloc[index]),np.matrix(cov_info.iloc[self.portfolio_number*index:self.portfolio_number*(index+1)])]
    def optimize(self, train_return_mean,train_return_covar,test_return,weight_pre,tran_cost_p, epsilon, shortsale_sign):
        m = Model("Popescu_Bound_no_clusters")

        # Create variables
        #print(column_name)
        
        mu = pd.Series(train_return_mean.tolist())

        #print(type(mu))
        S = pd.DataFrame(train_return_covar,index = None)
        #print(type(S))

        (num_of_sample, port_num) = test_return.shape 

        if shortsale_sign == 0:
            weight = pd.Series(m.addVars(port_num))
        else:
            weight = pd.Series(m.addVars(port_num, lb = -GRB.INFINITY))
            
        weight_dif = pd.Series(m.addVars(port_num))
        #print(type(weight))

        covar = m.addVar(name = 'covar',lb = 0)#sqrt((mu*x+v)^2+x'sigmax
        v = m.addVar(name = 'v', lb = -GRB.INFINITY, ub = GRB.INFINITY)
        ## Set the objective: 
        
        obj = v + (covar - np.dot(mu,weight) - v + tran_cost_p*np.sum(weight_dif))/(2*epsilon) 
        m.setObjective(obj, GRB.MINIMIZE)
    
        # Set the constraints:
        m.addConstr(S.dot(weight).dot(weight) + (np.dot(mu,weight) + v - tran_cost_p*np.sum(weight_dif))*(np.dot(mu,weight) + v - tran_cost_p*np.sum(weight_dif)) <= covar*covar, "c0")
        m.addConstr(weight.sum() == 1,'budget')
        
        m.addConstrs((weight[i] - weight_pre[i] <= weight_dif[i]    
                    for i in range(port_num)),'abs1')
        m.addConstrs((weight_pre[i] - weight[i] <= weight_dif[i]
                    for i in range(port_num)),'abs2')


        #Solve the Optimization Problem
        m.setParam('OutputFlag',0)
        m.optimize()

        #Retrieve the weight
        #print("The optimal weight portfolio from the Popescu Bound without clusters is :")
        
        weight = [v.x for v in weight]
        #print(weight)

        #test_return = np.matrix(test_return)

    
        tran_cost = 0
        for i in range(port_num):
            tran_cost = tran_cost + tran_cost_p*abs(weight[i] - weight_pre[i])
            
        self.turnover = self.turnover + np.sum(abs(weight - weight_pre)) 
        [return_list,weight] = self.show_return(test_return,weight)

        return [weight, return_list*(1 - tran_cost)]
    
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
            
            factor_data = self.df_factor[i: i + num_of_train]
            
            indicate_factor = self.df_factor[i + num_of_train - validation_period: i + num_of_train]
            
            [factor_center, mean_info, cov_info] = self.factor_return_cluster(train_return, factor_data, self.column_name,self.cluster_number)
            
            [train_return_mean, train_return_covar] = self.indicate(factor_center, indicate_factor.mean(), mean_info, cov_info)
            
            [self.weight, self.return_array[i:i + self.rolling_day]] = self.optimize(train_return_mean,train_return_covar,test_return,self.weight,tran_cost_p, epsilon, shortsale_sign)
            self.weight = np.array(self.weight)
            i = i + self.rolling_day