# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 20:30:21 2019

@author: wangtianyu6162
"""

# Markowitz_revised.py is to adjust the superiority of Markowitz model
# to add the constraint in Markowitz Model that x>=0 

import math
import pandas as pd
import numpy as np
from CVaR_parameter import *
from gurobipy import *
from method.strategy import *
from method.support import *

class Markowitz(strategy, strategy_cluster, resample):
    portfolio_number = 10
    weight = np.zeros(portfolio_number)
    method_name = "strategy"
    def __init__(self, risk_aversion, df_select, df_train, rolling_day, portfolio_number, df_factor, method_name):
        strategy.__init__(self, df_select, df_train, rolling_day, portfolio_number)
        self.df_factor = df_factor
        self.method_name = method_name
        self.risk_aversion = risk_aversion
    def optimize(self, train_return_mean,train_return_covar,test_return, weight_pre, tran_cost_p, shortsale_sign, robust_sign):
    #Create a Model
        m = Model("Markowitz")
        
        # Create variables
        #print(column_name)
        mu = pd.Series(train_return_mean.tolist())
        S = pd.DataFrame(train_return_covar,index = None)
        (num_of_sample,num) = test_return.shape

        if shortsale_sign == 0:
            weight = pd.Series(m.addVars(num))
        else:
            weight = pd.Series(m.addVars(num, lb = -GRB.INFINITY))
        
        weight_dif = pd.Series(m.addVars(num)) #abs(x(i,t+1)-x(i,t))

        #Set the general constraint:
        m.addConstr(weight.sum() == 1,'budget')
        
        m.addConstrs((weight[i] - weight_pre[i] <= weight_dif[i]
                    for i in range(num)),'abs1')
        m.addConstrs((weight_pre[i] - weight[i] <= weight_dif[i]
                    for i in range(num)),'abs2')
        if robust_sign == 0:
        #Set the objective :
            obj = self.risk_aversion*(S.dot(weight).dot(weight))/2 +tran_cost_p*np.sum(weight_dif)- np.dot(mu,weight)
            m.setObjective(obj, GRB.MINIMIZE)
            #print(type(weight),type(weight_pre))
        else:
            covar_bound = m.addVar(lb = 0)
            m.addConstr(np.dot(S, weight).dot(weight) <= covar_bound*covar_bound,'first0')
            obj = self.gamma2*self.risk_aversion*(S.dot(weight).dot(weight))/2 +tran_cost_p*np.sum(weight_dif)-np.dot(mu,weight) + math.sqrt(self.gamma1)*covar_bound
            m.setObjective(obj, GRB.MINIMIZE)

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

        [return_Markowitz_revised, weight] = self.show_return(test_return, weight) 

        return [weight, return_Markowitz_revised*(1 - tran_cost)]
    
    def optimize_cluster(self, cluster_number, cluster_freq, mean_info, cov_info,test_return,
                         risk_aversion, weight_pre, tran_cost_p, shortsale_sign, robust_sign):
        (num_of_sample, port_num) = test_return.shape
        covar = list()
        for i in range(cluster_number):
            covar.append(cov_info.iloc[port_num*i:port_num*(i+1)])


        # Create a new model
        m = Model("Markowitz_clusters")

        # Create variables


        if shortsale_sign == 0:
            weight = pd.Series(m.addVars(port_num))
        else:
            weight = pd.Series(m.addVars(port_num, lb = -GRB.INFINITY))
        #print(type(weight))
        weight_dif = pd.Series(m.addVars(port_num))
    
        # Set the general constraints:
        
        m.addConstr(weight.sum() == 1,'budget')
        
        m.addConstrs((weight[i] - weight_pre[i] <= weight_dif[i]
                    for i in range(port_num)),'abs1')
        m.addConstrs((weight_pre[i] - weight[i] <= weight_dif[i]
                    for i in range(port_num)),'abs2')
    
        if robust_sign == 0:
        ## Set the objectinve: 
            obj = 0 
            for i in range(cluster_number):
                obj = obj + cluster_freq[i]*(risk_aversion*(np.dot(covar[i],weight).dot(weight)/2) + tran_cost_p*np.sum(weight_dif)- np.dot(mean_info.iloc[i],weight))
            
            m.setObjective(obj, GRB.MINIMIZE)
        else:
            covar_bound = m.addVars(cluster_number, lb = 0)
        
            m.addConstrs((np.dot(covar[i], weight).dot(weight) <= covar_bound[i]*covar_bound[i]
                         for i in range(cluster_number)))
            #robust approximate version

            self.gamma1 = np.array([0.1/math.sqrt(cluster_freq[i]) for i in range(cluster_number)])
            self.gamma2 = np.array([1 + 4/math.sqrt(cluster_freq[i]) for i in range(cluster_number)])
            obj = 0
            for i in range(cluster_number):
                obj = obj + cluster_freq[i]*(risk_aversion*(self.gamma2[i]*np.dot(covar[i], weight).dot(weight)/2 + 
                                            tran_cost_p*np.sum(weight_dif) + math.sqrt(self.gamma1[i])*covar_bound[i] -np.dot(mean_info.iloc[i],weight)))
            m.setObjective(obj, GRB.MINIMIZE)
            
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

        [return_MarkowitzCluster,weight] = self.show_return(test_return,weight)
        
        return [weight, return_MarkowitzCluster*(1 - tran_cost)]
    
    def rolling(self, shortsale_sign, cluster_sign, cluster_number, robust_sign):
        i = 0
        pre_weight = np.zeros(self.portfolio_number)
        num_of_sample = len(self.df_select)
        num_of_train = len(self.df_train)
        while i < num_of_sample - num_of_train:   
            train_return = self.df_select[i: i + num_of_train]
            if i + num_of_train + self.rolling_day < len(self.df_select):       
                test_return = np.array(self.df_select[i + num_of_train : i + num_of_train + self.rolling_day])
            else:
                test_return = np.array(self.df_select[i + num_of_train : len(self.df_select)])
            if cluster_number == 1:     
                train_return_mean = np.array(train_return.mean())
                train_return_covar = np.cov(train_return, rowvar = False, ddof = 1)
                self.robust_level(train_return, train_return_mean, train_return_covar)
                [self.weight, self.return_array[i:i + self.rolling_day]] = self.optimize(train_return_mean,train_return_covar,
                                                                                        test_return,self.weight,tran_cost_p, shortsale_sign, robust_sign)
            else:
                factor_data = self.df_factor[i: i + num_of_train]
                if cluster_sign == 1:
                    [cluster_freq, mean_info, cov_info] = self.factor_cluster(train_return, factor_data, self.column_name,cluster_number)
                else:
                    [cluster_freq, mean_info, cov_info] = self.return_cluster(train_return,self.column_name,cluster_number)
                cov_info = cov_info.fillna(0)
                [self.weight, self.return_array[i:i + self.rolling_day]] = self.optimize_cluster(cluster_number, cluster_freq, mean_info, cov_info, test_return,
                                                                                        risk_aversion, self.weight, tran_cost_p, shortsale_sign, robust_sign)
            self.weight = np.array(self.weight)
            i = i + self.rolling_day

