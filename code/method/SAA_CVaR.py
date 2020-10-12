# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 17:15:50 2019

@author: wangtianyu6162
"""

#SAA_CVaR.py also sets a benchmark for robust optimization

import pandas as pd
import numpy as np
from CVaR_parameter import *
from gurobipy import *
from method.strategy import *
    
class SAA_CVaR(strategy):
    portfolio_number = 10
    weight = np.zeros(portfolio_number)
    method_name = "strategy"
    def __init__(self, df_select, df_train, rolling_day, portfolio_number, method_name, mean_target = False, adj_level = False):
        strategy.__init__(self, df_select, df_train, rolling_day, portfolio_number)
        self.method_name = method_name
        self.mean_target = mean_target
        self.adj_level = adj_level
        if type(self.adj_level) != bool:
            self.weight_opt = []
            print('yes')
        
    def optimize(self, train_return, test_return, weight_pre, tran_cost_p, shortsale_sign):
        m = Model('SAA_CVaR')

        #Create the Variables
        (train_num,port_num) = train_return.shape
        #print(train_return.iloc[0])
        
        return_weight = pd.Series(m.addVars(train_num)) # - mu*x - v
        if shortsale_sign == 0:
            weight = pd.Series(m.addVars(port_num))
        else:
            weight = pd.Series(m.addVars(port_num, lb = -GRB.INFINITY))
            
        weight_dif = pd.Series(m.addVars(port_num))
        #print(weight)
        v = m.addVar(name = 'v', lb = -GRB.INFINITY, ub = GRB.INFINITY)

        #Set the objective
        obj = v + return_weight.sum()/(epsilon*train_num)
        m.setObjective(obj, GRB.MINIMIZE)
        #Set the constraints

        if self.mean_target != False:
            max_value = max(np.array(train_return.mean()))
            m.addConstr(np.dot(np.array(train_return.mean()), weight) >= (1 - np.sign(max_value)*self.mean_target)*max_value, "to draw mean-cvar frontier")

        m.addConstrs((return_weight[i] >= 0
                      for i in range(train_num)),'nonnegative for positive expectation')
        m.addConstrs((return_weight[i] >=  -np.dot(train_return.iloc[i],weight)+ tran_cost_p*np.sum(weight_dif) - v
                      for i in range(train_num)),'c0')
        m.addConstrs((weight[i] - weight_pre[i] <= weight_dif[i]
                    for i in range(port_num)),'abs1')
        m.addConstrs((weight_pre[i] - weight[i] <= weight_dif[i]
                    for i in range(port_num)),'abs2')
        
        m.addConstr(weight.sum() == 1,'budget')        
        #Solve the Linear Optimization Problem
        m.setParam('OutputFlag',0)
        m.optimize()
        
        #Retrieve the weight
        #print("The optimal weight portfolio from the SAA method is :")
        weight = [v.x for v in weight]

        #print(weight) 
        (num_of_sample,port_num) = test_return.shape
        self.base_line = m.objVal
        
        tran_cost = 0
        for i in range(port_num):
            tran_cost = tran_cost + tran_cost_p*abs(weight[i] - weight_pre[i])
            
        self.turnover = self.turnover + np.sum(abs(weight - weight_pre)) 
        [return_SAACVaR, weight] = self.show_return(test_return,weight)    

        return [weight, return_SAACVaR*(1 - tran_cost)]
    
    def optimize_adj(self, train_return, test_return, weight_pre, tran_cost_p, shortsale_sign):
        m = Model('SAA_CVaR')

        #Create the Variables
        (train_num,port_num) = train_return.shape
        #print(train_return.iloc[0])
        
        return_weight = pd.Series(m.addVars(train_num)) # - mu*x - v
        if shortsale_sign == 0:
            weight = pd.Series(m.addVars(port_num))
        else:
            weight = pd.Series(m.addVars(port_num, lb = -GRB.INFINITY))
            
        weight_dif = pd.Series(m.addVars(port_num))
        #print(weight)
        v = m.addVar(name = 'v', lb = -GRB.INFINITY, ub = GRB.INFINITY)

       ## Set the objective: 
        k = m.addVar(name = 'k', lb = 0, ub = 1)
        m.setObjective(k, GRB.MINIMIZE)
        # Set the constraints:
        m.addConstr(v + return_weight.sum()/(epsilon*train_num) <= (1 + self.adj_level*np.sign(self.base_line))*self.base_line)

        m.addConstrs((return_weight[i] >= 0
                      for i in range(train_num)),'nonnegative for positive expectation')
        m.addConstrs((return_weight[i] >=  -np.dot(train_return.iloc[i],weight)+ tran_cost_p*np.sum(weight_dif) - v
                      for i in range(train_num)),'c0')
        m.addConstrs((weight[i] - weight_pre[i] <= weight_dif[i]
                    for i in range(port_num)),'abs1')
        m.addConstrs((weight_pre[i] - weight[i] <= weight_dif[i]
                    for i in range(port_num)),'abs2')
        
        m.addConstr(weight.sum() == 1,'budget')        
        m.addConstrs((k >= weight[i] for i in range(port_num)), 'budget1')
        if shortsale_sign != 0:
            m.addConstrs((k >= -weight[i] for i in range(port_num)), 'budget2')
        m.setParam('OutputFlag',0)
        m.optimize()
        self.weight_opt.append(m.objVal)
        
        #Retrieve the weight
        #print("The optimal weight portfolio from the SAA method is :")
        weight = [v.x for v in weight]

        #print(weight) 
        (num_of_sample,port_num) = test_return.shape
    
        tran_cost = 0
        for i in range(port_num):
            tran_cost = tran_cost + tran_cost_p*abs(weight[i] - weight_pre[i])
            
        self.turnover = self.turnover + np.sum(abs(weight - weight_pre)) 
        [return_SAACVaR, weight] = self.show_return(test_return,weight)    

        return [weight, return_SAACVaR*(1 - tran_cost)]
    
    def rolling(self, shortsale_sign):
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
            
            [self.weight, self.return_array[i:i + self.rolling_day]] = self.optimize(train_return, test_return, self.weight, tran_cost_p, shortsale_sign)
            if type(self.adj_level) != bool:
                [self.weight, self.return_array[i:i + self.rolling_day]] = self.optimize_adj(train_return, test_return, self.weight, tran_cost_p, shortsale_sign)

            self.weight = np.array(self.weight)
            i = i + self.rolling_day
