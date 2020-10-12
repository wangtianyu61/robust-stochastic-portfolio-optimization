# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 14:11:17 2019

@author: wangtianyu6162
"""

#FCVaR_no_cluster.py is meant for robust CVaR with no clusters
#by minimizing the worst-case CVaR and evaluate the performance
import pandas as pd
import numpy as np
from CVaR_parameter import *
from gurobipy import *
from method.strategy import *
    
class F_CVaR(strategy):
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
        
        if self.mean_target != False:
            max_value = max(np.array(train_return_mean))
            m.addConstr(np.dot(train_return_mean, weight) >= (1 - self.mean_target*np.sign(max_value))*max_value, "to draw mean-cvar frontier")


        #Solve the Optimization Problem
        m.setParam('OutputFlag',0)
        m.optimize()

        #Retrieve the weight
        #print("The optimal weight portfolio from the Popescu Bound without clusters is :")
        weight = [v.x for v in weight]
        #print(weight)

        #test_return = np.matrix(test_return)
        self.base_line = m.objVal
    
        tran_cost = 0
        for i in range(port_num):
            tran_cost = tran_cost + tran_cost_p*abs(weight[i] - weight_pre[i])
            
        self.turnover = self.turnover + np.sum(abs(weight - weight_pre)) 
        [return_list,weight] = self.show_return(test_return,weight)

        return [weight, return_list*(1 - tran_cost)]
    
    def optimize_adj(self, train_return_mean,train_return_covar,test_return,weight_pre,tran_cost_p, epsilon, shortsale_sign):
        m = Model("Popescu_Bound_no_clusters")
        mu = pd.Series(train_return_mean.tolist())

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
        k = m.addVar(name = 'k', lb = 0, ub = 1)
        m.setObjective(k, GRB.MINIMIZE)
        # Set the constraints:
        m.addConstr(v + (covar - np.dot(mu,weight) - v + tran_cost_p*np.sum(weight_dif))/(2*epsilon) <= (1 + self.adj_level*np.sign(self.base_line))*self.base_line) 
        m.addConstr(S.dot(weight).dot(weight) + (np.dot(mu,weight) + v - tran_cost_p*np.sum(weight_dif))*(np.dot(mu,weight) + v - tran_cost_p*np.sum(weight_dif)) <= covar*covar, "c0")
        m.addConstr(weight.sum() == 1,'budget')
        
        m.addConstrs((weight[i] - weight_pre[i] <= weight_dif[i]    
                    for i in range(port_num)),'abs1')
        m.addConstrs((weight_pre[i] - weight[i] <= weight_dif[i]
                    for i in range(port_num)),'abs2')
        m.addConstrs((k >= weight[i] for i in range(port_num)), 'budget1')
        if shortsale_sign != 0:
            m.addConstrs((k >= -weight[i] for i in range(port_num)), 'budget2')
        m.setParam('OutputFlag',0)
        m.optimize()
        self.weight_opt.append(m.objVal)
        weight = [v.x for v in weight]
        
        tran_cost = 0
        for i in range(port_num):
            tran_cost = tran_cost + tran_cost_p*abs(weight[i] - weight_pre[i])
            
        self.turnover = self.turnover + np.sum(abs(weight - weight_pre)) 
        [return_list,weight] = self.show_return(test_return,weight)

        return [weight, return_list*(1 - tran_cost)]
        
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
            train_return_mean = np.array(train_return.mean())
            train_return_covar = np.cov(train_return, rowvar = False, ddof = 1)
            [self.weight, self.return_array[i:i + self.rolling_day]] = self.optimize(train_return_mean,train_return_covar,test_return,self.weight,tran_cost_p, epsilon, shortsale_sign)
            if type(self.adj_level) != bool:
                [self.weight, self.return_array[i:i + self.rolling_day]] = self.optimize_adj(train_return_mean,train_return_covar,test_return,self.weight,tran_cost_p, epsilon, shortsale_sign)
            self.weight = np.array(self.weight)

            i = i + self.rolling_day
            