# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 18:50:14 2020

@author: wangt
"""

#naive_policy.py is meant to give the performance 
#give each portfolio the same weight:1/N
import pandas as pd
import numpy as np
from CVaR_parameter import *
from method.strategy import *
    
class vw_strategy(strategy):
    portfolio_number = 10
    weight = np.zeros(portfolio_number)
    method_name = "strategy"
    def __init__(self, df_select, df_train, rolling_day, portfolio_number, method_name):
        strategy.__init__(self, df_select, df_train, rolling_day, portfolio_number)
        self.method_name = method_name
    
    def optimize(self, train_return_mean, test_return, tran_cost_p):
        (num_of_sample,port_num) = test_return.shape 
        portfolio_weight =  train_return_mean / train_return_mean.sum()
        print("weight",portfolio_weight)
        
        tran_cost = tran_cost_p*np.sum(abs(portfolio_weight - self.weight))

        self.turnover = self.turnover + np.sum(abs(portfolio_weight - self.weight))
        [return_naivepolicy, portfolio_weight] = self.show_return(test_return, portfolio_weight)                
        print("the return of vw is ",return_naivepolicy)
        print("===========================")
        return [portfolio_weight, return_naivepolicy*(1 - tran_cost)]
    
    def rolling(self):
        i = 0
        num_of_sample = len(self.df_select)
        num_of_train = len(self.df_train)
        while i < num_of_sample - num_of_train:
            train_return = self.df_select[i: i + num_of_train]
            if i + num_of_train + self.rolling_day < len(self.df_select):       
                test_return = np.array(self.df_select[i + num_of_train : i + num_of_train + self.rolling_day])
            else:
                test_return = np.array(self.df_select[i + num_of_train : len(self.df_select)])
            train_return_mean = np.array(train_return.mean()) 
            print("mean",train_return_mean)
            [self.weight, self.return_array[i:i + self.rolling_day]] = self.optimize(train_return_mean, test_return, tran_cost_p)
            self.weight = np.array(self.weight)
            i = i + self.rolling_day
