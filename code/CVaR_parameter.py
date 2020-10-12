# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 17:23:13 2019

@author: wangtianyu6162
"""
#CVaR_parameter.py
#store some risk parameters
epsilon = 0.05
## epsilon is for fixing the parameters of problems regarding cvar
 
target_rate = 2.5
threshold = 0.5
## When we compare the robust form of fcvar_cluster with robustness optimization form
## threshold is a number we use in binary search.
## target_rate is the right-hand side of constraint 


risk_aversion = 1
## the gamma in mean_var model. 
## When gamma becomes infinity, the mean_var will tend to min-var model


tran_cost_p = 0.0
## we include the linear propotional transaction costs with MAD between two weights
## Applications: all target functions containing mean vector and return results with transaction costs   

rolling_day = 1
## for the whole rolling approach, the number of days we consider in each period

shortsale_sign = 0
## for optimization models whether we include the shortsale constraints

sharpe_ratio_open = False
## whether we take the risk-free rate into account, true means we need to minus the risk-free rate and false measn not

cluster_type = "KMeans"
## choose which type to make the cluster in choosing the ambiguity set

## below two are parameters for bootstrap methods
resample_number = 1000
block_size = 5


##
validation_period = 1

