# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 01:47:22 2020

@author: wangtianyu6162
"""
import pandas as pd
import numpy as np
import math
from CVaR_parameter import *
from gurobipy import *
from method.strategy import *

#The paper shown in Zhenzhen.
class FCVaR_wasserstein(strategy, strategy_cluster):
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
        
        #estimate and get the state of hmm model
        
    def optimize(self, cluster_freq, num_time_in_cluster, train_return, test_return, weight_pre,tran_cost_p, shortsale_sign):
        #the variable of train return is a 3-d array, cluster * cluster_index * portfolio_dim
        #data preprocessing    
        (num_of_sample, port_num) = test_return.shape
        
        # preprocessing the data to get the num_time_in_cluster
        ## cluster frequency means the frequency of the transition probability from the final datapoint
        cluster_freq = num_time_in_cluster/sum(num_time_in_cluster)
        # other parameter
        theta = 0.05
        
        # create a model
        m = Model("HMM_2020")
        
        # Create variables
        v = m.addVar(name = 'v', lb = -GRB.INFINITY, ub = GRB.INFINITY)
        if shortsale_sign == 0:
            weight = pd.Series(m.addVars(port_num))
        else:
            weight = pd.Series(m.addVars(port_num, lb = -GRB.INFINITY))
        
        #initiation for the auxiliary variables
        aux_a = []
        aux_b = [[] for i in range(self.cluster_number)]
        aux_c = []
        aux_alpha = []
        num_train = 120
        for i in range(self.cluster_number):
            aux_a.append(pd.Series(m.addVars(num_train, lb = -GRB.INFINITY)))
            
            for cls_index in range(num_time_in_cluster[i]):
                aux_b[i].append(pd.Series(m.addVars(port_num, lb = -GRB.INFINITY)))
            #budget constraint for c, "c6"
            aux_c.append(pd.Series(m.addVars(num_train, lb = 0)))
            aux_alpha.append(pd.Series(m.addVars(num_train, lb = -GRB.INFINITY)))
        
        aux_beta = pd.Series(m.addVars(self.cluster_number, lb = -GRB.INFINITY))
        obj = v
        for i in range(self.cluster_number):
            #inner loop of the target
            target_inside = 0
            for cls_index in range(num_time_in_cluster[i]):
                target_inside += cluster_freq[i]/num_time_in_cluster[i]*aux_a[i][cls_index]
            obj += 1/(1 - epsilon)*(target_inside + aux_beta[i]*cluster_freq[i]*theta)
            
        # Set the constraints:
        ## aux constraint
        print(train_return[0][0], aux_b[0][0])
        for i in range(self.cluster_number):
            m.addConstrs((-np.dot(train_return[i][cls_index], aux_b[i][cls_index]) >= aux_a[i][cls_index] - aux_alpha[i][cls_index]
                        for cls_index in range(num_time_in_cluster[i])), "c1")
            m.addConstrs((aux_beta[i] - aux_c[i][cls_index] >= 0 
                        for cls_index in range(num_time_in_cluster[i])), "c2")

            m.addConstrs((aux_beta[i] - aux_c[i][cls_index] >= aux_b[i][cls_index][port_index]
                        for port_index in range(port_num) for cls_index in range(num_time_in_cluster[i])), "c3")

            m.addConstrs((aux_beta[i] - aux_c[i][cls_index] >= -aux_b[i][cls_index][port_index]
                        for port_index in range(port_num) for cls_index in range(num_time_in_cluster[i])), "c4")

            m.addConstrs((np.dot(train_return[i][cls_index], aux_b[i][cls_index]) + np.dot(train_return[i][cls_index], weight) >= -aux_a[i][cls_index] - v
                        for cls_index in range(num_time_in_cluster[i])), "c5")
            m.addConstrs((aux_c[i][cls_index] >= aux_b[i][cls_index][port_index] + weight[port_index]
                            for port_index in range(port_num) for cls_index in range(num_time_in_cluster[i])), "c7")
            m.addConstrs((aux_c[i][cls_index] >= -aux_b[i][cls_index][port_index] - weight[port_index]
                            for port_index in range(port_num) for cls_index in range(num_time_in_cluster[i])), "c8")
            m.addConstrs((np.dot(train_return[i][cls_index], aux_b[i][cls_index]) >= -aux_a[i][cls_index]
                            for cls_index in range(num_time_in_cluster[i])), "c9")
            m.addConstrs((aux_c[i][cls_index] >= aux_b[i][cls_index][port_index]
                            for port_index in range(port_num) for cls_index in range(num_time_in_cluster[i])), "c10")
            m.addConstrs((aux_c[i][cls_index] >= -aux_b[i][cls_index][port_index]
                            for port_index in range(port_num) for cls_index in range(num_time_in_cluster[i])), "c11")
        ## budget constraint
        
        m.addConstr(weight.sum() == 1,'budget')

    
        m.setObjective(obj, GRB.MINIMIZE)
        #Solve the Optimization Problem
        m.setParam('OutputFlag',0)
        m.optimize()         
        print(m.status)
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
        
    def rolling(self, shortsale_sign):
        i = 0
        num_of_sample = len(self.df_select)
        num_of_train = len(self.df_train)
        pre_info = 0
        while i < num_of_sample - num_of_train:
            
            train_return = np.array(self.df_select[i: i + num_of_train])
            
            if i + num_of_train + self.rolling_day < len(self.df_select):       
                test_return = np.array(self.df_select[i + num_of_train : i + num_of_train + self.rolling_day][0: self.portfolio_number])
            else:
                test_return = np.array(self.df_select[i + num_of_train : len(self.df_select)][0: self.portfolio_number])

            # hmm estimate
            ## num_time_in_cluster shows the history cluster and cluster_freq shows the transitin probability for the last data point
            [num_time_in_cluster, cluster_freq, train_return] = self.hmm_train(train_return, i, self.cluster_number)
                
            print(train_return)
            [self.weight, self.return_array[i:i + self.rolling_day]] = self.optimize(cluster_freq, num_time_in_cluster, train_return, test_return, self.weight, tran_cost_p, shortsale_sign)
            
            self.weight = np.array(self.weight)
            i = i + self.rolling_day
