# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 18:18:03 2020

@author: wangtianyu6162
"""

import numpy as np
from numpy import linalg
from scipy.linalg import sqrtm
import pandas as pd
import math
from CVaR_parameter import *
from gurobipy import *

class resample:
    #for single cluster
    #the value of sample epsilon
    gamma1 = 0
    gamma2 = 1
    epsilon = 1 - math.sqrt(1 - epsilon)
    
    #the variable of robust_level controls the robustness of the support of the distribution.
    robust_param = 2
    
    def __init__(self, portfolio_number):
        self.portfolio_number = portfolio_number
    def robust_level(self, df_train, sample_mean, sample_covariance):
    #input data estimated from sample
        df_train = np.array(df_train)
        self.sample_number = len(df_train)
        sample_R = 0
        #support of the whole number of portfolios    
        v, Q = linalg.eig(sample_covariance)
        V = np.diag(v**(-0.5))
        #\Sigma power -0.5
        cov_neg_half = Q * V * (Q**(-1))
        
        for i in range(self.sample_number):
            sample_vector = np.dot(cov_neg_half, df_train[i] - sample_mean)*self.robust_param
            sample_R = max(sample_R, linalg.norm(sample_vector, ord = 2))
        sample_R = sample_R/100
       # print(sample_R)
        #COMPUTE THE SUPPORT OF SAMPLE
        #R_bar = (1 - (sample_R**2 + 2)*(2 + math.sqrt(2*math.log(4/self.epsilon)))/math.sqrt(self.sample_number))
        #**(-0.5)*sample_R
        #print(R_bar)
        alpha = 0 
        beta = 0 
        
        alpha = sample_R**2/math.sqrt(self.sample_number)*(math.sqrt(math.log(4/self.epsilon)) + 1)
        beta = sample_R**2/self.sample_number*(2 + math.sqrt(2*math.log(2/self.epsilon)))**2
        #compute the confidential level
        self.gamma1 = beta/(1 - alpha - beta)
        self.gamma2 = (1 + beta)/(1 - alpha - beta)        
    
    def robust_level_cluster(self, df_train, cluster_freq, mean_info, cov_info):
        cluster_number = len(mean_info)
        #parameters of alpha and beta
        alpha = np.zeros(cluster_number)
        beta = np.zeros(cluster_number)
        number_in_cluster = len(df_train)*cluster_freq
        #show the number of samples in each cluster
        
        mean_info = np.array(mean_info)
        covar = [cov_info.iloc[i*self.portfolio_number:(i+1)*self.portfolio_number] for i in range(cluster_number)]
        sample_R = np.zeros(cluster_number)
        #support of the number of portfolios for each cluster
        for i in range(cluster_number):
            if number_in_cluster[i]<=1.1:
                sample_R[i] = 0
                break
            #in case the cluster has only one member, we just assume it is the correct value of sample
            else:
                
                Q = linalg.inv(np.array(covar[i]))
                
                #print(v, Q)
                #V = np.diag(v**(-0.5))
                cov_neg_half = sqrtm(Q)
                #print(cov_neg_half)
                for j in list(range(cluster_number)):
                #use the HALF VECTOR of centers diff as the approximate boundary 
                    if j!=i:
                        boundary = (mean_info[i] - mean_info[j])*self.robust_param/2
                        
                        sample_vector = np.dot(cov_neg_half, boundary)
                        if sample_R[i] == 0:
                            sample_R[i] = linalg.norm(sample_vector, ord = 2)
                        else:    
                            sample_R[i] = min(sample_R[i], linalg.norm(sample_vector, ord = 2))
        sample_R = sample_R/100
        
        alpha = [sample_R[i]**2/math.sqrt(number_in_cluster[i])*math.sqrt(math.log(4/self.epsilon) + 1) for i in range(cluster_number)]
        beta = [sample_R[i]**2/number_in_cluster[i]*(2 + math.sqrt(2*math.log(2/self.epsilon)))**2 for i in range(cluster_number)]
        print(sample_R, number_in_cluster)
        print(alpha, beta)
        #compute the confidence level
        self.gamma1 = beta/(np.ones(cluster_number) - alpha - beta)
        self.gamma2 = (np.ones(cluster_number) + beta)/(np.ones(cluster_number) - alpha - beta)        

        
        