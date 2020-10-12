# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 18:50:04 2019

@author: wangtianyu6162
"""

#test_result.py is meant for outputing the results of return and criterion for comparison
#it is also for visualization
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from scipy import stats
import datetime, math
from method.FCVaR_cluster import *
from CVaR_parameter import *

def return_to_matlab(method_list, return_list):
    for i in range(len(return_list)):
        csvpath = "../compute_sharperatio/" + method_list[i] + ".csv"
        csvFile = open(csvpath, 'a', newline = '')
        for j in range(len(return_list[i])):
            writer = csv.writer(csvFile)
            writer.writerow([return_list[i][j]])
        csvFile.close()
        
def plt_return (method_name, return_policy):
    #visualize the return distributions in the test_sample
    return_list = list(return_policy)
    #change the form to plot
    num_of_sample = len(return_list)
    print("The number of test sample in the plot",num_of_sample)
    plt.plot(range(num_of_sample),return_list)
    #if we use the data to illustrate, it is not easy to see clearly.
    plt.xlabel("testing time")
    plt.ylabel("actual return")
    plt.title(method_name)
    plt.figure(dpi = 1000)
    plt.show()
    plt.savefig("return.jpg", dpi = 2000)

def plt_return_tran (method_list,return_list): 

    plt.figure(figsize = (10,5))
    
    for i in range(len(return_list)):
        base = 1
        return_val = list()
        return_val.append(base)
    
        for j in range(len(return_list[i])):            
            base = base*(1 + return_list[i][j]/100)
            return_val.append(base)
        
        plt.plot(range(j + 2),return_val,label = method_list[i])
        #if we use the date to illustrate, it is not easy to see clearly.
        
        plt.xlabel("test time")
        plt.ylabel("real return")
        plt.legend()
        plt.title("Return under different policies (c = " + str(tran_cost_p) + ")")
    
    plt.show()

def print_return(method_list, return_list):
    csvFile = open("return_list3.csv",'a',newline = '')
    writer = csv.writer(csvFile)
    
    for i in range(len(return_list)):
        base = 1
        return_val = list()
        return_val.append(base)
        writer.writerow([method_list[i]])
        for j in range(len(return_list[i])):            
            base = base*(1 + return_list[i][j]/100)
            return_val.append(base)
            writer.writerow([base])
    csvFile.close()

def empirical_mean_cvar(return_list):
    stat_mean = []
    stat_cvar = []
    for item in return_list:
        VaR = -np.percentile(item, 100*epsilon)
        CVaR = np.zeros(len(item))
        count_CVaR = 0
        for i in range(len(item)):
            if item[i] <-VaR:
                count_CVaR = count_CVaR + 1
                CVaR[i] = -item[i]
        stat_mean.append(np.mean(item))
        stat_cvar.append(np.mean(CVaR)/(count_CVaR/len(item)))
    return [stat_mean, stat_cvar]

class output():
    def __init__(self, csv_name, data_head, data_parameter, base_return):
        self.csv_name = csv_name
        self.data_parameter = data_parameter
        self.base_return = base_return
        self.data_head = data_head
    def head(self):
        csvFile = open(self.csv_name,'a',newline = '')
        writer = csv.writer(csvFile)
        #output some specific parameters
        writer.writerow(["tran_cost","shortsale","rolling_day","sharpe_ratio_open","cluster_type"])
        writer.writerow([str(tran_cost_p),str(shortsale_sign), str(rolling_day),str(sharpe_ratio_open),cluster_type])
        
        #output the statistic information of this case
        stat_info = ["method","number of sample","mean","std","Sharper ratio","VaR","CVaR","Turnover"]
        writer.writerow(self.data_head + stat_info)
        print('done')
        csvFile.close()
        
    def return_info(self, strategy, rfr_data = 0):
        # whether to delete the risk free rate in our case
        return_policy = strategy.return_array - rfr_data
        
        ## the result of one specific policy return
        csvFile = open(self.csv_name,'a',newline = '')
        writer = csv.writer(csvFile)
        
        num_of_sample = int(len(return_policy))
        ## compute the sample mean and variance of one constructed portfolio
        re_mean = np.mean(return_policy)
        re_std = np.std(return_policy, ddof = 1)
    
        ## other measures of risks: sharpe ratio
        Sharpe_ratio = re_mean / re_std
        ## risk measures: var and cvar
        VaR = - np.percentile(return_policy,100*epsilon)
        CVaR = np.zeros(num_of_sample)
        count_CVaR = 0
        for i in range(num_of_sample):
            if return_policy[i]< - VaR:
                count_CVaR = count_CVaR + 1
                CVaR[i] = - return_policy[i]
                
        #Here we just set the benchmark as risk_free_rate, in case not sufficient test samples            
    
        stat_info = [strategy.method_name,num_of_sample,re_mean,re_std, Sharpe_ratio, VaR, np.mean(CVaR)/(count_CVaR/num_of_sample), (strategy.turnover-1)/(num_of_sample-1)]
        writer.writerow(self.data_parameter + stat_info)
        csvFile.close()        
        
    def pvalue_sharperatio(self, return_policy):
        #for computing pvalue of sharperatio with long tail
        base_mean = np.mean(self.base_return)
        base_mean2 = np.mean(self.base_return**2)
        re_mean = np.mean(return_policy)
        re_mean2 = np.mean(return_policy**2)
        base_std = np.std(self.base_return, ddof = 1)
        re_std = np.std(return_policy, ddof = 1)
        sharpe_dif = abs(base_mean/base_std - re_mean/re_std)
        stat_vector = np.array([[base_mean2/math.pow(base_mean2 - base_mean**2,1.5),
                                 -re_mean2/math.pow(re_mean2 - re_mean**2,1.5),
                                 -0.5*base_mean/math.pow(base_mean2 - base_mean**2, 1.5),
                                 0.5*re_mean/math.pow(re_mean2 - re_mean**2,1.5)]])
    
        Phi = 0
        for i in range(int(resample_number/block_size)):
            base_resample = np.random.choice(self.base_return, size = block_size, replace = True, p = None)
            re_resample = np.random.choice(return_policy, size = block_size, replace = True, p = None)
            base_vector = np.sum(np.array([[base_resample[j] - base_mean, re_resample[j] - re_mean, base_resample[j]**2 - base_mean2, re_resample[j]**2 - re_mean2]]) 
                                for j in range(block_size))/math.sqrt(block_size)
            Phi = Phi + np.dot(base_vector.T, base_vector)
        bootstrap_se = math.sqrt(np.dot(stat_vector, np.dot(Phi, stat_vector.T))/(resample_number**2/block_size))
        return stats.norm.sf(sharpe_dif/bootstrap_se)*2
    
    def pvalue(self, return_policy):
        csvFile = open(self.csv_name,'a',newline = '')
        writer = csv.writer(csvFile)
        num_of_sample = int(len(return_policy))
        # basic mean and variance information of two different policies
        base_mean = np.mean(self.base_return)
        base_std = np.std(self.base_return, ddof = 1)
        re_mean = np.mean(return_policy)
        re_std = np.std(return_policy, ddof = 1)
        
        #p-value for p_mean, we use the t-distribution here
        t_mean = (re_mean - base_mean)/np.sqrt((base_std**2 + re_std**2)/num_of_sample)
        num_of_freedom = ((base_std**2 + re_std**2)/num_of_sample)**2/((base_std**4 + re_std**4)/(num_of_sample**2*(num_of_sample - 1)))
        p_mean = stats.t.sf(np.abs(t_mean), int(num_of_freedom))*2
        
        #p-value for p_variance, we use the f-distribution here
        p_variance = stats.f.sf(base_std**2/(re_std**2), num_of_sample - 1, num_of_sample - 1)*2
        if p_variance >1:
            p_variance = 2 - p_variance
        #or we can use the bootstrapping methods
        
        #p-value for p_sp
        # method 1: we assume i.i.d normal distribution for portfolios in this case
        dis_covar = np.cov(self.base_return, return_policy)[0][1]
        theta = (2*re_std**2*base_std**2 - 2*re_std*base_std*dis_covar + 0.5*re_std**2*base_mean**2 + 0.5*re_mean**2*base_std**2 - 
                 dis_covar**2*re_mean*base_mean/(base_std*re_std))/num_of_sample
        z_value = (re_mean*base_std - re_std*base_mean)/np.sqrt(theta)
        #print(dis_covar, theta, z_value)
        p_sp = stats.norm.sf(np.abs(z_value))*2
        
        ## method2: asset distribution with long tail
        #p_sp = self.pvalue_sharperatio(return_policy)
        
        ## method 2: general bootstrap methods for studentize distribution (long tail)
        
        empty = ["" for i in range(len(self.data_parameter) + 2)]
        p_info = ["["+str(round(p_mean,3))+"]", "["+str(round(p_variance,3))+"]", "["+ str(round(p_sp,3))+"]"]
        writer.writerow(empty + p_info)
        csvFile.close()
        
    def tail(self):
        csvFile = open(self.csv_name,'a',newline = '')
        writer = csv.writer(csvFile)
        writer.writerow([datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        csvFile.close()        


def just_output_CVaR (return_policy):
    VaR = - np.percentile(return_policy,100*epsilon)
    num_of_sample = int(len(return_policy))
    count_CVaR = 0
    CVaR = np.zeros(num_of_sample)
    for i in range(num_of_sample):
        if return_policy[i] < - VaR:
            count_CVaR = count_CVaR + 1
            CVaR[i] = - return_policy[i]
    return CVaR.mean()

    #p_value for p_cvar
#    base_VaR = -np.percentile(base_return, 100*epsilon)
#    re_VaR = -np.percentile(return_policy, 100*epsilon)
#    base_CVaR = np.zeros(int(num_of_sample*epsilon)+1)
#    re_CVaR = np.zeros(int(num_of_sample*epsilon)+1)
#    k = j = 0    
#    for i in range(num_of_sample):
#        if return_policy[i] < -re_VaR:
#            re_CVaR[k] = - return_policy[i]
#            k = k + 1
#        if base_return[i] < -base_VaR:
#            base_CVaR[j] = - return_policy[i]
#            j = j + 1
#    base_CVaR_mean = np.mean(base_CVaR)
#    base_CVaR_std = np.std(base_CVaR, ddof = 1)
#    re_CVaR_mean = np.mean(re_CVaR)
#    re_CVaR_std = np.std(re_CVaR, ddof = 1)
#    t_cvar = (re_CVaR_mean - base_CVaR_mean)/np.sqrt((base_CVaR_std**2 + re_CVaR_std**2)/k)
#    p_cvar = stats.t.sf(np.abs(t_cvar), k - 1)*2
    

    
    
def cross_validation(df_train, split_percent, portfolio_number, df_factor, cluster_sign, shortsale_sign):
    num_of_sample = len(df_train)
    df_traintrain = df_train[0:int(num_of_sample*split_percent)]
    sharpe_ratio_list = np.zeros(4)
    for cluster_number in [1,2,3,4]:
        method_name = "FCVaR (" + str(cluster_number) + " cls, return)"
        fcvar_cluster = FCVaR_cluster(df_train, df_traintrain, 20, portfolio_number, df_factor, 0, cluster_number, method_name)
        fcvar_cluster.rolling(shortsale_sign)    
        sharpe_ratio_list[cluster_number-1] = np.mean(fcvar_cluster.return_array)/np.std(fcvar_cluster.return_array, ddof = 1)
    return np.argmax(sharpe_ratio_list)