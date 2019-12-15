# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 10:02:51 2019

@author: wangtianyu6162
"""
#data_stat.py is to give some statistics info about the dataset for the optimization

import numpy as np
import matplotlib.pyplot as plt

def plot_return(df_train, start_time, train_test_split, column_name):
    return_base = 1
    count = 0
    #print(column_name)
    for industry_name in column_name:
        #count = count + 1
        return_list = list()
        
        plt_name = industry_name + " from " + str(start_time) + " to " + str(train_test_split)
        plt_path = "../plot/" + plt_name + ".png"
        
        return_daily = list(df_train[industry_name])
        num_of_date = len(return_daily)
        real_return = return_base
        return_list.append(1)
        for i in range(num_of_date):
            real_return = real_return * (1 + return_daily[i]/100)
            return_list.append(real_return)
    plt.plot(range(num_of_date + 1),return_list)
        #if we use the date to illustrate, it is not easy to see clearly.
    plt.xlabel("training time")
    plt.ylabel("real return")
    plt.title(plt_name)
        #plt.show()
    plt.savefig(plt_path, bbox_inches = 'tight')
        
        # to clear the plot background
    plt.clf()
        
def data_stat(df_train,df_test,column_name):
    #to realize the functions of data_input.py

    train_return = df_train[column_name]
    test_return = df_test[column_name]
    #output the information of each data

    train_return_mean = np.array(train_return.mean())
    #print(train_return_mean)
    #print(len(train_return_mean))
    train_return_covar = np.matrix(train_return.cov())
    #print(train_return_covar)
    return [train_return,test_return,train_return_mean,train_return_covar]



