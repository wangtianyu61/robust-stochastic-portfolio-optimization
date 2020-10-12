# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 16:28:34 2019

@author: wangtianyu6162
"""
# factor_cluster
## to import data of three factors and thus use the factor_cluster to classify the data
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
class Input():
    portfolio_number = 10
    freq = "Daily"
    value = ""
    start_time = 20120101
    end_time = 20180101
    train_test_split = 20161231
    
    def __init__(self, portfolio_number, freq, value, start_time, end_time, train_test_split):
        self.portfolio_number = portfolio_number
        self.freq = freq
        self.value = value
        self.start_time = start_time
        self.end_time = end_time
        self.train_test_split = train_test_split
        self.filepath = "../factor model/" + str(self.portfolio_number) + "_Industry_Portfolios_" + self.freq + self.value + ".csv"
        
    def parameter_output(self):
        
        data_name = str(self.portfolio_number) + self.freq + self.value +""
        #print(filepath)
        data_head = ["filepath","start_time","train_test_split","end_time"]
        data_parameter = [self.filepath,self.start_time,self.train_test_split,self.end_time]
        #for writing to csv

        csv_name = "result_tran/rolling_tran_ " + data_name + "_v0932.csv"
        return [data_head, data_parameter, csv_name]
    
    def data_load(self):
        df = pd.read_csv(self.filepath)
        #we let the dataset to generate its index
        #we choose the right data without dates information
        df_select = df[(df['Date']<self.end_time)&(df['Date']>=self.start_time)]
        column_name = list(df_select.columns)
        column_name.remove('Date')
        #initial training set is below
        df_train = df_select[(df_select['Date']<= self.train_test_split)]
        
        #remove the data index
        df_select = df_select[column_name]
        df_train = df_train[column_name]
        return [df_select, df_train]
        
    def risk_free_rate(self):
        filepath_factor = "../factor model/F_F_Research_Data_Factors_" + self.freq + ".csv"
        #this is the csv path and freq is the same as data_input.py for convenience
        df_rfr = pd.read_csv(filepath_factor)
        df_rfr = df_rfr[((df_rfr['Date'])>self.train_test_split)&((df_rfr['Date'])<self.end_time)]
    
        rfr_data = np.array(df_rfr['RF'])
        return rfr_data


    
    def three_factor_load(self):
        three_factor = ['Mkt-RF','SMB','HML']
        filepath_factor = "../factor model/F_F_Research_Data_Factors_" + self.freq + ".csv"
        #this is the csv path and freq is the same as data_input.py for convenience

        #print(filepath_factor)
        #read and data and generate its own index
        df_factor = pd.read_csv(filepath_factor)
        #df_factor.info()
        #print(df_factor.head())

        df_factor = df_factor[(df_factor['Date']<self.end_time)&(df_factor['Date']>=self.start_time)]
        #print(df_factor.head())
        
        factor_data = df_factor[three_factor]
        #plt.scatter(list(df_factor['Mkt-RF']),list(df_factor['HML']),s = 10)
        #choose the data to classify
        return factor_data
    
    def five_factor_load(self):
        five_factor = ['Mkt-RF','SMB','HML','RMW','CMA']
        filepath_factor = "../factor model/F_F_Research_Data_5Factors_" + self.freq + ".csv"
        #this is the csv path and freq is the same as data_input.py for convenience

        #print(filepath_factor)
        #read and data and generate its own index
        df_factor = pd.read_csv(filepath_factor)
        #df_factor.info()
        #print(df_factor.head())
        
        df_factor = df_factor[(df_factor['Date']<=self.end_time)&(df_factor['Date']>=self.start_time)]
        #print(df_factor.head())
    
        factor_data = df_factor[five_factor]
        #plt.scatter(list(df_factor['Mkt-RF']),list(df_factor['HML']),s = 10)
        #choose the data to classify
        
        return factor_data


