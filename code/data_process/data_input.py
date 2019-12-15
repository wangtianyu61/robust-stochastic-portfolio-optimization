# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 09:35:05 2019

@author: wangtianyu6162
"""
import pandas as pd
## data_input.py to split the trainset and testset
def data_input(filepath,start_time,end_time,train_test_split):
#read the data
    df = pd.read_csv(filepath)
    #we let the dataset to generate its index

    df_select = df[(df['Date']<end_time)&(df['Date']>=start_time)]
    
    df_select['tag_train'] = df_select.Date.apply(lambda x: 1 if x > train_test_split else 0) 
    #print(df_select.head())

    #tag_train is a indicating sign 
    #if the date is before the first day of train_test_split, then classify into train_data
    #the others are classified into test_data

    #classify df_select according to train or test
    cls_info = df_select['tag_train'].unique()
    #print(cls_info)

    df_train = df_select[df_select['tag_train'].isin([cls_info[0]])]
    df_test = df_select[df_select['tag_train'].isin([cls_info[1]])]
        
    column_name = list(df_train.columns)
    #print(column_name)

    column_name.remove('Date')
    column_name.remove('tag_train')

    #print(column_name)
    return [df_train,df_test,column_name]



