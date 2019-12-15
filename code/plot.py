# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 08:42:04 2019

@author: wangt
"""

from main_head import *
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

from gurobipy import *

#------------------------------------------------#


#choose the dataset and the file path
portfolio_number = str(10)
#different kinds of datasets (10/17/30)

freq = "Daily"
#Daily/Weekly/Monthly

value = ""
#NONE represents average value weighted return
#_eq represents average equal weighted return

filepath = "../factor model/" + portfolio_number + "_Industry_Portfolios_" + freq + value + ".csv"


#print(filepath)

#select part of the data
start_time = 20140101
end_time = 20190930
train_test_split = 20190731

[df_train,df_test,column_name] = data_input(filepath,start_time,end_time,train_test_split)

plot_return(df_train, start_time, train_test_split, column_name)