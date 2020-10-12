# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 16:21:02 2019

@author: wangt
"""

# For the numerical data in Brown and Smith (2011) based on Lynch (2001)
## Three-asset Model with Predictability Monthly Data

import numpy as np
import pandas as pd
import math
import csv
import matplotlib.pyplot as plt

def return_sim(mean, coeff, market_state, sto_value):
    rho = mean + coeff * market_state + sto_value
   # print(market_state,sto_value)
    return math.exp(rho) - 1


data_name = "Lynch2001"
csv_name = "../factor model/" + data_name + ".csv"

#parameter areas
return_mean = [0.0053,0.0067,0.0072,0.0000]
return_coeff = [0.0028,0.0049,0.0061,0.9700]
sto_mean = (0,0,0,0)
sto_covar = np.array([[0.002894,0.003532,0.003910,-0.000115],
                      [0.003532,0.004886,0.005712,-0.000144],
                      [0.003910,0.005712,0.007259,-0.000163],
                      [-0.000115,-0.000144,-0.000163,0.052900]])

month = 120 + 60
temp_return = np.zeros(len(return_mean)-1)
temp_sto = np.zeros(len(return_mean))
market_state = 0
list_x = list()
list_y = list()

csvFile = open(csv_name,'a',newline = '')
writer = csv.writer(csvFile)
writer.writerow(['A','B','C'])

for i in range(month):
    temp_sto = np.random.multivariate_normal(sto_mean, sto_covar)
    for j in range(len(return_mean)-1):
        temp_return[j] = return_sim(return_mean[j],return_coeff[j],market_state,temp_sto[j])
    writer.writerow(temp_return)
    market_state = return_mean[3] + return_coeff[3]*market_state + temp_sto[3]
    list_x.append(temp_return[1])
    list_y.append(temp_return[2])
    #print(temp_return)
#plt.scatter(list_x,list_y)
csvFile.close()

