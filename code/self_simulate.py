# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 19:04:11 2019

@author: wangt
"""

#for generating data given a multivariate normal distribution
import pandas as pd
import numpy as np
import csv

data_name = "Fan2008"
csv_name = "../factor model/" + data_name + ".csv"
csv_factor_name = "../factor model/" + data_name + "_factor.csv"

def eps():
    k = np.random.gamma(3.3586,0.1876)
    while k<0.195:
        k = np.random.gamma(3.3586,0.1876)
    return k/10

## the area of parameters
num_of_portfolio = 30
factor_load_mean = (0.78282,0.51803,0.41003)
factor_load_covar = np.array([[0.029145,0.023873,0.010184],
                              [0.023873,0.053951,-0.006967],
                              [0.010184,-0.006967,0.086856]])


three_factor_mean = (0.023558, 0.012989, 0.020714)
three_factor_covar = np.array([[1.2507,-0.034999,-0.20419],
                               [-0.034999,0.31564,-0.0022526],
                               [-0.20419,-0.0022526,0.19303]])


#once generated, fix factor_load through the whole simulation process
factor_load  = np.random.multivariate_normal(factor_load_mean, factor_load_covar, num_of_portfolio)

csvFile = open(csv_name,'a',newline = '')
csvFile_factor = open(csv_factor_name,'a',newline = '')

writer = csv.writer(csvFile)
writer_factor = csv.writer(csvFile_factor)

#name the portfolio
portfolio_name = list()
for i in range(num_of_portfolio):
    portfolio_name.append("S" + str(i+1))
writer.writerow(portfolio_name)
writer_factor.writerow(['Mkt-RF','SMB','HML'])

for j in range(750 + 360):
    return_data = np.zeros(num_of_portfolio)
    three_factor = np.random.multivariate_normal(three_factor_mean, three_factor_covar)
    writer_factor.writerow(three_factor)
    for i in range(num_of_portfolio):
        return_data[i] = np.dot(factor_load[i],three_factor) + eps()
    writer.writerow(return_data)
    
#print(return_data)

csvFile_factor.close()
csvFile.close()