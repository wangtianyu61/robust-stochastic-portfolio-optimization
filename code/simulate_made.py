# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 17:13:30 2019

@author: wangt
"""

import numpy as np
import pandas as pd
import random
import csv
import matplotlib.pyplot as plt

def generate_data(threshold1,threshold2):
    return_mean1 = (0.4, -0.2)
    return_covar1 = np.array([[0.0021, 0],[0, 0.0021]])
    return_mean2 = (-0.2, 0.4)
    return_covar2 = np.array([[0.0021, 0],[0, 0.0021]])
    return_mean3 = (0.4, 0.4)
    return_covar3 = np.array([[0.0021, 0],[0, 0.0021]])
    epsilon_mean = (0, 0)
    epsilon_covar = np.array([[0.000009, 0.000004],[0.000004, 0.000009]])
    
    k = np.random.multivariate_normal(epsilon_mean, epsilon_covar)
    r1 = random.Random()
    u = r1.uniform(0,1)
    if u < threshold1:
        return np.random.multivariate_normal(return_mean1, return_covar1) + k
    else:
        if u < threshold2:
            return np.random.multivariate_normal(return_mean2, return_covar2) + k        
        else:
            return np.random.multivariate_normal(return_mean3, return_covar3) + k      

data_name = "xjbgg"
csv_name =  "result_self_sim/" + data_name + ".csv"
csvFile = open(csv_name,'a',newline = '')
writer = csv.writer(csvFile)
writer.writerow(['A','B'])

threshold1 = 0.33
threshold2 = 0.67
for i in range(400):
    writer.writerow(generate_data(threshold1,threshold2))

csvFile.close()
print('ok')
df_select = pd.read_csv(csv_name)
getA = list(df_select['A'])
getB = list(df_select['B'])
plt.scatter(getA, getB)
plt.show()