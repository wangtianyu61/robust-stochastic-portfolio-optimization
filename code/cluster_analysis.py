# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 21:24:40 2019

@author: wangtianyu6162
"""

import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

#for clustering analysis and visualization
#parameters
portfolio_number = 10
cluster_number = 2
flag = 0
suffix = ""
if flag == 1:
    suffix = " (factor)"

start_time = 20161001
end_time = 20161231

#read the data
csv_name = "trainset_" + str(portfolio_number) + "_" + str(cluster_number) + " clusters" + suffix 
csv_path = "../cluster_tag/" + csv_name + ".csv"
plt_path = "../cluster_tag/" + csv_name + str(start_time) + "-" + str(end_time) + ".png"

df = pd.read_csv(csv_path)
#select to look better
df = df[(df['Date'] >= start_time)&(df['Date'] < end_time)]
tag_cluster = list(df['tag_cluster'])
num_of_date = len(tag_cluster)

plt.scatter(range(num_of_date),tag_cluster,marker = ".")
#if we use the date to illustrate, it is not easy to see clearly.
plt.xlabel("traing time")
plt.ylabel("clustering type")
plt.title(csv_name  + str(start_time) + "-" + str(end_time) )
plt.savefig(plt_path, bbox_inches = 'tight')

