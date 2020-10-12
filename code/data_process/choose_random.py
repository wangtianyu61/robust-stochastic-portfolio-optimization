# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:47:16 2019

@author: wangt
"""

import random
def choose_random(threshold, train_return_mean, train_return_covar, 
                  cluster_freq, cluster_mean, cluster_covariance):
    #a = np.random.multivariate_normal(tuple(train_return_mean),np.array(train_return_covar))
    r1 = random.Random()
    u = r1.uniform(0,1)
    if u < threshold:
        return [train_return_mean,train_return_covar]
    else:
        stop_value = 0
        r2 = random.Random()
        u = r2.uniform(0,1)
        for k in range(len(cluster_freq)):
            stop_value += cluster_freq[k]
            if stop_value > u:
                return [cluster_mean[k],cluster_covariance[k]]