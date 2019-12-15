# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 15:22:41 2019

@author: wangtianyu6162
"""

# main_head.py is meant for importing modules
#import sys
#sys.path.append('method/')
##comment on because of no use


# function for data_processing
from data_process import *
from data_process.data_input import data_input
from data_process.data_stat import *
# functions for statistics and clustering
from data_process.factor_cluster import *
from data_process.cluster import return_cluster

#different optimization problems
## parameters
from CVaR_parameter import *


# method
## benchmark

### whole portfolio benchmark
from method.naive_policy import *
from method.Markowitz import *
from method.Markowitz_revised import *
from method.SAA_CVaR import *


#Markowitz, Markowitz_revised, SAA_CVaR, FCVaR_no_cluster
### include naive policy, Markowitz, Markowitz_revised, SAA_CVaR, Popescu_no_cluster

## methods we suggest
from method.FCVaR_no_cluster import *
from method.FCVaR_cluster import *
from method.FCVaR_cluster_bs import *
## Popescu_cluster
## Popescu_cluster_bs

#outputting the result
from data_process.test_result import *