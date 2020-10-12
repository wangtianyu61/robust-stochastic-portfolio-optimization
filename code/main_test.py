#basic module
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from scipy import stats
from hmmlearn import hmm

import time

from gurobipy import *

from main_head import *

#df = pd.read_csv("../factor model/11_Industry_Portfolios_Monthly.csv")
#df_seq = pd.read_csv("../factor model/HMFSS.csv")
##
#remodel = hmm.GaussianHMM(n_components = 2, covariance_type = "full", n_iter = 100)
#dfseq = np.array(list(df_seq['seq'])).reshape(-1, 1)
##hmm_data = np.array(df[df.columns[1:]])
#remodel.fit(dfseq)
#z = remodel.predict(dfseq)
#
##X = np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 1]).reshape(-1, 1)
##model = hmm.GaussianHMM(n_components=2, covariance_type='full')
##model.fit(X)
##
##model.predict(X)
##model.monitor_.history
##
### pi
##model.startprob_
##
### state transform matrix
#print(remodel.transmat_)
##
## emission_matrix
#np.power(np.e, model._compute_log_likelihood(np.unique(X).reshape(-1, 1)))

#choose the dataset and the file path
portfolio_number = 11
#different kinds of datasets (6*/10/17/30/48)
freq = "Monthly"
#Daily/Weekly/Monthly

value = ""

#select part of the data
start_time = 199701
end_time = 201905
train_test_split = 200612
#------------------------------------------------#



#data input
Input_csv = Input(portfolio_number, freq, value, start_time, end_time, train_test_split)
[data_head, data_parameter, csv_name] = Input_csv.parameter_output()
[df_select, df_train] = Input_csv.data_load()

df_factor = Input_csv.three_factor_load()

#df_five_factor = Input_csv.five_factor_load()

if sharpe_ratio_open == False:
    rfr_data = 0
else:
    rfr_data = Input_csv.risk_free_rate()

fcvar_hmm = FCVaR_wasserstein(df_select, df_train, rolling_day, portfolio_number, df_factor, 1, 2, 'test', False, False, True)
fcvar_hmm.rolling(shortsale_sign)
