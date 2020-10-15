# Document for the Codes
written by Tianyu Wang

updated on 16th Oct, 2020
## Current Functions
1. Given the data fetched from [Fama French website](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html), change the training and test time in the main function, read and split the data.

2. Apply the ```rolling window``` method, which is used commonly in financial data, into the training period for statistics summary such as ```mean``` and ```covariance```. 

3. Construct interfaces of different policies including benchmarks such as ```1/N```, ```min-var``` and ```mean-var``` and our clustering algorithms. Implement each algorithm by the rolling window method.

**Note**
* Our polices of benchmarks implemented include ```1/N```, ```mean-var```, ```min-var```, ```sample average approximation (SAA CVaR)```, ```worst-case CVaR (F-CVaR)```.
* And our policies include ```F-CVaR with clusters``` (different **clustering method** such as ```GMM``` and ```KMeans``` and by different **metrics** such as ```return``` and ```factor data```), ```F-CVaR with clusters (bs)``` (robustness optimization), ```F-CVaR with HMM```, ```F-CVaR with stochastic clusters```(generalize the results in ```F-CVaR with clusters```), ```mean-var/min-var/F-CVaR approximation with clusters```.
* The optimization problems are implemented by Gurobi 8.1.0.

4. Evaluate the result of the output returns in the test data and give different evaluation criteria such as ```empirical_mean```, ```empirical_std```, ```Sharpe Ratio```, ```turnover (transaction costs)```.
   
5. Illustrate the result by figures of the empirical returns.
   
## Notes for document
### main head files

### IO related files
**data_process/input.py**: The input data from arg param to constructing data and split them into different periods (training and testing).

**data_process/test.py**: The output tables (to ```.csv``` file) and figures (to ```.png```).

**show_result.py**: The output figures illustrated for better showing.
### Method files
The base model is shown below:

**method/strategy.py**: ```strategy``` and ```strategy_cluster``` class are designed for basic algorithms and algorithms with clusters including data-driven approaches such as clusters and hmm models.

The other concrete algorithms are derivatives of those py files.
#### Benchmarks
**method/naive_policy.py**: 1/N policy.

**method/Markowitz.py**: mean-var policy and its clustering approaches even with approximation.

**method/MVP.py**: min-var policy and its clustering approaches even with approximation.

**method/SAA_CVaR.py**:sample average approximation for CVaR given the training return data.

**method/FCVaR_no_cluster.py**: worst-case CVaR given the training return data.
#### Our Policies
**method/FCVaR_cluster.py**: worst-case CVaR given the training return data with different clusters.

**method/FCVaR_side_cluster.py**: worst-case CVaR with different clusters with predictive analytics given the future side information. It reduces to a single cluster problem with smaller ambiguity set.

**method/FCVaR_framework.py**: worst-case CVaR with general ambiguity sets with clusters.

**method/FCVaR_cluster_bs.py**: worst-case CVaR with clusters by transformation to the robustness optimization.

**method/FCVaR_wasserstein.py**: worst-case CVaR with HMM estimation.

**method/FCVaR_approximate.py**: worst-case CVaR with clusters given Sharpe-Ratio approximation.

### head files
**main_head.py**: include all the classes in each header of the main function which under the ```data_process``` and ```method``` folder.

**CVaR_parameter.py**: include the default parameters set in the files.
### main files specified for different data sets.
**main_rolling_tran.py**: the current classical file used in our algorithm in the website.

**main_china.py**: result given by IPCAnet data.

**main_yahoo.py**: result given by Yahoo! data.

**main_rolling_cls.py**: cross-validation and model selection by chosing the best approach with the minimum worst-case CVaR.

**main_rolling_saa.py**: simulation result by the robustness optimization given ```F-CVaR```, ```SAA-CVaR``` and ```F-CVaR with clusters```.

**main_sim.py**:simulation within the synthetic data which is in ```self_simulate.py```.

**main_test.py**: for current test given the FCVaR with cluster method with Wasserstein distance.
## Basic Results

## Reference
Markowitz, H. M. (1952). Portfolio Selection. Journal of Finance 7:77–91.

Popescu, Ioana. 2007. Robust mean-covariance solutions for stochastic optimization. Operations Research 55(1) 98–112

DeMiguel, Victor, Lorenzo Garlappi, Raman Uppal. 2007. Optimal versus naive diversification: How inefficient is the 1/n portfolio strategy? The review of Financial studies 22(5) 1915–1953.

Chen, Zhi, Melvyn Sim, Peng Xiong. 2020. Robust Stochastic Optimization. To forthcoming in Management Science.

Gao, Rui, and Anton J. Kleywegt. Data-driven robust optimization with known marginal distributions. Working paper, 2017.

Fan J, Fan Y, Lv J (2008) High dimensional covariance matrix estimation using a factor model. Journal of Econometrics 147(1): 186-197
