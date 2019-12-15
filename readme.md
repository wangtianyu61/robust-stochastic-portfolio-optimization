# Document for the Codes
written by Tianyu Wang 

updated on 15th Dec., 2019

## 目前实现功能
1.给定任一规范的Fama-French网站上的portfolio.csv(daily/monthly)，在主函数调整训练集和测试集时间，进行数据读取与划分。

2.统计训练集相关统计信息（均值/协方差），并可按照要求根据three factor/five factor/portfolio_return运用k_means方法聚类。

3.根据统计信息，连接gurobi软件，运用不同方法选取最优投资组合。并给出每种方法在test集上return的mean和std，进而画出收益分布图。

4.目前能够实现的策略有naive policy(1/N), Markowitz, Markowitz_revised, SAA_CVaR, FCVaR_no_cluster, FCVaR_cluster, FCVaR_cluster_bs等

5.将结果并进行处理，给出不同策略针对不同数据集的test_mean, std, Sharper ratio, Loss_Probability, VaR, CVaR供比较.

6.在cluster_tag文件夹里有带标签的聚类结果csv和可视化输出，标签保存在tag_cluster列。

7.在plot文件夹里有相关训练集不同portfolio不同时间的单个/全部的可视化输出。


=========================

update: 更新了关于target rate /risk_aversion等不同参数的敏感性分析和五因子接口。

update: 加入带有transaction cost的结果。

update: 加入可以自生成仿真数据的代码。

note: FCVaR_cluster_bs.py没有写相关调整show_return.py的部分；并且5因子的月度数据没有加进来。


## 文件说明
目前共有23个已完成的.py文件，主要分为以下三类：
### 头文件
#### main_head.py
给main.py import所有其他自己设定的函数。


#### CVaR_parameter.py
包含部分策略初始化的参数。
epsilon: SAA_CVaR / FCVaR_no_cluster / FCVaR_cluster

target_rate: FCVaR_cluster_bs 

risk_aversion: Markowitz_revised

### IO相关文件
#### main.py (155 lines
终端控制，目前可以改变的参数有：选取的数据集(number of portfolio/data frequency), 时间划分(start_time/end_time/train_test_split).

#### main_rolling.py
大体同main_head.py，其中样本选取方式为rolling window sample的方式。且不需要调用data_input, data_stat函数。但是运行较慢。

#### main_target.py
采用The Dao of Robustness (2019)里面的方法进行测试。即使用作为1/N policy的target rate作为tau，然后Minimize \epsilon。
结果不够理想，可能是因为worst-case CVaR还是过于保守。

#### main_rolling_tran.py
加入transaction_cost进入分析，tran_cost的参数依然储存在CVaR_parameter.py里。
【没有完全写完整】

#### main_self_sim.py
对自己在self_simulate.py生成的数据，按照main_rolling_tran.py的方法分析。

#### data_input.py
读取给定输入参数的portfolio.csv, 并进行训练集测试集的划分和数据类型的处理。

#### test_result.py
主要包含两类函数：一是数据收益的可视化处理；二是输出表头、表尾分割和各个策略的数据处理与输出。

### 数据处理 （位于data_process文件夹中）
#### data_stat.py
统计df_train的mean和covar信息。

#### cluster.py
按照cluster_number对df_train进行k_means聚类，并打标签。

#### factor_cluster.py
按照cluster_number对与df_train相应时间的three_factor进行k_means聚类，并对df_train打标签。

update: 加入获取risk_free_rate数据的函数。

### 策略 （位于method文件夹中）
前3(4)个作为benchmark
#### naive_policy.py
对所有给定的portfolio一视同仁，每个取1/N的比例。

#### Markowitz.py [matrix inverse] 
经典马科维茨模型，但没有考虑weight必须非负（这样能够使得原问题有解析的结果）。有些weight最后会出现负的结果。

#### Markowitz_revised.py [QP]
对马科维茨模型加上非负条件，运用gurobi求解。

#### SAA_CVaR.py [LP with too many constraints] 
将df_train运用sample average approximation方法最好地在CVaR指标上拟合训练集，给出相应的weight.

#### FCVaR_no_cluster.py [SOCP]
运用Popescu (2007)给出的worst_case CVaR的界解出相应的weight.

#### FCVaR_cluster.py [SOCP + cluster]
运用Popescu (2007)给出的worst_case CVaR的界，并加训练集分为不同cluster解出相应的weight.

#### FCVaR_cluster_bs.py [SOCP + cluster + bisearch]
与前三个py文件目标是CVaR不同的是，该策略将worst_case CVaR约束在target rate之下，采用二分法找到满足条件的最小的epsilon.
【理论证明可以直接求解，但还没有写，并且我们在计算中允许二分查找的时间复杂度

#### show_return.py
给出同一个rolling period内部return 的变化情况。

【用于update: 发现trans 和之前rolling版本每个周期内部结果不够理想，尝试重新调整。这样更加符合实际情况

### 其他
#### just_try.py
尝试了对portfolio_return进行PCA处理后再聚类的效果，但是结果一般。

#### cluster_analysis.py
对于聚类方法的分析和可视化。

#### self_simulate.py
利用 Gao et al. (2019)和Fan (2008)的参数replica相关参数，构建新的自己生成的数据集，结果和实际数据集储存在factor model文件夹。

## Test_sample
选取20120101-20161231五年的日数据，对不同方法进行测试。

分别预测2017开始到20170401，20170701，20180101，20190101的performance.

选取10/17/30 *eq_average/weight_average 6种industry_portfolio的日数据。

除了17_eq, 30_eq的csv解析时出现问题，其余均已完成。结果保存在test_result_(后缀).csv中。

总体来看Popescu_cluster_factor效果占优。

## Thinking
一点启示是只有相对更好的policy，在不同指标上不可能一直占优。

高维的时候Popescu_cluster_return表现较差，可能是出现curse_of_dimensionality现象。高维K-means效果不佳，并且计算耗时较大。
而factor可以从ML角度上看是对return进行PCA处理。three_factor_model是将return降低为三维, five_factor_model将return降低为五维。

目前在2，3，4， 5比较中分为2-3类较为合理，且表现较好。【可以进一步在train set里面分为train和validation set进行比较

Popescu_cluster_bs和Popescu_cluster差别不大。【仍然需要人为设置target rate

用
## Note
本程序需要有Gurobi与Python的接口.

本程序实现的portfolio组合不含有risk_free asset.

投资组合权重和为1，并且除了Markowitz.py均要求非负约束。

Sharper ratio这里的return都是超额收益，事先已减去risk_free rate，

loss_probability与同期的risk_free rate比较，Conditional Loss代表给定损失(小于risk_free rate)概率的情况下损失的均值。
VaR直接代表给定epsilon值的分位数，CVaR为给定epsilon值的value at risk.

本程序已经测试的数据集有10_Daily, 10_Daily_eq，10_Monthly, 17_Daily， 17_Monthly, 30_Daily.csv.其余有些数据集本身格式有些问题。
各数据集选取样本时间可参考csv文件。

v1.csv和/result/v2.csv是给定数据集*对于不同参数、不同时间段给出了一个sensitivity anlysis的更全策略方法。

result/v3.csv是对于时间段调整（将训练集开始时间后移至只有3/2/1 years)做的结果。

result/v4.csv是对于时间段整体调整(前后各平移1，2年)做的结果。
*给定数据集即20120101-20161231作为训练集，之后3/6/12/24个月作为测试集进行训练。

在17v4的结果不是很好。

result/v5.csv将return样本PCA处理后，结果提升不显著。

有些结果不够稳定，在2-4聚类之间。

rolling版本：一周（5个交易日）rolling一次，结果在result文件夹里标记为_rolling的csv中，cluster method保持了较好的效果。

_vXXXX.csv里rolling_day = 1（理想情况下），此时cluster_method表现依旧比较好。

之后在transaction的部分主要考虑选用rolling_day = 30.

========

10个portfolio
10-14 15 5个交易日

========
## Reference
Markowitz, H. M. (1952). Portfolio Selection. Journal of Finance 7:77–91.

Popescu, Ioana. 2007. Robust mean-covariance solutions for stochastic optimization. Operations Research 55(1) 98–112

DeMiguel, Victor, Lorenzo Garlappi, Raman Uppal. 2007. Optimal versus naive diversification: How inefficient is the 1/n portfolio strategy? The review of Financial studies 22(5) 1915–1953.

Chen, Zhi, Melvyn Sim, Peng Xiong. 2019. Robust Stochastic Optimization. Working Paper . 

Gao, Rui, and Anton J. Kleywegt. Data-driven robust optimization with known marginal distributions. Working paper, 2017.

Fan J, Fan Y, Lv J (2008) High dimensional covariance matrix estimation using a factor model. Journal of Econometrics 147(1): 186-197
