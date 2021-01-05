import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import math
import statsmodels.stats.anova as anova
import scipy as sci
from sklearn import metrics

pd.set_option('display.float_format', lambda x: '%.10f' % x)
# pd.reset_option('display.float_format')  # undo

# Loading data, just copy the Training_Dataset.csv file into the working directory of your python project:
traindata = pd.read_csv("Training_Dataset.csv", sep=";")

# Build some groups in dataset based on codebook
pl_vars = traindata.loc[:, "sales":"annual_profit"]
bs_vars = traindata.loc[:, "total_assets":"trade_receivables_lt"]
cf_vars = traindata.loc[:, "cf_operating":"cf_financing"]

# Build some groups to use as indices when accessing traindata
catvar = [i for i in list(traindata.columns) if traindata[i].dtype == 'O']  # category variables
numvar = [i for i in list(traindata.columns) if traindata[i].dtype in ['float64', 'int64']]  # numerical variables
boolvar = [i for i in list(traindata.columns) if traindata[i].dtype == bool]  # boolean variables
#%%
#Interest coverage ratio (Olli)








+%%
# n/a's Fredi
#%%
# with this code all NA's should be replaced with the respective mean! (excluding firm context variables)
# x = traindata.mean()
# traindata.fillna(x)

# 1. Ratio "Interest Coverage Ratio" - P&L Variables - earn_from_op & oth_interest_exp
pl_na_overview = pd.DataFrame({'Valid': pl_vars.notnull().sum(),
              'NAs': pl_vars.isnull().sum(),
              'NAs of total': pl_vars.isnull().sum() / pl_vars.shape[0]}
            ).sort_values('NAs of total', ascending=True)
print(pl_na_overview)

# Storing Mean of PL variables
pl_vars_mean = pl_vars.mean()
print(pl_vars_mean)

#print(traindata.groupby("legal_form").fin_result.mean())
#print(traindata.groupby("default").fin_result.mean())

# Callable grouping for default and non-default comparison
#default_groups = traindata.groupby("default")
#print(default_groups.sales.mean())  # example, call as default_groups.column.function


#Manipulation of earn_from_op & oth_interest_exp
traindata["earn_from_op"].fillna(pl_vars_mean["earn_from_op"])
traindata["oth_interest_exp"].fillna(pl_vars_mean["oth_interest_exp"])

#%%
# 2. Ratio "ROA" - total_result & total assets
bs_na_overview = pd.DataFrame({'Valid': bs_vars.notnull().sum(),
              'NAs': bs_vars.isnull().sum(),
              'NAs of total': bs_vars.isnull().sum() / bs_vars.shape[0]}
            ).sort_values('NAs of total', ascending=True)
print(bs_na_overview)

# Storing Central Tendencies of BS variables
#bs_vars_describe = bs_vars.describe()
bs_vars_mean = bs_vars.mean()
print(bs_vars_mean)

#Manipulation of total_result & total assets
traindata["total_assets"].fillna(bs_vars_mean["total_assets"])
traindata["total_result"].fillna(pl_vars_mean["total_result"])

#%%
# 3. Ratio - Leverage Ratio - (total_liabilities_st + mt + lt) / total_equity

#Manipulation of variables
traindata["total_liabilities_st"].fillna(bs_vars_mean["total_liabilities_st"]) # not necessary
traindata["total_liabilities_mt"].fillna(bs_vars_mean["total_liabilities_mt"]) # not necessary
traindata["total_liabilities_lt"].fillna(bs_vars_mean["total_liabilities_lt"]) # not necessary
#traindata["total_equity"].fillna(bs_vars_mean["total_equity "])

#%%
# 4. Ratio - year inc (Julian?)

#%%
# 5. Ratio - Equity Ratio - total_equity & total assets
# Already manipulated in previous ratios

#%%
# 6. Ratio - Operating margin - earn_from_op & sales
# Earning from operations already adjusted

# Adjustment Sales (P&L)
traindata["sales"].fillna(pl_vars_mean["sales"])

#%%
# 7. Ratio - Cashflow Measure ??

#%%
#8. Ratio - Liquidity measures - current_assets & total_liabilities_st (Umlaufvermögen/kurz. FK)
# Look into BS DATA

# total_liabilities_st --> already done
traindata["current_assets"].fillna(bs_vars_mean["current_assets"])