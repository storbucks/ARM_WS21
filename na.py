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
additionaldata = pd.read_csv("sectors_overview_5.csv", sep=";").set_index("sector")
#%%
#Interest coverage ratio (Olli)









#%%
# with this code all NA's should be replaced with the respective mean! (excluding firm context variables)
# x = traindata.mean()
# traindata.fillna(x)

# Overview
pl_na_overview = pd.DataFrame({'Valid': pl_vars.notnull().sum(),
              'NAs': pl_vars.isnull().sum(),
              'NAs of total': pl_vars.isnull().sum() / pl_vars.shape[0]}
            ).sort_values('NAs of total', ascending=True)
print(pl_na_overview)
#%%
bs_na_overview = pd.DataFrame({'Valid': bs_vars.notnull().sum(),
              'NAs': bs_vars.isnull().sum(),
              'NAs of total': bs_vars.isnull().sum() / bs_vars.shape[0]}
            ).sort_values('NAs of total', ascending=True)
print(bs_na_overview)
#%%
cf_na_overview = pd.DataFrame({'Valid': cf_vars.notnull().sum(),
              'NAs': cf_vars.isnull().sum(),
              'NAs of total': cf_vars.isnull().sum() / cf_vars.shape[0]}
            ).sort_values('NAs of total', ascending=True)
print(cf_na_overview)
#%%
# Storing Means of PL, BS & CF Variables
pl_vars_mean = pl_vars.mean()
print(pl_vars_mean)

bs_vars_mean = bs_vars.mean()
print(bs_vars_mean)

cf_vars_mean = cf_vars.mean()
print(cf_vars_mean)

#%%
# Manipulation Backup for Ratios - also look at Excel --> NA's
#traindata["earn_from_op"].fillna(pl_vars_mean["earn_from_op"])
#traindata["oth_interest_exp"].fillna(pl_vars_mean["oth_interest_exp"])
#traindata["total_assets"].fillna(bs_vars_mean["total_assets"])
#traindata["total_result"].fillna(pl_vars_mean["total_result"])
#traindata["total_liabilities_st"].fillna(bs_vars_mean["total_liabilities_st"])
#traindata["total_liabilities_mt"].fillna(bs_vars_mean["total_liabilities_mt"])
#traindata["total_liabilities_lt"].fillna(bs_vars_mean["total_liabilities_lt"])
#traindata["total_equity"].fillna(bs_vars_mean["total_equity "])
#traindata["sales"].fillna(pl_vars_mean["sales"])
#traindata["current_assets"].fillna(bs_vars_mean["current_assets"])

#%%
#TO DO:
# 1. Group By Sektors  --> Übersektor (Fredi)
# 2. Look deeper into oth_interest_exp & total_equity
# 3. Design If Rule for the variables (Levels)

# maybe not necessarytraindata.insert(4, "Übersektor", "x", allow_duplicates= True)
#maybe delete second column in additionaldata
left_join = pd.merge(traindata, additionaldata, on = 'sector', how = 'left')

#print(traindata.groupby("legal_form").fin_result.mean())
#print(traindata.groupby("default").fin_result.mean())

# Callable grouping for default and non-default comparison
#default_groups = traindata.groupby("default")
#print(default_groups.sales.mean())  # example, call as default_groups.column.function

