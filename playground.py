import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.anova as anova
import scipy as sci

from sklearn import metrics

# Loading data, just copy the Training_Dataset.csv file into the working directory of your python project:
traindata = pd.read_csv("Training_Dataset.csv", sep=";")
#%%
# Run some checks if you want to:
print(traindata.head())
print(traindata.tail())
print(traindata.isnull().sum().sort_values(ascending=False))

# Display information about the dataset at a glance:
print(traindata.info())  # Output: 40 cols, 669 rows, dtypes: float, int, object(here: strings), added: bool (2)
#%%
# Build some groups in dataset based on codebook
pl_vars = traindata.loc[:, "sales":"annual_profit"]
bs_vars = traindata.loc[:, "total_assets":"trade_receivables_lt"]
cf_vars = traindata.loc[:, "cf_operating":"cf_financing"]

# Add boolean var whether fin result pos/neg, True when negative
loser_col = []
for value in traindata["fin_result"]:
    if value < 0:
        loser_col.append(True)
    else:
        loser_col.append(False)

traindata["losers"] = loser_col

# Build some groups to use as indices when accessing traindata
catvar = [i for i in list(traindata.columns) if traindata[i].dtype=='O']  # category variables
numvar = [i for i in list(traindata.columns) if traindata[i].dtype in ['float64','int64']]  # numerical variables
boolvar = [i for i in list(traindata.columns) if traindata[i].dtype==bool]  # boolean variables

#%%
# Group by and check financial result --> explanatory power in legal form, etc. ?
print(traindata.groupby("type_pl").fin_result.mean())
print(traindata.groupby("legal_form").fin_result.mean())
print(traindata.groupby("default").fin_result.mean())

# Callable grouping for default and non-default comparison
default_groups = traindata.groupby("default")
print(default_groups.sales.mean())  # example, call as default_groups.column.function

#%%
# Some data analztics
# Check for missing values:
print(traindata.isnull().sum())

# Examine categorial variables
for i in catvar[1:]:  # w/o id
    print('============================================')
    print(f'Variable: {i} \n')
    x1 = traindata[i].value_counts()
    x2 = x1 / np.sum(x1) * 100
    x = pd.concat([x1,x2], axis=1)
    x.columns = ['Count', 'in %']
    print(x)
    print()


#%%
# # Check some key figures: gross_performance, gross_profit
# print(traindata["gross_performance"].describe())  # no negative performance gross
# print(traindata["gross_profit"].describe())  # no losses gross
# print(traindata["fin_result"].describe())  # losses net prevalent
#
# loser = traindata.groupby(by="losers")  # create group of losers and winners
# print(loser.get_group(False).mean())

#%%
# Check for correlations with heatmap
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
# fig.suptitle("Correlation matrices", size=16)

sns.heatmap(pl_vars.corr(method="pearson"), ax=ax[0], annot=False, vmax=1, vmin=-1)
ax[0].set_title("P&L")
sns.heatmap(bs_vars.corr(method="pearson"), ax=ax[1], annot=False, vmax=1, vmin=-1)
ax[1].set_title("Balance Sheet")
sns.heatmap(cf_vars.corr(method="pearson"), ax=ax[2], annot=False, vmax=1, vmin=-1)
ax[2].set_title("Cash Flow")

plt.show()

#%%
# Some nonsense ols regression which i couldn't even interpret
traindata["losers_dum"] = traindata.losers.astype(int)

regr1 = smf.ols("default ~ year_inc", data=traindata)
res = regr1.fit()
print(res.summary2())

fig, ax = plt.subplots(1, figsize=(5,5))

xs = np.arange(traindata.year_inc.min(), traindata.year_inc.max(), 1)

ys = res.params[0] + res.params[1] * xs

ax = plt.scatter(traindata.year_inc, traindata.default, color='darkblue')
ax = plt.plot(xs,ys, color='black')

plt.xlabel('year_inc')
plt.ylabel('PD / Default')
plt.show()

# hallo oli

# Hallo