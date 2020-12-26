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
print(traindata.info())  # Output: 40 cols, 669 rows, dtypes: float, int, object(here: strings)
#%%
# Build some groups in dataset
catvar = [i for i in list(traindata.columns) if traindata[i].dtype=='O']  # category variables
numvar = [i for i in list(traindata.columns) if traindata[i].dtype in ['float64','int64']]  # numerical variables
pl_vars = traindata.loc[:, "sales":"annual_profit"]
bs_vars = traindata.loc[:, "total_assets":"trade_receivables_lt"]
cf_vars = traindata.loc[:, "cf_operating":"cf_financing"]

#%%
# Check for missing values:
print(traindata.isnull().sum())

# test
# yeehaa

# Test OLS regression made by me
testreg = smf.ols(formula="sales ~ gross_profit", data=traindata)
res = testreg.fit()
print(res.summary2())

#%% <- ist ein code separator, wenn es bei euch hängt einfach weg machen
# Check some key figures: gross_performance, gross_profit
print(traindata["gross_performance"].describe())  # no negative performance gross
print(traindata["gross_profit"].describe())  # no losses gross
print(traindata["fin_result"].describe())  # losses net prevalent

# Add whether loser or winner in terms of fin result
loser_col = []
for value in traindata["fin_result"]:
    if value < 0:
        loser_col.append("loser")
    else:
        loser_col.append("winner")

traindata["losers"] = loser_col

loser = traindata.groupby(by="losers")  # create group of losers and winners
print(loser.get_group("winner").mean())

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
