import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.anova as anova
import scipy as sci
# hey was geht
from sklearn import metrics

# Loading data, just copy the Training_Dataset.csv file into the working directory of your python project:
traindata = pd.read_csv("Training_Dataset.csv", sep=";")

# Default as a boolean
traindata.default = traindata.default.replace([0, 1], [False, True])

# Run some checks if you want to:
print(traindata.head())
print(traindata.tail())
print(traindata.isnull().sum().sort_values(ascending=False))

# Display information about the dataset at a glance:
print(traindata.info())  # Output: 40 cols, 669 rows, dtypes: float, int, object(here: strings)
catvar = [i for i in list(traindata.columns) if traindata[i].dtype == 'O']  # category variables
numvar = [i for i in list(traindata.columns) if traindata[i].dtype in ['float64', 'int64']]  # numerical variables
boolvar = 'default'

# Check for missing values:
print(traindata.isnull().sum())

# test

# Test OLS regression made by me
testreg = smf.ols(formula="sales ~ gross_profit", data=traindata)
res = testreg.fit()
print(res.summary2())

# description of boolean variable
print(traindata.default.describe())
print(traindata.default.astype(float).describe())

# mean, st. deviation ... of all variables
print(traindata.describe())

# frequency of categories (absolute and in percent)
for i in catvar[1:]:  # makes no sense for ID, therefore not for index 0 in catvar
    print('============================================')
    print(f'Variable: {i} \n')
    x1 = traindata[i].value_counts()
    x2 = x1 / np.sum(x1) * 100
    x = pd.concat([x1, x2], axis=1)
    x.columns = ['Count', 'in %']
    print(x)
    print()

# Descriptive analysis
print(traindata[numvar+[boolvar]].corr())

# hier müssen wir dann in die eckigen Klammern die vars einfügen, für die wir die correlation visualisieren möchten
# corrvar = [hier einfügen] wenn es keine Liste ist, sondern eine Bezeichnung [[Bez] + ..]
#f, ax = plt.subplots(figsize=(15,5))
#sns.heatmap(traindata[corrvar].corr(method='pearson'),
#            annot=True,cmap="coolwarm",
#            vmin=-1, vmax=1, ax=ax);


# Financial ratios
# Liquidity ratios
current_ratio = traindata.current_assets.copy()/traindata.total_liabilities_st.copy()
cash_ratio = traindata.cash.copy()/traindata.total_liabilities_st.copy()
oper_cash_flow_ratio = traindata.cf_operating.copy() / traindata.total_liabilities_st.copy()

frame = {'id': traindata.id, 'current_ratio': current_ratio, 'cash_ratio': cash_ratio, 'oper_cash_flow_ratio': oper_cash_flow_ratio}
liquidity_ratios = pd.DataFrame(frame)

print(liquidity_ratios)
# Leverage ratios
total_liabilities = traindata.total_liabilities_st.copy() + traindata.total_liabilities_mt.copy() + traindata.total_liabilities_lt.copy()
debt_ratio = total_liabilities.copy() / traindata.total_assets.copy()
debt_to_equity_ratio = total_liabilities.copy() / traindata.total_equity.copy()

frame = {'id': traindata.id, 'total_liabilities': total_liabilities, 'debt_ratio': debt_ratio, 'debt_to_equity_ratio': debt_to_equity_ratio}
leverage_ratios = pd.DataFrame(frame)
print(leverage_ratios)
# Efficiency ratios
asset_turnover = traindata.sales.copy() / traindata.total_assets.copy()
frame = {'id': traindata.id, 'asset_turnover': asset_turnover}
efficiency_ratios = pd.DataFrame(frame)
print(efficiency_ratios)
# Profitability ratios
gross_margin_ratio = traindata.gross_profit.copy() / traindata.sales.copy()
oper_margin_ratio = traindata.earn_from_op / traindata.sales.copy()
roa = traindata.fin_result / traindata.total_assets.copy()
roe = traindata.fin_result / traindata.total_equity.copy()

frame = {'id': traindata.id, 'gross_margin_ratio': gross_margin_ratio, 'oper_margin_ratio': oper_margin_ratio, 'roa': roa, 'roe': roe}
profitability_ratios = pd.DataFrame(frame)
print(profitability_ratios)
