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
# %%
# Run some checks if you want to:
print(traindata.head())
print(traindata.tail())
print(traindata.isnull().sum().sort_values(ascending=False))

# Display information about the dataset at a glance:
print(traindata.info())  # Output: 40 cols, 669 rows, dtypes: float, int, object(here: strings), added: bool (2)
# %%
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
catvar = [i for i in list(traindata.columns) if traindata[i].dtype == 'O']  # category variables
numvar = [i for i in list(traindata.columns) if traindata[i].dtype in ['float64', 'int64']]  # numerical variables
boolvar = [i for i in list(traindata.columns) if traindata[i].dtype == bool]  # boolean variables

# %%
# Group by and check financial result --> explanatory power in legal form, etc. ?
print(traindata.groupby("type_pl").fin_result.mean())
print(traindata.groupby("legal_form").fin_result.mean())
print(traindata.groupby("default").fin_result.mean())

# Callable grouping for default and non-default comparison
default_groups = traindata.groupby("default")
print(default_groups.sales.mean())  # example, call as default_groups.column.function

# %%
# Some data analztics
# Check for missing values:
#print(traindata.isnull().sum())

# Examine categorial variables
for i in catvar[1:]:  # w/o id
    print('============================================')
    print(f'Variable: {i} \n')
    x1 = traindata[i].value_counts()
    x2 = x1 / np.sum(x1) * 100
    x = pd.concat([x1, x2], axis=1)
    x.columns = ['Count', 'in %']
    print(x)
    print()

# %%
# # Check some key figures: gross_performance, gross_profit
# print(traindata["gross_performance"].describe())  # no negative performance gross
# print(traindata["gross_profit"].describe())  # no losses gross
# print(traindata["fin_result"].describe())  # losses net prevalent
#
# loser = traindata.groupby(by="losers")  # create group of losers and winners
# print(loser.get_group(False).mean())

# %%
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

# %%
# Some nonsense ols regression which i couldn't even interpret
traindata["losers_dum"] = traindata.losers.astype(int)

regr1 = smf.ols("default ~ year_inc", data=traindata)
res = regr1.fit()
print(res.summary2())

fig, ax = plt.subplots(1, figsize=(5, 5))

xs = np.arange(traindata.year_inc.min(), traindata.year_inc.max(), 1)

ys = res.params[0] + res.params[1] * xs

ax = plt.scatter(traindata.year_inc, traindata.default, color='darkblue')
ax = plt.plot(xs, ys, color='black')

plt.xlabel('year_inc')
plt.ylabel('PD / Default')
plt.show()

#%% Financial Ratios (Eva)
# Descriptive analysis
# print(traindata[numvar+[boolvar]].corr())

# hier müssen wir dann in die eckigen Klammern die vars einfügen, für die wir die correlation visualisieren möchten
# corrvar = [hier einfügen] wenn es keine Liste ist, sondern eine Bezeichnung [[Bez] + ..]
#f, ax = plt.subplots(figsize=(15,5))
#sns.heatmap(traindata[corrvar].corr(method='pearson'),
#            annot=True,cmap="coolwarm",
#            vmin=-1, vmax=1, ax=ax);

####################
# Financial ratios #
####################

# Liquidity ratios
current_ratio = traindata.current_assets.copy()/traindata.total_liabilities_st.copy() #  A healthy Current Ratio is between 1.5 and 3
cash_ratio = traindata.cash.copy()/traindata.total_liabilities_st.copy()
oper_cash_flow_ratio = traindata.cf_operating.copy() / traindata.total_liabilities_st.copy()
asset_structure = (traindata.trade_receivables_st.copy() + traindata.trade_receivables_lt.copy()) / traindata.current_assets.copy()

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
oper_margin_ratio = traindata.earn_from_op.copy() / traindata.sales.copy()
roa = traindata.total_result.copy() / traindata.total_assets.copy() # annual_profit instead of fin_result?
roe = traindata.fin_result.copy() / traindata.total_equity.copy()

frame = {'id': traindata.id, 'gross_margin_ratio': gross_margin_ratio, 'oper_margin_ratio': oper_margin_ratio, 'roa': roa, 'roe': roe}
profitability_ratios = pd.DataFrame(frame)
print(profitability_ratios)

frame = {'id': traindata.id, 'default': traindata.default, 'current_ratio': current_ratio, 'cash_ratio': cash_ratio, 'oper_cash_flow_ratio': oper_cash_flow_ratio,
          'total_liabilities': total_liabilities, 'debt_ratio': debt_ratio, 'asset_turnover': asset_turnover, 'debt_to_equity_ratio': debt_to_equity_ratio,
          'gross_margin_ratio': gross_margin_ratio, 'oper_margin_ratio': oper_margin_ratio, 'roa': roa, 'roe': roe}
ratios = pd.DataFrame(frame)
print(ratios)

f, ax = plt.subplots(figsize=(15,5))
sns.heatmap(ratios[2:].corr(method='pearson'),
            annot=True,cmap="coolwarm",
            vmin=-1, vmax=1, ax=ax);

plt.show()

# linear regressions to get an impression
ratios['Default_Dum'] = ratios.default.astype(int) # dummy variable for linear regression

mod = smf.ols(formula='Default_Dum ~ debt_ratio', data=ratios)  # significant !!
res = mod.fit()
print(res.summary2())

mod = smf.ols(formula='debt_ratio ~ Default_Dum', data=ratios)  # significant !!
res = mod.fit()
print(res.summary2())

res2 = sm.Logit.from_formula('Default_Dum ~ debt_ratio', data=ratios).fit(disp=False, maxiter=100)
print(res2.summary2())

fig, axes = plt.subplots(figsize=(15,5))

xs = np.arange(-10,ratios.debt_ratio.max()+10)

ys2 = res2.predict(exog=pd.DataFrame({'debt_ratio': xs}))

axes = plt.scatter(ratios.debt_ratio, ratios.Default_Dum, color='darkblue')
axes = plt.plot(xs,ys2, color='black')

plt.xlabel('debt_ratio')
plt.ylabel('Default_Dum');

plt.show()

#%%
# looking at missing values
# missing values based on variable
na = pd.DataFrame({'Valid': traindata.notnull().sum(),
                    'NAs': traindata.isnull().sum(),
                    'NAs of total': traindata.isnull().sum() / traindata.shape[0]}
                   ).sort_values('NAs of total', ascending=False)
print(na)


#%% Distribution analysis
# P&L variables
fig, axes = plt.subplots(len(pl_vars.columns), 1, figsize=(5, 5*len(pl_vars.columns)))

row = 0
for column in pl_vars.columns[0:]:
    print(column)
    sns.boxplot(y=pl_vars[column], ax=axes[row])
    axes[row].set_ylabel(column)
    row += 1
plt.show()

#%%
# BS variables
fig, axes = plt.subplots(len(bs_vars.columns), 1, figsize=(5, 5*len(pl_vars.columns)))

row = 0
for column in bs_vars.columns[0:]:
    print(column)
    sns.boxplot(y=bs_vars[column], ax=axes[row])
    axes[row].set_ylabel(column)
    row += 1
plt.show()

#%%
# CF variables
fig, axes = plt.subplots(len(cf_vars.columns), 1, figsize=(5, 5*len(cf_vars.columns)))

row = 0
for column in cf_vars.columns[0:]:
    print(column)
    sns.boxplot(y=cf_vars[column], ax=axes[row])
    axes[row].set_ylabel(column)
    row += 1
plt.show()