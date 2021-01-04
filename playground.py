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
#%%
# Build some groups in dataset based on codebook
pl_vars = traindata.loc[:, "sales":"annual_profit"]
bs_vars = traindata.loc[:, "total_assets":"trade_receivables_lt"]
cf_vars = traindata.loc[:, "cf_operating":"cf_financing"]

# Build some groups to use as indices when accessing traindata
catvar = [i for i in list(traindata.columns) if traindata[i].dtype == 'O']  # category variables
numvar = [i for i in list(traindata.columns) if traindata[i].dtype in ['float64', 'int64']]  # numerical variables
boolvar = [i for i in list(traindata.columns) if traindata[i].dtype == bool]  # boolean variables

#%%
# # Check for correlations with heatmap
# fig, ax = plt.subplots(1, 3, figsize=(18, 6))
# # fig.suptitle("Correlation matrices", size=16)
#
# sns.heatmap(pl_vars.corr(method="pearson"), ax=ax[0], annot=False, vmax=1, vmin=-1)
# ax[0].set_title("P&L")
# sns.heatmap(bs_vars.corr(method="pearson"), ax=ax[1], annot=False, vmax=1, vmin=-1)
# ax[1].set_title("Balance Sheet")
# sns.heatmap(cf_vars.corr(method="pearson"), ax=ax[2], annot=False, vmax=1, vmin=-1)
# ax[2].set_title("Cash Flow")
#
# plt.show()
#%%
#############################
# EXPLORATIVE DATA ANALYSIS #
#############################

# Winsorizing function (can winsorize several cols at once, but with same percentiles)
def percentile_capping(df, cols, from_lower_end, from_higher_end):
    for col in cols:
        sci.stats.mstats.winsorize(a=df[col], limits=(from_lower_end, from_higher_end), inplace=True)

####################
##### YEAR_INC #####
####################
print(traindata.year_inc.describe())  #for comparison, compare min/max values

# Wins year_inc
percentile_capping(traindata, ['year_inc'], 0.01, 0.005) # keeps values betwnn 1% and 99.5%
print(traindata.year_inc.describe())

# Transform year_inc it into Age
for i in range(0, len(traindata.year_inc)):
    traindata.year_inc[i] = 2021 - traindata.year_inc[i]
oldest_company = max(traindata.year_inc)

age_level = []
for i in range(0, len(traindata.year_inc)):
    age_level.append(traindata.year_inc[i].copy()/oldest_company)
#%%
####################
##### TOTAL_EQUITY #####
####################
print(traindata.total_equity.describe())
eq_min = traindata[traindata.total_equity <= 0]

sns.distplot(a=traindata.total_equity)
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
curr_rat = ["current_assets", "total_liabilities_st"]
#cash_ratio = traindata.cash.copy()/traindata.total_liabilities_st.copy()
#oper_cash_flow_ratio = traindata.cf_operating.copy() / traindata.total_liabilities_st.copy()
#asset_structure = (traindata.trade_receivables_st.copy() + traindata.trade_receivables_lt.copy()) / traindata.current_assets.copy()

#frame = {'id': traindata.id, 'current_ratio': current_ratio, 'cash_ratio': cash_ratio, 'oper_cash_flow_ratio': oper_cash_flow_ratio}
#liquidity_ratios = pd.DataFrame(frame)
#print(liquidity_ratios)

# Leverage ratios
total_liabilities = traindata.total_liabilities_st.copy() + traindata.total_liabilities_mt.copy() + traindata.total_liabilities_lt.copy()
debt_ratio = total_liabilities.copy() / traindata.total_assets.copy()
debt_to_equity_ratio = total_liabilities.copy() / traindata.total_equity.copy()
d_rat = ["total_liabilities_st", "total_liabilities_mt", "total_liabilities_lt", "total_assets"]
dte_rat = ["total_liabilities_st", "total_liabilities_mt", "total_liabilities_lt", "total_equity"]

#frame = {'id': traindata.id, 'total_liabilities': total_liabilities, 'debt_ratio': debt_ratio, 'debt_to_equity_ratio': debt_to_equity_ratio}
#leverage_ratios = pd.DataFrame(frame)
#print(leverage_ratios)

# Efficiency ratios
#asset_turnover = traindata.sales.copy() / traindata.total_assets.copy()

#frame = {'id': traindata.id, 'asset_turnover': asset_turnover}
#efficiency_ratios = pd.DataFrame(frame)
#print(efficiency_ratios)

# Profitability ratios
#gross_margin_ratio = traindata.gross_profit.copy() / traindata.sales.copy()
#oper_margin_ratio = traindata.earn_from_op.copy() / traindata.sales.copy()
roa = traindata.total_result.copy() / traindata.total_assets.copy() # annual_profit instead of fin_result?
roa_rat = ["total_result", "total_assets"]
#roe = traindata.fin_result.copy() / traindata.total_equity.copy()

#frame = {'id': traindata.id, 'gross_margin_ratio': gross_margin_ratio, 'oper_margin_ratio': oper_margin_ratio, 'roa': roa, 'roe': roe}
#profitability_ratios = pd.DataFrame(frame)
#print(profitability_ratios)

# Other ratios
interest_coverage = traindata.earn_from_op.copy() / traindata.oth_interest_exp.copy()
ic_rat = ["earn_from_op", "oth_interest_exp"]

equity_ratio = traindata.total_equity.copy() / traindata.total_assets.copy()
e_rat = ["total_equity", "total_assets"]

ebit_margin = traindata.earn_from_op.copy() / traindata.sales.copy()
ebt_rat = ["earn_from_op", "sales"]

frame = {'id': traindata.id, 'default': traindata.default, 'interest_coverage': interest_coverage, 'roa': roa, 'debt_ratio': debt_ratio,
         'debt_to_equity_ratio': debt_to_equity_ratio, 'age_level': age_level, 'equity_ratio': equity_ratio,
         'ebit_margin': ebit_margin, 'cf_operating': traindata.cf_operating, 'current_ratio': current_ratio}
indicators = pd.DataFrame(frame)
print(indicators)

f, ax = plt.subplots(figsize=(20,5))
sns.heatmap(indicators[2:].corr(method='pearson'),
            annot=True,cmap="coolwarm",
            vmin=-1, vmax=1, ax=ax);

plt.show()

#%%
# linear regressions (dummy and indicators) to get an impression
indicators['Default_Dum'] = indicators.default.astype(int) # dummy variable for linear regression

mod = smf.ols(formula='Default_Dum ~ debt_ratio', data=indicators)  # significant !!
res = mod.fit()
print(res.summary2())

mod = smf.ols(formula='debt_ratio ~ Default_Dum', data=indicators)  # significant !!
res = mod.fit()
print(res.summary2())

res2 = sm.Logit.from_formula('Default_Dum ~ debt_ratio', data=indicators).fit(disp=False, maxiter=100)
print(res2.summary2())

fig, axes = plt.subplots(figsize=(15,5))

xs = np.arange(-10,indicators.debt_ratio.max()+10)

ys2 = res2.predict(exog=pd.DataFrame({'debt_ratio': xs}))

axes = plt.scatter(indicators.debt_ratio, indicators.Default_Dum, color='darkblue')
axes = plt.plot(xs,ys2, color='black')

plt.xlabel('debt_ratio')
plt.ylabel('Default_Dum');

plt.show()

mod = smf.ols(formula='Default_Dum ~ interest_coverage', data=indicators)  # significant !!
res = mod.fit()
print(res.summary2())

mod = smf.ols(formula='Default_Dum ~ roa', data=indicators)  # significant !!
res = mod.fit()
print(res.summary2())

mod = smf.ols(formula='Default_Dum ~ age_level', data=indicators)  # significant !!
res = mod.fit()
print(res.summary2())

mod = smf.ols(formula='Default_Dum ~ equity_ratio', data=indicators)  # significant !!
res = mod.fit()
print(res.summary2())

mod = smf.ols(formula='Default_Dum ~ ebit_margin', data=indicators)  # significant !!
res = mod.fit()
print(res.summary2())

mod = smf.ols(formula='Default_Dum ~ cf_operating', data=indicators)  # significant !!
res = mod.fit()
print(res.summary2())

mod = smf.ols(formula='Default_Dum ~ current_ratio', data=indicators)  # significant !!
res = mod.fit()
print(res.summary2())

#%%
# # looking at missing values (Fredi)! 1. step Looking on variables & inserting mean if value is missing
# --> Further adjustments should be made # Note: Allgemein fehlen bei unseren Variablen kaum Werte!
# However, is a high number of missing values a potential indicator for a higher probability of default?

# Focusing on yellow marked variables in Excel
# 1. oth_interest_exp - not working
#tbl = traindata.assign(IsMissing = lambda x: x.oth_interest_exp.isnull()).groupby('IsMissing').default.describe()
#tbl['Def'] = tbl['count'] - tbl['freq']
#tbl['Avg'] = tbl['Def'] / tbl['count']
#print(tbl)

# Is the difference statistical significant? - not working
#mdl = sm.Logit.from_formula('defn ~ IsMissing + 1',
                           # data=traindata.assign(IsMissing = lambda x: x.oth_interest_exp.isnull())
                           #).fit(disp=False, maxiter=100)
#print(mdl.summary2())

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


#%% Distribution analysis
# P&L variables
fig, axes = plt.subplots(len(pl_vars.columns), 2, figsize=(10, 5*len(pl_vars.columns)))
row = 0
for column in pl_vars.columns[0:]:
    sns.distplot(pl_vars[column], kde=True, ax=axes[row, 0])
    sns.boxplot(y=pl_vars[column], ax=axes[row, 1])
    row += 1
plt.show()

# BS variables
fig, axes = plt.subplots(len(bs_vars.columns), 2, figsize=(10, 5*len(bs_vars.columns)))
row = 0
for column in bs_vars.columns[0:]:
    sns.distplot(bs_vars[column], kde=True, ax=axes[row, 0])
    sns.boxplot(y=bs_vars[column], ax=axes[row, 1])
    row += 1
plt.show()

# CF variables
fig, axes = plt.subplots(len(cf_vars.columns), 2, figsize=(10, 5*len(cf_vars.columns)))
row = 0
for column in cf_vars.columns[0:]:
    sns.distplot(cf_vars[column], kde=True, ax=axes[row, 0])
    sns.boxplot(y=cf_vars[column], ax=axes[row, 1])
    row += 1
plt.show()

#%%
#############################################
# 01: Data analysis for Interest coverage ratio #
#############################################
fig, axes = plt.subplots(len(ic_rat), 2, figsize=(10, 10))
fig.suptitle("Interest Coverage Ratio")
row = 0
for var in ic_rat:
    sns.distplot(traindata[var], kde=True, ax=axes[row, 1])
    sns.boxplot(y=traindata[var], ax=axes[row, 0])
    row += 1
plt.show()

for var in ic_rat:
    print(traindata[var].describe())
print(interest_coverage.describe())

#########################
# 02: Data analysis for ROA #
#########################
fig, axes = plt.subplots(len(roa_rat), 2, figsize=(10, 10))
fig.suptitle("ROA")
row = 0
for var in roa_rat:
    sns.distplot(traindata[var], kde=True, ax=axes[row, 1])
    sns.boxplot(y=traindata[var], ax=axes[row, 0])
    row += 1
plt.show()

for var in roa_rat:
    print(traindata[var].describe())
print(roa.describe())

#############################################
# 03: Data analysis for Debt Ratio #
#############################################
fig, axes = plt.subplots(len(d_rat), 2, figsize=(10, 10))
fig.suptitle("Debt Ratio")
row = 0
for var in d_rat:
    sns.distplot(traindata[var], kde=True, ax=axes[row, 1])
    sns.boxplot(y=traindata[var], ax=axes[row, 0])
    row += 1
plt.show()

for var in d_rat:
    print(traindata[var].describe())
print(debt_ratio.describe())

#############################################
# 04: Data analysis for Debt-to-Equity Ratio #
#############################################
fig, axes = plt.subplots(len(dte_rat), 2, figsize=(10, 10))
fig.suptitle("Debt-to-Equity Ratio")
row = 0
for var in dte_rat:
    sns.distplot(traindata[var], kde=True, ax=axes[row, 1])
    sns.boxplot(y=traindata[var], ax=axes[row, 0])
    row += 1
plt.show()

for var in dte_rat:
    print(traindata[var].describe())
print(debt_to_equity_ratio.describe())

#############################################
# 05: Data analysis for Equity Ratio #
#############################################
fig, axes = plt.subplots(len(e_rat), 2, figsize=(10, 10))
fig.suptitle("Equity Ratio")
row = 0
for var in e_rat:
    sns.distplot(traindata[var], kde=True, ax=axes[row, 1])
    sns.boxplot(y=traindata[var], ax=axes[row, 0])
    row += 1
plt.show()

for var in e_rat:
    print(traindata[var].describe())
print(equity_ratio.describe())

#############################################
# 06: Data analysis for EBIT Margin #
#############################################
fig, axes = plt.subplots(len(ebt_rat), 2, figsize=(10, 10))
fig.suptitle("EBIT Margin")
row = 0
for var in ebt_rat:
    sns.distplot(traindata[var], kde=True, ax=axes[row, 1])
    sns.boxplot(y=traindata[var], ax=axes[row, 0])
    row += 1
plt.show()

for var in ebt_rat:
    print(traindata[var].describe())
print(ebit_margin.describe())

#############################################
# 07: Data analysis for Current Ratio #
#############################################
fig, axes = plt.subplots(len(curr_rat), 2, figsize=(10, 10))
fig.suptitle("Current Ratio")
row = 0
for var in curr_rat:
    sns.distplot(traindata[var], kde=True, ax=axes[row, 1])
    sns.boxplot(y=traindata[var], ax=axes[row, 0])
    row += 1
plt.show()

for var in curr_rat:
    print(traindata[var].describe())
print(ebit_margin.describe())

#############################################
# 08: Data analysis for CF op #
#############################################
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("Operating CF")
sns.distplot(traindata["cf_operating"], kde=True, ax=axes[1])
sns.boxplot(y=traindata["cf_operating"], ax=axes[0])
plt.show()

print(traindata["cf_operating"].describe())
#%%
#############################################
# 09: Data analysis for Year Inc #
#############################################
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("Year Inc")
sns.distplot(traindata["year_inc"], kde=True, ax=axes[1])
sns.boxplot(y=traindata["year_inc"], ax=axes[0])
plt.show()

print(traindata["year_inc"].describe())
print(traindata["year_inc"].value_counts())
#%%
#############################################
# 10: Data analysis for Indicators #
#############################################
newinds = indicators[indicators.columns.difference(["id", "default", "debt_to_equity_ratio"])]
indics = newinds.columns.tolist()

fig, axes = plt.subplots(len(indics), 2, figsize=(10, 30))
fig.suptitle("Indicators")
row = 0
for var in newinds.columns[0:]:
    sns.distplot(indicators[var], kde=True, ax=axes[row, 1])
    sns.boxplot(y=indicators[var], ax=axes[row, 0])
    row += 1
plt.show()

for var in indics:
    print(indicators[var].describe())

#%% Alternative, aber imo schlechter
df_inds = indicators[indics]
ax = sns.boxplot(data=df_inds, orient="h", palette="Set2")
plt.show()

########################################################################################################################
#%%
# Logit regression with indicators (multivariate)

res2 = sm.Logit.from_formula('Default_Dum ~ debt_ratio + roa + age_level', data=indicators).fit(disp=False, maxiter=100)
print(res2.summary2())

# debt_ratio + interest_coverage + roa + age_level + equity_ratio + ebit_margin + cf_operating' + current_ratio'


frame = {'id': indicators.id, 'default_dum': indicators.default.astype(int), 'debt_ratio': indicators.debt_ratio, 'roa': indicators.roa, 'age_level': indicators.age_level}
history = pd.DataFrame(frame)
history["pd"] = ""
history["estimation"] = ""
nan_index = []

for i in range(0, len(history['id'])):
    x = res2.params[0] + res2.params[1] * history['debt_ratio'][i] + res2.params[2]* history['roa'][i] + res2.params[3] * history['age_level'][i]
    pi = (np.exp(x)/(1 + np.exp(x)))
    if not math.isnan(pi):
        history.pd[i] = pi
    else:
        history = history.drop([i])
        nan_index.append(i)


x = np.array(history['default_dum']).reshape((-1, 1))
y = np.array(history['pd'])
model = LinearRegression()
model.fit(x, y)
model = LinearRegression().fit(x, y)
intercept = float(model.intercept_)
slope = float(model.coef_[0])

for i in range(0, len(history['id'])):
    if i not in nan_index:
        if history.pd[i] >= intercept: # mit intercept+slope höhere trefferquote, aber weniger D's gefunden
            history.estimation[i] = 1
        else:
            history.estimation[i] = 0

count = 0
for i in range(0, len(history['id'])):
    if i not in nan_index:
        if history.default_dum[i] == history.estimation[i]:
            count += 1

strikes = count/len(history['id'])
print(str(round(strikes*100,2)) + " %")

#import xlsxwriter
#with xlsxwriter.Workbook('pds.xlsx') as workbook:
 #   worksheet = workbook.add_worksheet()
  #  for row_num, data in enumerate(history):
   #     worksheet.write_row(row_num, 0, data)

