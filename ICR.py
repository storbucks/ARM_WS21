# NB settings
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:60% !important; }</style>"))

# Settings
import pandas as pd
import numpy as np
import scipy as sci
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
plt.style.use(['bmh','seaborn-white'])
import seaborn as sns

pd.set_option('display.width', 1200)
np.set_printoptions(linewidth=1200)
pd.set_option('display.max_columns',50)
pd.set_option('display.max_rows',100)

traindata = pd.read_csv("Training_Dataset.csv", sep=';',
                    dtype={'zip_code':str,
                           'sector':str,
                           'default':bool}).set_index('id')

traindata['defn'] = traindata.default.astype(np.float)

#%%
pd.DataFrame({'Valid': traindata.notnull().sum(),
              'NAs': traindata.isnull().sum(),
              'NAs of total': traindata.isnull().sum() / traindata.shape[0]}
            ).sort_values('NAs of total', ascending=False)

#%%
tbl = traindata.assign(IsMissing = lambda x: x.oth_interest_inc.isnull()).groupby('IsMissing').default.describe()
tbl['Def'] = tbl['count'] - tbl['freq']
tbl['Avg'] = tbl['Def'] / tbl['count']
tbl

#%%
mdl = sm.Logit.from_formula('defn ~ IsMissing + 1',
                            data=traindata.assign(IsMissing = lambda x: x.oth_interest_inc.isnull())
                           ).fit(disp=False, maxiter=100)
print(mdl.summary2())

#%%
print(mdl.get_margeff(dummy=True).summary())

#### all negative sign.

#%%
traindata['oth_interest_inc_grouped'] = np.select(
    [traindata['oth_interest_inc'].between(0, 2.447041e+04, inclusive=True),
     traindata['oth_interest_inc'].between(2.447041e+04, 3.305790e+03, inclusive=True),
     traindata['oth_interest_inc'].between(3.305790e+03, 4.354600e+02, inclusive=True),
     traindata['oth_interest_inc'].between(4.354600e+02, np.inf, inclusive=True)],
    ['small', 'medium', 'large', 'extra large'], default='Unknown')

traindata[['oth_interest_inc', 'oth_interest_inc_grouped']]

#%%
traindata['oth_interest_inc_grouped'].value_counts()
#%%
traindata.groupby('oth_interest_inc_grouped').defn.mean()
#%%
traindata.defn.mean()

#%%
mdl = sm.Logit.from_formula('defn ~ oth_interest_inc_grouped + 1', data=traindata).fit(disp=False, maxiter=100)
print(mdl.summary2())

#%%
print(mdl.get_margeff(dummy=True).summary())

#%%
mdl = sm.OLS.from_formula('np.log(oth_interest_exp) ~ np.log(total_assets) + 1', data=traindata).fit(disp=False, maxiter=100)
print(mdl.summary2())

#%%
traindata['oth_interest_exp_hat'] = np.exp(mdl.predict(exog=traindata.total_assets)).round(0)
traindata['oth_interest_exp_imp'] = traindata['oth_interest_exp'].fillna(traindata['oth_interest_exp_hat'])
tmp = traindata[['oth_interest_exp', 'oth_interest_exp_hat','oth_interest_exp_imp']]
tmp.head(10)

np.log(tmp).plot(kind='scatter',x = 'oth_interest_exp', y='oth_interest_exp_hat', figsize=(15,10))

#für n arsch, alles nicht signifikant
################################################
#%%
# ICR na ersetzen

total_liabilities = traindata.total_liabilities_st.copy() + traindata.total_liabilities_mt.copy() + traindata.total_liabilities_lt.copy()
interest_exp_rate = traindata.oth_interest_exp.copy() / total_liabilities
oth_interest_exp_filler = []
for i in range(0, len(traindata.oth_interest_exp)):
    oth_interest_exp_filler.append(interest_exp_rate.mean() * total_liabilities[i])
    traindata.oth_interest_exp.fillna(oth_interest_exp_filler[i], inplace=True)
traindata.oth_interest_exp.head()

################### Freddi back-up
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

