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
tbl = traindata.assign(IsMissing = lambda x: x.oth_interest_exp.isnull()).groupby('IsMissing').default.describe()
tbl['Def'] = tbl['count'] - tbl['freq']
tbl['Avg'] = tbl['Def'] / tbl['count']
tbl

#%%
mdl = sm.Logit.from_formula('defn ~ IsMissing + 1',
                            data=traindata.assign(IsMissing = lambda x: x.oth_interest_exp.isnull())
                           ).fit(disp=False, maxiter=100)
print(mdl.summary2())

#%%
print(mdl.get_margeff(dummy=True).summary())

#### all negative sign.

#%%
traindata['oth_interest_exp_grouped'] = np.select(
    [traindata['oth_interest_exp'].between(0, 20000, inclusive=True),
     traindata['oth_interest_exp'].between(20001, 120000, inclusive=True),
     traindata['oth_interest_exp'].between(120001, 250000, inclusive=True),
     traindata['oth_interest_exp'].between(250001, np.inf, inclusive=True)],
    ['small', 'medium', 'large', 'extra large'], default='Unknown')

traindata[['oth_interest_exp', 'oth_interest_exp_grouped']]

#%%
traindata['oth_interest_exp_grouped'].value_counts()
#%%
traindata.groupby('oth_interest_exp_grouped').defn.mean()
#%%
traindata.defn.mean()

#%%
mdl = sm.Logit.from_formula('defn ~ oth_interest_exp_grouped + 1', data=traindata).fit(disp=False, maxiter=100)
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
