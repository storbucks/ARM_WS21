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

pd.set_option('display.float_format', lambda x: '%.10f' % x)
# pd.reset_option('display.float_format')  # undo


#%%
traindata = pd.read_csv("Training_Dataset.csv", sep=';',
                    dtype={'zip_code':str,
                           'sector':str,
                           'default':bool}).set_index('id')

traindata['defn'] = traindata.default.astype(np.float)

#%%
#Interest coverage ratio (Olli)

data_na_overview = pd.DataFrame({'Valid': traindata.notnull().sum(),
              'NAs': traindata.isnull().sum(),
              'NAs of total': traindata.isnull().sum() / pl_vars.shape[0]}
            ).sort_values('NAs of total', ascending=True)
print(data_na_overview)

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

###Everything is not significant here.

#%%


#%%
# n/a's Fredi
