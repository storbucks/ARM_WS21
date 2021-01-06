import pandas as pd
import numpy as np
import math

# Loading data, just copy the Training_Dataset.csv file into the working directory of your python project:
traindata_m = pd.read_csv("Training_Dataset.csv", sep=";")
testdata = pd.read_csv("Test_Dataset.csv", sep=";")
sector_data = pd.read_csv("sectors_overview_6.csv", sep=";",
                             dtype={'sector':'int64',
                                    'sector_string':'str'})
#%%
# Merging traindata_m & sector_data & filling sector_string NA's with unknown
traindata = pd.merge(traindata_m, sector_data, on = 'sector', how = 'left')
traindata['sector_string'] = traindata['sector_string'].fillna('Unknown')

#%%
# Build some groups in dataset based on codebook
pl_vars = traindata.loc[:, "sales":"annual_profit"]
bs_vars = traindata.loc[:, "total_assets":"trade_receivables_lt"]
cf_vars = traindata.loc[:, "cf_operating":"cf_financing"]
special_vars = traindata.loc[:, "sales":"sector_string"]

# Build some groups to use as indices when accessing traindata
catvar = [i for i in list(traindata.columns) if traindata[i].dtype == 'O']  # category variables
numvar = [i for i in list(traindata.columns) if traindata[i].dtype in ['float64', 'int64']]  # numerical variables
boolvar = [i for i in list(traindata.columns) if traindata[i].dtype == bool]  # boolean variables

#%%
#Overview over NA's
pl_na_overview = pd.DataFrame({'Valid': pl_vars.notnull().sum(),
              'NAs': pl_vars.isnull().sum(),
              'NAs of total': pl_vars.isnull().sum() / pl_vars.shape[0]}
            ).sort_values('NAs of total', ascending=True)
print(pl_na_overview)

bs_na_overview = pd.DataFrame({'Valid': bs_vars.notnull().sum(),
              'NAs': bs_vars.isnull().sum(),
              'NAs of total': bs_vars.isnull().sum() / bs_vars.shape[0]}
            ).sort_values('NAs of total', ascending=True)
print(bs_na_overview)

cf_na_overview = pd.DataFrame({'Valid': cf_vars.notnull().sum(),
              'NAs': cf_vars.isnull().sum(),
              'NAs of total': cf_vars.isnull().sum() / cf_vars.shape[0]}
            ).sort_values('NAs of total', ascending=True)
print(cf_na_overview)

#%%
# Storing sector specific Means of Numerical variables
special_vars_mean = special_vars.groupby("sector_string").mean()
pl_vars_mean = pl_vars.mean()
bs_vars_mean = bs_vars.mean()
cf_vars_mean = cf_vars.mean()

#%%
# Manipulation - Substituing NA's
traindata["earn_from_op"].fillna(pl_vars_mean["earn_from_op"])
traindata["total_assets"].fillna(bs_vars_mean["total_assets"])
traindata["total_result"].fillna(pl_vars_mean["total_result"])
traindata["total_liabilities_st"].fillna(bs_vars_mean["total_liabilities_st"])
traindata["total_liabilities_mt"].fillna(bs_vars_mean["total_liabilities_mt"])
traindata["total_liabilities_lt"].fillna(bs_vars_mean["total_liabilities_lt"])
traindata["total_equity"].fillna(special_vars_mean["total_equity"]) # another approach could be useful
traindata["sales"].fillna(pl_vars_mean["sales"])
traindata["current_assets"].fillna(bs_vars_mean["current_assets"])

#%%
# hier Funktionen zur Datenbereinigung einfügen
# Dealing with na: ICR - Notiz Fredi (habe statt traindata_m --> wieder traindata genommen --> siehe oben)
total_liabilities = traindata.total_liabilities_st.copy() + traindata.total_liabilities_mt.copy() + traindata.total_liabilities_lt.copy()
interest_exp_rate = traindata.oth_interest_exp.copy() / total_liabilities
oth_interest_exp_filler = []
for i in range(0, len(traindata.oth_interest_exp)):
    oth_interest_exp_filler.append(interest_exp_rate.mean() * total_liabilities[i])
    traindata.oth_interest_exp.fillna(oth_interest_exp_filler[i], inplace=True)
#%%
# Funktion für Equity einfügen


#############################
# functions to analyse data #
#############################


#  calculate and save different ratios or indicators
def create_indicator_frame(data):
    current_ratio = data.current_assets.copy()/data.total_liabilities_st.copy()
    total_liabilities = data.total_liabilities_st.copy() + data.total_liabilities_mt.copy() + data.total_liabilities_lt.copy()
    debt_ratio = total_liabilities.copy() / data.total_assets.copy()
    debt_to_equity_ratio = total_liabilities.copy() / data.total_equity.copy()
    roa = data.total_result.copy() / data.total_assets.copy()
    interest_coverage = data.earn_from_op.copy() / data.oth_interest_exp.copy()
    equity_ratio = data.total_equity.copy() / data.total_assets.copy()
    ebit_margin = data.earn_from_op.copy() / data.sales.copy()
    age = []
    for i in range(0, len(data.year_inc)):
        age.append(2021 - data.year_inc[i].copy())
    #  age_level = []
    #  for i in range(0, len(data.year_inc)):
    #      age_level.append(data.year_inc[i].copy()/oldest_company)
    frame = {'id': data.id, 'interest_coverage': interest_coverage, 'roa': roa, 'debt_ratio': debt_ratio,
            'debt_to_equity_ratio': debt_to_equity_ratio, 'equity_ratio': equity_ratio,
            'ebit_margin': ebit_margin, 'cf_operating': data.cf_operating, 'current_ratio': current_ratio, 'age': age}
    indicators = pd.DataFrame(frame)
    return indicators


#  calculate the values to classify companies in default and non-default
def calculate_pds(indicators, nan_index):
    frame = {'id': indicators.id}
    estimations = pd.DataFrame(frame)
    estimations['estimated_pd'] = ""

    # classification depends on one liquidity ratio (current assets/current liabs), one leverage ratio (debt ratio) and one profitability ratio (roa)
    for i in range(0, len(indicators['id'])):
        # hier mit tests geeignete betas finden
        x = -3.4163441714312563 + 0.9435381378963345 * indicators['debt_ratio'][i] - 1.0848159927024499 * indicators['current_ratio'][i] - (0.00010046357934681097/10000) * indicators['roa'][i]
        pi = (np.exp(x)/(1 + np.exp(x)))
        if not math.isnan(pi):
            estimations['estimated_pd'][i] = pi
        else:
            estimations = estimations.drop([i])
            nan_index.append(i)  # löschen sobald Funktionen für nan's drin sind
    return estimations


def create_default_booleans(estimations, nan_index):
    estimations['default_boolean'] = ""
    default_threshold = 0.06162628810257741  # geeigneten threshold finden
    for i in range(0, len(estimations)):
        if i not in nan_index:
            if estimations['estimated_pd'][i] >= default_threshold:
                estimations['default_boolean'] = True
            else:
                estimations['default_boolean'] = False
    return estimations


# function that runs all functions for a given dataset
def pd_estimations(data, nan_index):
    indicators = create_indicator_frame(data)
    estimations = calculate_pds(indicators, nan_index)
    default_booleans = create_default_booleans(estimations, nan_index)
    return default_booleans


nan_index = []  # später löschen wenn alle nans entfernt wurden
print(pd_estimations(traindata_m, nan_index))
print(pd_estimations(testdata, nan_index))
