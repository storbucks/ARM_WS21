import pandas as pd
import numpy as np

# Loading data
traindata_m = pd.read_csv("Training_Dataset.csv", sep=";")
testdata = pd.read_csv("Test_Dataset.csv", sep=";")
sector_data = pd.read_csv("sectors_overview_6.csv", sep=";", dtype={'sector': 'int64', 'sector_string': 'str'})


# Merging data & sector data & filling sector_string NA's with unknown
def data_merging(data, sectors):
    data = pd.merge(data, sectors, on='sector', how='left')
    data['sector_string'] = data['sector_string'].fillna('Unknown')
    return data


def data_modification(data):
    # Build some groups in dataset based on codebook
    pl_vars = data.loc[:, "sales":"annual_profit"]
    bs_vars = data.loc[:, "total_assets":"trade_receivables_lt"]
    cf_vars = data.loc[:, "cf_operating":"cf_financing"]
    special_vars = data.loc[:, "sales":"sector_string"]

    # Build some groups to use as indices when accessing traindata
    catvar = [i for i in list(data.columns) if data[i].dtype == 'O']  # category variables
    numvar = [i for i in list(data.columns) if data[i].dtype in ['float64', 'int64']]  # numerical variables
    boolvar = [i for i in list(data.columns) if data[i].dtype == bool]  # boolean variables

    # Overview over NA's
    pl_na_overview = pd.DataFrame({'Valid': pl_vars.notnull().sum(),
                                   'NAs': pl_vars.isnull().sum(),
                                   'NAs of total': pl_vars.isnull().sum() / pl_vars.shape[0]}).sort_values('NAs of total', ascending=True)
    print(pl_na_overview)

    bs_na_overview = pd.DataFrame({'Valid': bs_vars.notnull().sum(),
                                   'NAs': bs_vars.isnull().sum(),
                                   'NAs of total': bs_vars.isnull().sum() / bs_vars.shape[0]}).sort_values('NAs of total', ascending=True)
    print(bs_na_overview)

    cf_na_overview = pd.DataFrame({'Valid': cf_vars.notnull().sum(),
                                   'NAs': cf_vars.isnull().sum(),
                                   'NAs of total': cf_vars.isnull().sum() / cf_vars.shape[0]}).sort_values('NAs of total', ascending=True)
    print(cf_na_overview)

    # Storing sector specific Means of Numerical variables
    special_vars_mean = special_vars.groupby("sector_string").mean()
    pl_vars_mean = pl_vars.mean()
    bs_vars_mean = bs_vars.mean()
    cf_vars_mean = cf_vars.mean()  # not necessary regarding the chosen ratios

    # Manipulation - Substituing NA's
    data["earn_from_op"].fillna(pl_vars_mean["earn_from_op"])
    data["total_assets"].fillna(bs_vars_mean["total_assets"])
    data["total_result"].fillna(pl_vars_mean["total_result"])
    data["total_liabilities_st"].fillna(bs_vars_mean["total_liabilities_st"])
    data["total_liabilities_mt"].fillna(bs_vars_mean["total_liabilities_mt"])
    data["total_liabilities_lt"].fillna(bs_vars_mean["total_liabilities_lt"])
    # data["total_equity"].fillna(special_vars_mean["total_equity"])  # another approach could be useful
    data["sales"].fillna(pl_vars_mean["sales"])
    data["current_assets"].fillna(bs_vars_mean["current_assets"])

    # Dealing with na: ICR
    total_liabilities = data.total_liabilities_st.copy() + data.total_liabilities_mt.copy() + data.total_liabilities_lt.copy()
    interest_exp_rate = data.oth_interest_exp.copy() / total_liabilities
    oth_interest_exp_filler = []
    for i in range(0, len(data.oth_interest_exp)):
        oth_interest_exp_filler.append(interest_exp_rate.mean() * total_liabilities[i])
        data.oth_interest_exp.fillna(oth_interest_exp_filler[i], inplace=True)
    total_equity = data.total_assets.copy() - (data.total_liabilities_st.copy() + data.total_liabilities_mt.copy() + data.total_liabilities_lt.copy())
    for i in range (0, len(data.total_equity)):
        data['total_equity'].fillna(total_equity[i], inplace=True)
    return data


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
    # create data frame including all indicators
    frame = {'id': data.id, 'interest_coverage': interest_coverage, 'roa': roa, 'debt_ratio': debt_ratio,
             'debt_to_equity_ratio': debt_to_equity_ratio, 'equity_ratio': equity_ratio,
             'ebit_margin': ebit_margin, 'current_ratio': current_ratio, 'age': age}
    indicators = pd.DataFrame(frame)
    return indicators


#  calculate the values to classify companies in default and non-default
def calculate_pds(indicators):
    frame = {'id': indicators.id}
    estimations = pd.DataFrame(frame)
    estimations['estimated_pd'] = ""
    # classification depends on one liquidity ratio (current assets/current liabs), one leverage ratio (debt ratio) and one profitability ratio (roa)
    for i in range(0, len(indicators['id'])):
        # hier müssen wir mit tests noch geeignete allgemeingültige betas finden
        x = -3.4163441714312563 + 0.9435381378963345 * indicators['debt_ratio'][i] - 1.0848159927024499 * indicators['current_ratio'][i] - (0.00010046357934681097/10000) * indicators['roa'][i]
        pi = (np.exp(x)/(1 + np.exp(x)))
        estimations['estimated_pd'][i] = pi
    return estimations


def create_default_booleans(estimations):
    estimations['default_boolean'] = ""
    default_threshold = 0.06162628810257741  # geeigneten threshold finden
    for i in range(0, len(estimations)):
        if estimations['estimated_pd'][i] >= default_threshold:
            estimations['default_boolean'] = True
        else:
            estimations['default_boolean'] = False
    return estimations


# function that runs all functions for a given dataset
def pd_estimations(data):
    data = data_merging(data, sector_data)  # add sector variable
    data = data_modification(data)  # modify data regarding missing values
    indicators = create_indicator_frame(data)  # calculation of indicators that may be used in the model
    estimations = calculate_pds(indicators)  # calculate values with logit regression betas
    default_booleans = create_default_booleans(estimations)  # declare companies that stride a fixed threshold as defaulted
    return default_booleans  # returns a matrix with the default booleans


estimations_traindata = pd_estimations(traindata_m)
estimations_testdata = pd_estimations(testdata)
