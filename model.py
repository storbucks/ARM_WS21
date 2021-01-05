import pandas as pd
import numpy as np
import math

# Loading data, just copy the Training_Dataset.csv file into the working directory of your python project:
traindata_m = pd.read_csv("Training_Dataset.csv", sep=";")
testdata = pd.read_csv("Test_Dataset.csv", sep=";")

# hier Funktionen zur Datenbereinigung einfügen

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
