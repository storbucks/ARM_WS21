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

# functions that already work
from model import data_merging
from model import data_modification
from model import winsorize_indicators
from model import create_indicator_frame
from model import pd_estimations


#  functions that we have to adjust to get the best result
def calculate_pds(indicators, indicator1, indicator2, indicator3, beta0, beta1, beta2, beta3):
    frame = {'id': indicators.id}
    estimations = pd.DataFrame(frame)
    estimations['estimated_pd'] = ""
    # classification depends on one liquidity ratio (current assets/current liabs), one leverage ratio (debt ratio) and one profitability ratio (roa)
    for i in range(0, len(indicators['id'])):
        # hier müssen wir mit tests noch geeignete allgemeingültige betas finden
        x = beta0 + beta1 * indicator1[i] - beta2 * indicator2[i] - beta3 * indicator3[i]
        pi = (np.exp(x)/(1 + np.exp(x)))
        estimations['estimated_pd'][i] = pi
    return estimations


def create_default_booleans(estimations, threshold):
    estimations['default_boolean'] = ""
    default_threshold = threshold  # geeigneten threshold finden
    for i in range(0, len(estimations)):
        if estimations['estimated_pd'][i] >= default_threshold:
            estimations['default_boolean'] = True
        else:
            estimations['default_boolean'] = False
    return estimations


def create_indicators_for_testing(data):
    new_data = data_merging(data, sector_data)  # add sector variable
    data = data_modification(new_data)  # modify data regarding missing values
    indicators = create_indicator_frame(data)  # calculation of indicators that may be used in the model
    indicators = winsorize_indicators(indicators)
    return indicators


# function to calculate how many D's are right
def evaluate_results(indicators, estimations):
    count_defaults = 0
    count_default_strikes = 0
    count_non_defaults = 0
    count_non_default_strikes = 0
    default_estimations = estimations['default_boolean'].astype(int)

    for i in range(0, len(estimations['id'])):
        if indicators['Default_Dum'][i] == 1:
            count_defaults += 1
            if indicators['Default_Dum'][i] == estimations['default_boolean'].astype(int)[i]:
                count_default_strikes += 1
        else:
            count_non_defaults += 1
            if indicators['Default_Dum'][i] == estimations['default_boolean'].astype(int)[i]:
                count_non_default_strikes += 1

    default_strikes = count_default_strikes/count_defaults
    non_default_strikes = count_non_default_strikes/count_non_defaults
    print("Identified " + str(round(default_strikes * 100, 2)) + "% of defaults and " + str(round(non_default_strikes * 100, 2)) + "% of non_defaults")


# Loading data
traindata_t = pd.read_csv("Training_Dataset.csv", sep=";")
testdata = pd.read_csv("Test_Dataset.csv", sep=";")
sector_data = pd.read_csv("sectors_overview_6.csv", sep=";", dtype={'sector': 'int64', 'sector_string': 'str'})

# shows coefficients that work out well (Hier könnt ihr herum experimentieren, welche Zusammensetzung gut ist; am besten kopieren und dies als Entwurf lassen)
indicators = create_indicators_for_testing(traindata_t)
indicators['Default_Dum'] = traindata_t.default
res2 = sm.Logit.from_formula('Default_Dum ~ equity_ratio + interest_coverage + roa', data=indicators).fit(disp=False, maxiter=100)
print("This is the result of the logit regression.")
print(res2.summary2())

# safes regression betas
param0 = res2.params[0]
param1 = res2.params[1]
param2 = res2.params[2]
param3 = res2.params[3]
#param4 = res2.params[4]
#param5 = res2.params[5]

# uses regression betas for estimation
estimations = calculate_pds(indicators, indicators['debt_ratio'], indicators['current_ratio'], indicators['roa'], param0, param1, param2, param3)
estimations['Default_Dum'] = traindata_t.default
print(estimations['estimated_pd'])
print(estimations['Default_Dum'])

x = np.array(estimations['estimated_pd']).reshape((-1, 1))
y = np.array(estimations['Default_Dum'])
model = LinearRegression()
model.fit(x, y)
model = LinearRegression().fit(x, y)
intercept = float(model.intercept_)
slope = float(model.coef_[0])

print(model.intercept_, model.coef_)

