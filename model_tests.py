import pandas as pd
import numpy as np
import scipy as sci

import statsmodels.api as sm
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sns
import math
from sklearn import model_selection
from sklearn import metrics

from model import data_merging
from model import data_modification
from model import winsorize_indicators
from model import create_indicator_frame

traindata_t = pd.read_csv("Training_Dataset.csv", sep=";")
testdata = pd.read_csv("Test_Dataset.csv", sep=";")
sector_data = pd.read_csv("sectors_overview_6.csv", sep=";", dtype={'sector': 'int64', 'sector_string': 'str'})

#  calculate the values to classify companies in default and non-default
def calculate_pds(indicators):
    frame = {'id': indicators.id}
    estimations = pd.DataFrame(frame)
    estimations['estimated_pd'] = ""
    # classification depends on one liquidity ratio (current assets/current liabs), one leverage ratio (debt ratio) and one profitability ratio (roa)
    for i in range(0, len(indicators['id'])):
        # hier müssen wir mit tests noch geeignete allgemeingültige betas finden und gute Indikatoren
        x = -3 + 0.7 * indicators['debt_ratio'][i] - 0.19 * indicators['current_ratio'][i] - 1 * indicators['roa'][i]
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


def pd_estimations(data):
    new_data = data_merging(data, sector_data)  # add sector variable
    data = data_modification(new_data)  # modify data regarding missing values
    new_indicators = create_indicator_frame(data)  # calculation of indicators that may be used in the model
    indicators = winsorize_indicators(new_indicators)
    estimations = calculate_pds(indicators)  # calculate values with logit regression betas
    default_booleans = create_default_booleans(estimations)  # declare companies that stride a fixed threshold as defaulted
    return default_booleans

def create_indicators(data):
    new_data = data_merging(data, sector_data)  # add sector variable
    data = data_modification(new_data)  # modify data regarding missing values
    new_indicators = create_indicator_frame(data)  # calculation of indicators that may be used in the model
    indicators = winsorize_indicators(new_indicators)
    return indicators

####################
# Model valuation  #
####################
indicators = create_indicators(traindata_t)
indicators['Default_Dum'] = traindata_t.default

# heatmap
f, ax = plt.subplots(figsize=(20,5))
sns.heatmap(indicators[2:].corr(method='pearson'),
            annot=True,cmap="coolwarm",
            vmin=-1, vmax=1, ax=ax);

plt.show()

# hier könnt ihr herumspielen und euch die pseudo r squared anschauen
mdl1 = sm.Logit.from_formula('Default_Dum ~ debt_ratio + bank_liab_st + interest_coverage + current_assets_ratio', data=indicators).fit(disp=False, maxiter=100)
mdl2 = sm.Logit.from_formula('Default_Dum ~ debt_ratio + bank_liab_st + interest_coverage + op_cash_flow + current_assets_ratio + ebit_margin', data=indicators).fit(disp=False, maxiter=100)
mdl3 = sm.Logit.from_formula('Default_Dum ~ debt_ratio + bank_liab_st + interest_coverage + op_cash_flow + current_assets_ratio + roa', data=indicators).fit(disp=False, maxiter=100)
mdl4 = sm.Logit.from_formula('Default_Dum ~ debt_ratio + bank_liab_st + interest_coverage + op_cash_flow + current_assets_ratio + current_ratio', data=indicators).fit(disp=False, maxiter=100)
print(mdl1.summary2())
print(mdl2.summary2())
print(mdl3.summary2())
print(mdl4.summary2())

print('================================= Model Comparison =================================\n')
print('Pseudo R2:   {}    {}    {}    {} \n'.format(mdl1.prsquared, mdl2.prsquared, mdl3.prsquared, mdl4.prsquared))
print('AIC:         {}    {}    {}    {} \n'.format(mdl1.aic, mdl2.aic, mdl3.aic, mdl4.aic))  # the lower the better
print('BIC:         {}    {}    {}    {} \n'.format(mdl1.bic, mdl2.bic, mdl3.bic, mdl4.bic))  # the lower the better



### GINI COEFF ###
pd_pred1 = mdl1.predict(exog=indicators)
pd_pred2 = mdl2.predict(exog=indicators)
pd_pred3 = mdl3.predict(exog=indicators)
pd_pred4 = mdl4.predict(exog=indicators)
# print(pd_pred.head())

# AUC Model 1
fpr1, tpr1, thresholds1 = metrics.roc_curve(indicators.Default_Dum, pd_pred1)
auc1 = metrics.auc(fpr1, tpr1)
# AUC Model 2
fpr2, tpr2, thresholds2 = metrics.roc_curve(indicators.Default_Dum, pd_pred2)
auc2 = metrics.auc(fpr2, tpr2)
# AUC Model 3
fpr3, tpr3, thresholds3 = metrics.roc_curve(indicators.Default_Dum, pd_pred3)
auc3 = metrics.auc(fpr3, tpr3)
# AUC Model 4
fpr4, tpr4, thresholds4 = metrics.roc_curve(indicators.Default_Dum, pd_pred4)
auc4 = metrics.auc(fpr4, tpr4)

# print("AUC_1: " + str(auc1))
# print("AUC_2: " + str(auc2))
# print("AUC_3: " + str(auc3))
# print("AUC_4: " + str(auc4))
# print("AUC_5: " + str(auc5))

# GINI
gini1 = 2 * auc1 - 1
gini2 = 2 * auc2 - 1
gini3 = 2 * auc3 - 1
gini4 = 2 * auc4 - 1
print("Gini_1: ", gini1)
print("Gini_2: ", gini2)
print("Gini_3: ", gini3)
print("Gini_4: ", gini4)

fig, axes = plt.subplots(figsize=(15,5))
lw = 2
axes = plt.plot(fpr1, tpr1, color='green',
         lw=lw, label='ROC curve (area = %0.2f)' % auc1)
axes = plt.plot(fpr2, tpr2, color='darkred',
         lw=lw, label='ROC curve (area = %0.2f)' % auc2)
axes = plt.plot(fpr3, tpr3, color='blue',
         lw=lw, label='ROC curve (area = %0.2f)' % auc3)
axes = plt.plot(fpr4, tpr4, color='purple',
         lw=lw, label='ROC curve (area = %0.2f)' % auc4)
axes = plt.plot([0, 1], [0, 1], color='lightblue', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

#%%
#### K Fold apporach ###
"""Regression: debt_ratio + bank_liab_st + interest_coverage + op_cash_flow + current_assets_ratio + roa"""

ratios = ['roa', 'debt_ratio', 'interest_coverage', 'op_cash_flow', 'current_assets_ratio', 'bank_liab_st']

X = indicators.loc[:, ratios].values
y = indicators.Default_Dum.values

kf = sk.model_selection.KFold(n_splits=13, random_state=23, shuffle=True)
kf.get_n_splits(X)

mse1 = []
mse2 = []

for train_index, test_index in kf.split(X):
    # Estimate Model 1
    mdl1 = sm.OLS(y[train_index], X[train_index]).fit()  # muss mit oben übereinstimmen
    # Prediction Model 1
    pred1 = mdl1.predict(X[test_index])

    # # Estimate Model 2
    # mdl2 = sm.OLS(y[train_index], X[train_index, 0:5]).fit()
    # # Prediction Model 2
    # pred2 = mdl2.predict(X[test_index, 0:5])

    # Calculate MSEs
    mse1.append(np.mean((pred1 - y[test_index]) ** 2))
    # mse2.append(np.mean((pred2 - y[test_index]) ** 2))

mse1 = np.array(mse1)
# mse2 = np.array(mse2)

f, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

ax[0].boxplot(mse1)
# ax[1].boxplot(mse2)

ax[0].set_title('Model 1')
# ax[1].set_title('Model 2')

plt.show()

# print(pd.DataFrame({'M1': mse1, 'M2': mse2}).describe())


mdl1 = sm.OLS(y, X).fit()
# mdl2 = sm.OLS(y, X[:, 0:5]).fit()

data_test = indicators
pred1 = mdl1.predict(data_test.loc[:, ratios])
# pred2 = mdl2.predict(data_test.iloc[:,1:6].values)

mse1 = ((data_test['Default_Dum'] - pred1)**2).mean()
# mse2 = ((data_test['Default_Dum'] - pred2)**2).mean()

print("MSE_m1:", mse1)
# print("MSE_m2:", mse2)


# zum erstellen von Excel-Dateien
# indicators
#import xlsxwriter
#with xlsxwriter.Workbook('indicators.xlsx') as workbook:
 #   worksheet = workbook.add_worksheet()
  #  for row_num, data in enumerate(indicators):
   #     worksheet.write_row(row_num, 0, data)
