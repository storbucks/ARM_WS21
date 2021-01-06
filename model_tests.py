import pandas as pd
import numpy as np

import statsmodels.api as sm
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sns
import math
from sklearn import model_selection

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
# set of all variables we consider: 'interest_coverage', 'roa', 'debt_ratio', 'debt_to_equity_ratio', 'equity_ratio',
#                                   'ebit_margin', 'cf_operating', 'current_ratio', 'age'

# could choose the full model  𝑉  - which is probably the model producing the highest  𝐴𝑈𝐶  in-sample. Nevertheless, will it perform well out-of-sample?

# adjusting the training error and the cross-validation approach and the application of bootstrapping
# Training Error: We get the by calculating the classification error of a model on the same data the model was trained on (just like the example above).
# Test Error: We get this by using two completely disjoint datasets: one to train the model and the other to calculate the classification error. Both datasets need to have values for y.
# The first dataset is called training data and the second, test data.

# results in two values of y:  the actual one (default), as well as the prediction from the model, which we will call p.
# comparing the predictions in p to the true values in y – this is called the classification error.
# count how often the values for y and p differ in this table, and then divide this count by the number of rows in the table

indicators = create_indicators(traindata_t)
# heatmap
f, ax = plt.subplots(figsize=(20,5))
sns.heatmap(indicators[2:].corr(method='pearson'),
            annot=True,cmap="coolwarm",
            vmin=-1, vmax=1, ax=ax);

plt.show()

indicators['Default_Dum'] = traindata_t.default

# hier könnt ihr herumspielen und euch die pseudo r squared anschauen
mdl1 = sm.Logit.from_formula('Default_Dum ~ interest_coverage + roa + debt_ratio + equity_ratio + ebit_margin + current_ratio + age', data=indicators).fit(disp=False, maxiter=100)
mdl2 = sm.Logit.from_formula('Default_Dum ~ debt_ratio + current_ratio + roa', data=indicators).fit(disp=False, maxiter=100)
print(mdl1.summary2())
print(mdl2.summary2())

print('======================= Model 1 vs. Model 2 =================\n')
print('Pseudo R2:       {}           {}\n'.format(mdl1.prsquared, mdl2.prsquared))
print('AIC:      {}           {}\n'.format(mdl1.aic, mdl2.aic))  # the lower the better
print('BIC:      {}             {}'.format(mdl1.bic, mdl2.bic))  # the lower the better


# main idea of the cross-validation approach
# repeatedly draw a subset from your available sample
# for each of these subsets, estimate your model
# evaluate each estimated subset-model on the data not included in this subset - on the validation sample

# randomly divide the full sample into two subsets - the new "training sample" and the "validation sample" (Validation-Set Approach)

#K-Fold Approach
#randomly dividing your sample into K folds of approx. equal size with distinct observations
#each time of the K estimation, one fold is used for validation and K-1 folds for estimation
#if K is large, you can use other metrics, like the AUC, to evaluate the model performance and can evaluate its distribution
#Repeated K-Fold Approach
#same as K-Fold approach, but repeated N times
#different random numbers are used to create different folds of size K
X = indicators.iloc[:, 1:len(indicators)-1].values # last row: Default_Dum not included
y = indicators.Default_Dum.values

kf = sk.model_selection.KFold(n_splits=13, random_state=23, shuffle=True)
kf.get_n_splits(X)

print(kf)

mse1 = []
mse2 = []

for train_index, test_index in kf.split(X):

    # Estimate Model 1
    mdl1 = sm.OLS(y[train_index], X[train_index, 0:2]).fit() # index 0-2: current, roa, debt; könnt andere ausprobieren

    # Prediction Model 1
    pred1 = mdl1.predict(X[test_index, 0:2])

    # Estimate Model 2
    mdl2 = sm.OLS(y[train_index], X[train_index, :]).fit()  # all indicators

    # Prediction Model 2
    pred2 = mdl2.predict(X[test_index, :])

    # Calculate MSEs
    mse1.append(np.mean((pred1 - y[test_index])**2))
    mse2.append(np.mean((pred2 - y[test_index])**2))

mse1 = np.array(mse1)
mse2 = np.array(mse2)

f, ax = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

ax[0].boxplot(mse1)
ax[1].boxplot(mse2)

ax[0].set_title('Model 1')
ax[1].set_title('Model 2')

plt.show()

print(pd.DataFrame({'M1': mse1, 'M2': mse2}).describe())

mdl1 = sm.OLS(y, X[:,0:2]).fit()
mdl2 = sm.OLS(y, X).fit()

data_test = generate_sample(500, 999)

pred1 = mdl1.predict(data_test.iloc[:,1:3].values)
pred2 = mdl2.predict(data_test.iloc[:,1:].values)

mse1 = ((data_test['y'] - pred1)**2).mean()
mse2 = ((data_test['y'] - pred2)**2).mean()

print(mse1)
print(mse2)

# zum erstellen von Excel-Dateien
# indicators
#import xlsxwriter
#with xlsxwriter.Workbook('indicators.xlsx') as workbook:
 #   worksheet = workbook.add_worksheet()
  #  for row_num, data in enumerate(indicators):
   #     worksheet.write_row(row_num, 0, data)
