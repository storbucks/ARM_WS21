import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import model_selection

from playground import indicators
from playground import history
from playground import nan_index

x = np.array(history['default_dum']).reshape((-1, 1))
y = np.array(history['pd'])
model = LinearRegression()
model.fit(x, y)
model = LinearRegression().fit(x, y)
intercept = float(model.intercept_)
slope = float(model.coef_[0])

print("pd threshold: " + str((intercept+(intercept+slope))/2))

for i in range(0, len(history['id'])):
    if i not in nan_index:
        if history.pd[i] >= (intercept+(intercept+slope))/2:  # mit intercept+slope weniger D's als mit intercept, aber mehr non-D's richtig
            history.estimation[i] = 1
        else:
            history.estimation[i] = 0

count_defaults = 0
count_default_strikes = 0

count_non_defaults = 0
count_non_default_strikes = 0

for i in range(0, len(history['id'])):
    if i not in nan_index:
        if history.default_dum[i] == 1:
            count_defaults += 1
            if history.default_dum[i] == history.estimation[i]:
                count_default_strikes += 1
        else:
            count_non_defaults += 1
            if history.default_dum[i] == history.estimation[i]:
                count_non_default_strikes += 1

default_strikes = count_default_strikes/count_defaults
non_default_strikes = count_non_default_strikes/count_non_defaults

print("Identified " + str(round(default_strikes * 100, 2)) + "% of defaults and " + str(round(non_default_strikes * 100, 2)) + "% of non_defaults")

#import xlsxwriter
#with xlsxwriter.Workbook('pds.xlsx') as workbook:
 #   worksheet = workbook.add_worksheet()
  #  for row_num, data in enumerate(pds):
   #     worksheet.write_row(row_num, 0, data)

subset_1 = indicators[:223]
subset_2 = indicators[223:446]
subset_3 = indicators[446:]

res2 = sm.Logit.from_formula('Default_Dum ~ debt_ratio + current_ratio + roa', data=subset_1).fit(disp=False, maxiter=100)
print(res2.summary2())

x = np.array(history[0:224]['default_dum']).reshape((-1, 1))
y = np.array(history[0:224]['pd'])
model = LinearRegression()
model.fit(x, y)
model = LinearRegression().fit(x, y)
intercept_1 = float(model.intercept_)
slope_1 = float(model.coef_[0])

res2 = sm.Logit.from_formula('Default_Dum ~ debt_ratio + current_ratio + roa', data=subset_2).fit(disp=False, maxiter=100)
print(res2.summary2())

x = np.array(history[224:447]['default_dum']).reshape((-1, 1))
y = np.array(history[224:447]['pd'])
model = LinearRegression()
model.fit(x, y)
model = LinearRegression().fit(x, y)
intercept_2 = float(model.intercept_)
slope_2 = float(model.coef_[0])

res2 = sm.Logit.from_formula('Default_Dum ~ debt_ratio + current_ratio + roa', data=subset_3).fit(disp=False, maxiter=100)
print(res2.summary2())

x = np.array(history[447:669]['default_dum']).reshape((-1, 1))
y = np.array(history[447:669]['pd'])
model = LinearRegression()
model.fit(x, y)
model = LinearRegression().fit(x, y)
intercept_3 = float(model.intercept_)
slope_3 = float(model.coef_[0])

print("mean intercept: " + str((intercept_1+intercept_2+intercept_3)/3))
print("mean slope: " + str((slope_1+slope_2+slope_3)/3))


def model_validation(history, start, end):
    count_defaults = 0
    count_default_strikes = 0
    count_non_defaults = 0
    count_non_default_strikes = 0

    for i in range(start, end):
        if i not in nan_index:
            if history.default_dum[i] == 1:
                count_defaults += 1
                if history.default_dum[i] == history.estimation[i]:
                    count_default_strikes += 1
            else:
                count_non_defaults += 1
                if history.default_dum[i] == history.estimation[i]:
                    count_non_default_strikes += 1

    default_strikes = count_default_strikes/count_defaults
    non_default_strikes = count_non_default_strikes/count_non_defaults
    print("Identified " + str(round(default_strikes*100, 2)) + "% of defaults and " + str(round(non_default_strikes*100, 2)) + "% of non_defaults")


print("subset_1")
model_validation(history, 0, 224)
print("subset_2")
model_validation(history, 224, 447)
print("subset_3")
model_validation(history, 447, 669)

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

mdl1 = sm.Logit.from_formula('Default_Dum ~ interest_coverage + roa + debt_ratio + equity_ratio + ebit_margin + cf_operating + current_ratio + age', data=indicators).fit(disp=False, maxiter=100)
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

for j in range(2, 12):
    for i in range(0, 669):
        if math.isnan(indicators[i,j]):
            indicators = indicators.drop(i)

X =  indicators.iloc[:,2:11].values
y = indicators.Default_Dum.values

kf = sk.model_selection.KFold(n_splits=13, random_state=23, shuffle=True)
kf.get_n_splits(X)

print(kf)

mse1 = []
mse2 = []

for train_index, test_index in kf.split(X):
    # Estimate Model 1
    mdl1 = sm.OLS(y[train_index], X[train_index, 0:2]).fit()
    # Prediction Model 1
    pred1 = mdl1.predict(X[test_index, 0:2])
    # Estimate Model 2
    mdl2 = sm.OLS(y[train_index], X[train_index, :]).fit()
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
