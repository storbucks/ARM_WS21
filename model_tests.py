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

from playground import indicators
from playground import history
from playground import nan_index
from model import pd_estimations


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
        if history.pd[i] >= (intercept+(intercept+slope))/2: # mit intercept+slope weniger D's als mit intercept, aber mehr non-D's richtig
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

print("Identified " + str(round(default_strikes*100, 2)) + "% of defaults and " + str(round(non_default_strikes*100,2)) + "% of non_defaults")

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

def model_validation(history,start,end):
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
    print("Identified " + str(round(default_strikes*100, 2)) + "% of defaults and " + str(round(non_default_strikes*100,2)) + "% of non_defaults")

print("subset_1")
model_validation(history,0,224)
print("subset_2")
model_validation(history,224,447)
print("subset_3")
model_validation(history,447,669)


