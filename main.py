import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.anova as anova
import scipy as sci
# hey was geht
from sklearn import metrics

# Loading data, just copy the Training_Dataset.csv file into the working directory of your python project:
traindata = pd.read_csv("Training_Dataset.csv", sep=";")

# Default as a boolean
traindata.default = traindata.default.replace([0, 1], [False, True])

# Run some checks if you want to:
print(traindata.head())
print(traindata.tail())
print(traindata.isnull().sum().sort_values(ascending=False))

# Display information about the dataset at a glance:
print(traindata.info())  # Output: 40 cols, 669 rows, dtypes: float, int, object(here: strings)
catvar = [i for i in list(traindata.columns) if traindata[i].dtype=='O']  # category variables
numvar = [i for i in list(traindata.columns) if traindata[i].dtype in ['float64','int64']]  # numerical variables
boolvar = 'default'

# Check for missing values:
print(traindata.isnull().sum())

# test

# Test OLS regression made by me
testreg = smf.ols(formula="sales ~ gross_profit", data=traindata)
res = testreg.fit()
print(res.summary2())

# description of boolean variable
print(traindata.default.describe())
print(traindata.default.astype(float).describe())

# mean, st. deviation ... of all variables
print(traindata.describe())

# frequency of categories (absolute and in percent)
for i in catvar[1:]:  # makes no sense for ID, therefore not for index 0 in catvar
    print('============================================')
    print(f'Variable: {i} \n')
    x1 = traindata[i].value_counts()
    x2 = x1 / np.sum(x1) * 100
    x = pd.concat([x1,x2], axis=1)
    x.columns = ['Count', 'in %']
    print(x)
    print()

# Descriptive analysis
print(traindata[numvar+[boolvar]].corr())

# hier müssen wir dann in die eckigen Klammern die vars einfügen, für die wir die correlation visualisieren möchten
# corrvar = [hier einfügen] wenn es keine Liste ist, sondern eine Bezeichnung [[Bez] + ..]
#f, ax = plt.subplots(figsize=(15,5))
#sns.heatmap(traindata[corrvar].corr(method='pearson'),
#            annot=True,cmap="coolwarm",
#            vmin=-1, vmax=1, ax=ax);


