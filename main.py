import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.anova as anova
import scipy as sci
# hey
from sklearn import metrics

# Loading data, just copy the Training_Dataset.csv file into the working directory of your python project:
traindata = pd.read_csv("Training_Dataset.csv", sep=";")

# Run some checks if you want to:
print(traindata.head())
print(traindata.tail())
print(traindata.isnull().sum().sort_values(ascending=False))

# Display information about the dataset at a glance:
print(traindata.info())  # Output: 40 cols, 669 rows, dtypes: float, int, object(here: strings)
catvar = [i for i in list(traindata.columns) if traindata[i].dtype=='O']  # category variables
numvar = [i for i in list(traindata.columns) if traindata[i].dtype in ['float64','int64']]  # numerical variables

# Check for missing values:
print(traindata.isnull().sum())

# test

# Test OLS regression made by me
testreg = smf.ols(formula="sales ~ gross_profit", data=traindata)
res = testreg.fit()
print(res.summary2())
