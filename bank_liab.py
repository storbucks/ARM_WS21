import pandas as pd
import numpy as np

traindata = pd.read_csv("Training_Dataset.csv", sep=';')

traindata["bank_liabilities_st"].fillna(0, inplace=True)
for i in range(0, len(traindata.bank_liabilities_st)):
    if traindata.bank_liabilities_st[i] != 0:
        traindata.bank_liabilities_st[i] = 1

traindata.bank_liabilities_st.head()
