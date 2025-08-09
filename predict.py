import pandas as pd
import joblib
import numpy as np

clf = joblib.load('svm_model.pkl')

data = pd.read_excel('test.xlsx')

for i in range(data.shape[1]):
    tempx = data.iloc[:,i].values
    tempx = tempx.reshape(1,-1)
    y_pred = clf.predict(tempx)
    print(y_pred)