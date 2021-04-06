# -*- coding:utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold

iris = load_iris()

idx = np.arange(len(iris.data)) + 100

X = pd.DataFrame(iris.data, columns=iris.feature_names, index=idx)
y = pd.DataFrame(iris.target, index=idx)

kf = KFold(n_splits=4, shuffle=True, random_state=42)

print(X.shape, y.shape)
#for train_idx, test_idx in kf.split(X):
for train_idx, test_idx in kf.split(X, y):
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    print(X_train.shape, y_train.shape)
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    print(X_test.shape, y_test.shape)

