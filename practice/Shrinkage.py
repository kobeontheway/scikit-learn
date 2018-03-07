import numpy as np
from sklearn import linear_model
from sklearn import datasets

diabetes = datasets.load_diabetes()
diabetes_X_train = diabetes.data[:-20]
diabetes_X_test  = diabetes.data[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test  = diabetes.target[-20:]


X = np.c_[ .5, 1].T
y = [.5, 1]
test = np.c_[ 0, 2].T
regr = linear_model.LinearRegression()

import matplotlib.pyplot as plt
plt.figure()

np.random.seed(0)
for _ in range(6):
   this_X = .1*np.random.normal(size=(2, 1)) + X
   regr.fit(this_X, y)
   plt.plot(test, regr.predict(test))
   plt.scatter(this_X, y, s=3)

#Ridge Regression另外一种方法
regr = linear_model.Ridge(alpha=.1)
plt.figure()
np.random.seed(0)
for _ in range(6):
   this_X = .1 * np.random.normal(size=(2, 1)) + X
   regr.fit(this_X, y)
   plt.plot(test, regr.predict(test))
   plt.scatter(this_X, y, s=3)
alphas = np.logspace(-4, -1, 6)

from __future__ import print_function
print([regr.set_params(alpha=alpha
            ).fit(diabetes_X_train, diabetes_y_train,
            ).score(diabetes_X_test, diabetes_y_test) for alpha in alphas])
regr = linear_model.Lasso()
scores = [regr.set_params(alpha=alpha
            ).fit(diabetes_X_train, diabetes_y_train
            ).score(diabetes_X_test, diabetes_y_test)
       for alpha in alphas]
best_alpha = alphas[scores.index(max(scores))]
regr.alpha = best_alpha
regr.fit(diabetes_X_train, diabetes_y_train)

print(regr.coef_)