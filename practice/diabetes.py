import numpy as np
from sklearn import linear_model
from sklearn import datasets

diabetes = datasets.load_diabetes()
diabetes_X_train = diabetes.data[:-20]
diabetes_X_test  = diabetes.data[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test  = diabetes.target[-20:]


regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)

print regr.coef_

# The mean square error
print np.mean((regr.predict(diabetes_X_test)-diabetes_y_test)**2)


# Explained variance score: 1 is perfect prediction
# and 0 means that there is no linear relationship
# between X and y.
print regr.score(diabetes_X_test, diabetes_y_test)