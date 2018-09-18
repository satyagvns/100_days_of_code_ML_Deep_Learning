print(__doc__)

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pn

diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data[:, np.newaxis, 2]

train_diabetes_X = diabetes_X[:-20]
test_diabetes_X = diabetes_X[-20:]

#target training and testing
train_diabetes_Y = diabetes.target[:-20]
test_diabetes_Y = diabetes.target[-20:]

regression = linear_model.LinearRegression()

regression.fit(train_diabetes_X,train_diabetes_Y)

diabetes_Y_predict = regression.predict(test_diabetes_X)

print('Coefficients: \n', regression.coef_)
#print('Mean Squared Error: %2f', mean_squared_error(test_diabetes_Y, diabetes_Y_predict))
print('Variance Score: %2f' % r2_score(test_diabetes_Y, diabetes_Y_predict))

plt.scatter(test_diabetes_X, test_diabetes_Y, color='black')
plt.plot(test_diabetes_X, diabetes_Y_predict, color='blue', linewidth=3)
plt.xticks(np.arange(-0.1, 0.1, step=0.02))
plt.yticks()

plt.show()
