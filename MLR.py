#mulitple linear regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


##importing dataset
#file = "Haberman.csv"
#dataset = pf.read_csv(file)
#x = dataset.iloc[:, :-1].values
#y = dataset.iloc[:,3].values

from sklearn.datasets import load_boston
x,y = load_boston(True)

#encoding the categorical data
from sklearn.preprocessing import LabelEncoder
#encoding the dependant variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#splitting dataset into the training and testing datasets
from sklearn.cross_validation import train_test_split
X_train , X_test, Y_train, Y_test = train_test_split(x,y, test_size = 0.33, random_state = 0)

#fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#predicting the y values for the x test values using the regressor
y_predicted = regressor.predict(X_test)
#test = [0 for i in range(13)]
#test_pred = regressor.predict(np.array(test).reshape(1,-1))
