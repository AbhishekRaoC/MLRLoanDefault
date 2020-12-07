import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import pandas as pd


from sklearn.datasets import load_boston
data = load_boston()

df = pd.DataFrame(data['data'], columns= data['feature_names'])
df['price'] = data['target']

#print(df.describe())
df.fillna(value=-99999, inplace = True)

X = np.array(df.drop(['price'],1))
X = preprocessing.scale(X)
y = np.array(df['price'])

#splitting dataset into the training and testing datasets
X_train , X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size = 0.2)

#fitting the data to the given classifier
clf = svm.SVR(gamma = 'auto')
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print("confidence of support vector machine ", confidence)

linclf = LinearRegression()
linclf.fit(X_train, y_train)
confidence = linclf.score(X_test, y_test)
print("Confidence of linear regression model ", confidence ,'\n')

#predicting the y values for the x test values using the regressor
y_pred = linclf.predict(X_test)
print("Predicted value {}".format(np.round(y_pred[:7], 1)))
print("Actual Value    {}".format(np.round(y_test[:7], 1)))
#^ compare actual to predicted for first 5