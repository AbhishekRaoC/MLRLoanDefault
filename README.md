Written By Ryan Boldi, Abhishek Rao Chimbili and Abdulla Alattar

The files for this program are MLR.py. It uses sklearn's boston dataset to predict house prices based on a bunch of independant variables.

There are two different classifiers that I tried to use, a support vector machine, and a simple linear regression model.
The linear regression model had much higher confidence and accuracy than the SVM, so I based all the predictions off of the linear regression model.


## Boston Dataset:
### Data Set Characteristics:

### Number of Instances: 506

### Number of Attributes: 13 numeric/categorical predictive

### Median Value (attribute 14) is the target for this project

### Attribute Information (in order):
- CRIM per capita crime rate by town
- ZN proportion of residential land zoned for lots over 25,
000 sq.ft.
- INDUS proportion of non-retail business acres per town
- CHAS Charles River dummy variable (= 1 if tract bounds rive
r; 0 otherwise)
- NOX nitric oxides concentration (parts per 10 million)
- RM average number of rooms per dwelling
- AGE proportion of owner-occupied units built prior to 1940
- DIS weighted distances to five Boston employment centres
- RAD index of accessibility to radial highways
- TAX full-value property-tax rate per $10,000
- PTRATIO pupil-teacher ratio by town
- B 1000(Bk - 0.63)^2 where Bk is the proportion of blacks
by town
- LSTAT % lower status of the population
- MEDV Median value of owner-occupied homes in $1000's

### Creator: Harrison, D. and Rubinfeld, D.L.

## Can predict MEDV if given the 13 Independant variables

