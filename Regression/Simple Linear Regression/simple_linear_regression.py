"""
This is a code for the Simple Linear Regression model for predicting the values
from a test set by learning the correlation between the training set data. This
is a machine learning model called Simple Linear Regression.
"""

# Simple Linear Regression

#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 1/3, random_state = 0)

# Feature Scaling
"""
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
"""

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_test_pred = regressor.predict(x_test)

# Predicting the Training set results
y_train_pred = regressor.predict(x_train)

# Visualising the Training set results
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, y_train_pred, color='blue')
plt.title('SALARY VS EXPERIENCE (TRAINING SET)')
plt.xlabel('YEARS OF EXPERIENCE')
plt.ylabel('SALARY')
plt.show()

# Visualising the Test set results
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, y_train_pred, color='blue')
plt.title('SALARY VS EXPERIENCE (TEST SET)')
plt.xlabel('YEARS OF EXPERIENCE')
plt.ylabel('SALARY')
plt.show()
