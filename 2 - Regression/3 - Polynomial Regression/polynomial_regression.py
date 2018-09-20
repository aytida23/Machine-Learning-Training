"""
This is the code for Polynomial Regression where we predict whether the employee is talking the
truth or bluff about his/her past company's salary he/she used to get, we use a dataset where we
have the dataset of different position with their levels and salary of their previous company.
"""
# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
# always remember to keep the x as matrix by adding extra column and y as vector
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
# for this dataset we won't be needing to create training set and test set because of not much data available
# and also we need to predict it most accurately so will be needing much info for that purpose here
# Data is not so big to train and test by learning correlation from training set for prediction of salaries here
# putting below lines of code in docstring
"""
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
"""

# Feature Scaling
# no need for this in here
"""
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
"""

# Fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y) #polynomial regression model is created and ready to reveal truth or bluff 

# Visualising the Linear Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'green')
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()


# to create more advanced plot of the curve, we can have the prediction from 1 to 10 incremented by
# higher resolution like 0.1 step by sacrificing some simplicity in the code
# create new x here called x_grid that contain all levels with resolution of 0.1 step
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))

# Visualising the Polynomial Regression results
#you will get a more continuous curve which is actually the real curve of polynomial regression itself
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color = 'green')
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Predicting a new result with Linear Regression
# the employee of position level 6.5's predicted salary only on Linear Regression 
lin_reg.predict(6.5)

# Predicting a new result with Polynomial Regression
# the employee of position level 6.5's predicted salary only on Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))