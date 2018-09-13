"""
Here in this regression template we get to know how to make a regression template for fitting all 
other non linear regression models. It will help efficiently to make some regression models for our
truth or bluff problem with the position, and salary.
"""

# Regression Template

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
# no need for this in here for non linear regression model
"""
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
"""

# Fitting the regression model to the dataset
# create your regressor here


# Predicting a new result
y_pred = regressor.predict(6.5)

# Visualising the Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x), color = 'green')
plt.title("Truth or Bluff (Regression Model)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualising the Regression results (for higher resolution and smoother curve)
x_grid = np.arange(min(x), max(x), 0.1)
x_gri = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'green')
plt.title("Truth or Bluff (Regression Model)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
