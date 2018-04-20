#
# Created by Diego Santos Seabra
#

import matplotlib.pyplot as py
import seaborn as sb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Read the CSV File
df = pd.read_csv('data.csv')

# Replaces Male -> 0
df['sex'] = df['sex'].replace('Male', 0)

# Replaces Female -> 1
df['sex'] = df['sex'].replace('Female', 1)

# Sets the X and Y Dataframe Values
X = df[['weight', 'height']]
Y = df[['sex']]

# Trains the A.I.
X_train, X_test, y_train, y_test = train_test_split(X, Y)

# Fits the data in the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Tests the Model
predictions = model.predict(X_test)

# Plots the Test and Predictions Chart
sb.distplot(y_test - predictions, axlabel="Test - Prediction")
py.show()

# Predicts the Sex given the Input (Change Here)
# 94 -> 94 Kg (Weight)
# 182 -> 182 cm (Height)

myvals = np.array([94, 182]).reshape(1, -1)
print(model.predict(myvals))

# The more close to 0, the more is a Male
# The more close to 1, the more is a Female

