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

# Deletes the Name Column (protecting the People Data)
del df['name']

# ax = sb.barplot(x="sex", y="height", data=df)
# ax = sb.barplot(x="sex", y="weight", data=df)

# df.plot(kind='pie', subplots=True, autopct='%1.1f%%',
#  startangle=90, shadow=False, labels=df['sex'], legend = False, fontsize=14)

# df.groupby(df['sex']).sum().plot(kind='pie', subplots=True, autopct='%1.1f%%')
# py.axis('equal')

# Replaces Male -> 0
df['sex'] = df['sex'].replace('Male', 0)

# Replaces Female -> 1
df['sex'] = df['sex'].replace('Female', 1)

X = df[['weight','height']]
Y = df[['sex']]

X_train, X_test , y_train , y_test = train_test_split(X,Y)

# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)

# print(X_train.shape)
# print(y_train.shape)

model = LinearRegression()
model.fit(X_train, y_train)

predictions=model.predict(X_test)

# print(X_test)

# print(predictions)

# g = sb.lmplot(x="weight", y="height", hue="sex",
#                truncate=True, size=5, data=df)

# g = sb.lmplot(x="weight", y="height", hue="weight",
#               truncate=True, size=5, data=df)

# sb.distplot(y_test-predictions, axlabel="{x, y}")
# py.xlabel("Colors")
# py.ylabel("Values")

py.show()

myvals = np.array([94,182]).reshape(1,-1)
print(model.predict(myvals))

# Prints the Data
# print(df)
