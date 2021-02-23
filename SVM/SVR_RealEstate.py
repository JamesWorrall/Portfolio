# -*- coding: utf-8 -*-
"""SVR.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MPi1_Hjj9yKN_6sm795Z64fjtgkiPHAk
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

#Upload File to Google Colab
from google.colab import files
uploaded = files.upload()

#Create DataFrame
houses_df = pd.read_csv('kc_house_data.csv')

#Create Useful Variables
bedrooms = houses_df['bedrooms']
bathrooms = houses_df['bathrooms']
sqft_living = houses_df['sqft_living']
sqft_lot = houses_df['sqft_lot']
floors = houses_df['floors']

X_dict = {'bedrooms': bedrooms,
          'bathrooms': bathrooms,
          'sqft_living': sqft_living,
          'sqft_lot': sqft_lot,
          'floors': floors}

X = pd.concat(X_dict, axis = 1)

y = houses_df['price']

#Scale Features
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = np.array(y)
y = y.reshape(-1, 1)
y = sc_y.fit_transform(y)

#Create Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Create and Fit Model
regressor = SVR()
regressor.fit(X_train, y_train)

#Score Model
regressor.score(X_test, y_test)