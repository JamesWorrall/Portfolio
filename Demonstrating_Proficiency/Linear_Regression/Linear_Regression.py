import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#Import Data
re_df = pd.read_csv('re_data.csv') #Real Estate DataFrame

#Separate Useful Values for Predicting Price
#Likely Positive Influences on Home Price (RM)
rooms = re_df['RM']
rooms.fillna(rooms.mean(), inplace = True)
pos = rooms
#Likely Negative Influences on Home Price (CRIM, DIS, PTRATIO, LSTAT)
crime = re_df['CRIM']
crime.fillna(crime.mean(), inplace = True)
students_per_teacher = re_df['PTRATIO']
students_per_teacher.fillna(students_per_teacher.mean(), inplace = True)
neg = students_per_teacher + crime
#Combine to Predicting Variable
predictor = pos / neg
#Home Price for Predictable Values
price = re_df['MEDV']

#Create Test Set
predictor_train, predictor_test, price_train, price_test = train_test_split(predictor, price, test_size = 0.20)

#Reshape Variables for Linear Regression
predictor_train = predictor.values.reshape(-1, 1)
predictor_test = predictor_test.values.reshape(-1, 1)
price_train = price.values.reshape(-1, 1)
price_test = price_test.values.reshape(-1, 1)

#Train and Score Model
model = LinearRegression().fit(predictor_train, price_train)
print(model.score(predictor_train, price_train))
print(model.score(predictor_test, price_test))

#Visualize Data and Model
plt.scatter(predictor, price, color = 'red')
plt.plot(predictor_train, model.predict(predictor_train), color = 'blue')
plt.xlabel('Number of Rooms / (Student-Teacher Ratio + Crime Rate)')
plt.ylabel('House Price in $1000s')
plt.title('Housing Price Predictor')
plt.show()
