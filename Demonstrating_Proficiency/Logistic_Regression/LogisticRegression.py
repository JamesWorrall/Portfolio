import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#Import Data
iris_df = pd.read_csv('Iris.csv')
iris_df = shuffle(iris_df)

#Create Useful Variables
X = iris_df.iloc[:, 1:-1]
y = iris_df.iloc[:, -1]
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2)

lReg = LogisticRegression()

#Fit to Model
lReg = lReg.fit(Xtrain, ytrain) #0.973
print(lReg.score(Xtest, ytest)) #0.900

#Variables for Plotting
sepal_length = iris_df['SepalLengthCm']
sepal_width = iris_df['SepalWidthCm']
petal_length = iris_df['PetalLengthCm']
petal_width = iris_df['PetalWidthCm']


#Plot Influences of Each Variable
plt.scatter(sepal_length, y, color = 'blue', marker = '*', label = 'Sepal Length (CM)')
plt.scatter(sepal_width, y, color = 'green', marker = 'o', label = 'Sepal Width (CM)')
plt.scatter(petal_length, y, color = 'red', marker = '^', label = 'Petal Length (CM)')
plt.scatter(petal_width, y, color = 'deeppink', marker = 's', label = 'Petal Width (CM)')
plt.xlabel('Dimension Size in Centimeters')
plt.ylabel('Types of Iris Flower')
plt.title('Dimensional Classification')
plt.legend()
plt.show()
