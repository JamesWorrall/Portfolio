import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#Import Data
iris_df = pd.read_csv('Iris.csv')
iris_df = shuffle(iris_df)
print(iris_df)

#Create Useful Variables
