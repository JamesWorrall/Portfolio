import numpy as np
import pandas as pd
from keras.models import Sequential
from keras import layers, regularizers
import math
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.model_selection import learning_curve, train_test_split
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#Create DataFrames
train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

#Visualize DataFrames
train_df.head()
test_df.head()

#Prepare Vectorizer
vectorizer = CountVectorizer()
vectorizer.fit(train_df['text'])

#Create Workable Variables
x_train = vectorizer.transform(train_df['text'])
y_train = train_df['target'].values
x_test = vectorizer.transform(test_df['text'])

#Split Data for Training and CV
xs_train, xs_val, ys_train, ys_val = (train_test_split(x_train, y_train, test_size = 0.2))

#Create Dimensions for Input
input_dim = xs_train.shape[1]

#Design Nearal Network
model = Sequential()
model.add(layers.Dense(10, input_dim = input_dim, activation = 'relu'))
model.add(layers.Dense(30, activation = 'relu', kernel_regularizer = regularizers.l1_l2(l1 = 0.01, l2 = 0.01)))
model.add(layers.Dense(30, activation = 'relu', kernel_regularizer = regularizers.l1_l2(l1 = 0.01, l2 = 0.01)))
model.add(layers.Dense(30, activation = 'relu', kernel_regularizer = regularizers.l1_l2(l1 = 0.01, l2 = 0.01)))
model.add(layers.Dense(1, activation = 'sigmoid', kernel_regularizer = regularizers.l1_l2(l1 = 0.01, l2 = 0.01)))

#Specifics of Model
model.compile(loss = 'binary_crossentropy',
             optimizer = 'adam',
             metrics = ['accuracy'])

#Visualize Model and Features by Layer
model.summary()

#Train Model
history = model.fit(xs_train, ys_train, epochs = 15, batch_size = 256, validation_data = (xs_val, ys_val))

#Create Variables for Plotting Learning Curve
loss_train = history.history['loss']
accuracy_train = history.history['accuracy']
loss_val = history.history['val_loss']
accuracy_val = history.history['val_accuracy']
epochs = range(1, 16)

#Plot Learning Curve
plt.plot(epochs, loss_train, 'g', label = 'Training Loss')
plt.plot(epochs, loss_val, 'b', label = 'validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Plot Accuracies for Training and CV by Epoch
plt.plot(epochs, accuracy_train, 'r', label = 'Training Accuracy')
plt.plot(epochs, accuracy_val, 'y', label = 'validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Submit to Competition
sample_submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
sample_submission['target'] = model.predict(x_test)
sample_submission['target'] = (sample_submission['target'] > 0.5)
sample_submission['target'] = np.multiply((sample_submission['target']), 1 )
sample_submission.head()
sample_submission.to_csv('submission.csv', index=False)
