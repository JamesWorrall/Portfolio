import pandas as pd
import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Conv2D, Dense, BatchNormalization
from keras.optimizers import RMSprop
from keras.metrics import categorical_accuracy, sparse_categorical_accuracy
from keras.activations import softmax, relu
from keras.losses import CategoricalCrossentropy
from keras.utils import to_categorical
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Create Dataframes
train_df = pd.read_csv('Face_Train.csv')
test_df = pd.read_csv('Face_Test.csv')

#Create Usable Variables
xtrain = train_df['pixels'].values.tolist()
ytrain = train_df['emotion'].values.tolist()

width, height = 48, 48

#Create Lists from Pixel Strings and Reshapes them to 48 x 48 Images
for i in range(len(xtrain)):
    xtrain[i] = xtrain[i].split(' ')
    xtrain[i] = [int(x) for x in xtrain[i]]
    xtrain[i] = np.reshape(xtrain[i], (width, height, 1))

#Convert to Array
xtrain = np.array(xtrain)
ytrain = np.array(ytrain)

#Split for CV and Test
x_train, xval, y_train, yval = train_test_split(xtrain, ytrain, test_size = 0.40)
x_val, x_test, y_val, y_test = train_test_split(xval, yval, test_size = 0.50)

#Change to Vectors for Compatibility
y_train = to_categorical(y_train, 7)
y_val = to_categorical(y_val, 7)
y_test = to_categorical(y_test, 7)

#Create Model
model = Sequential()
model.add(Conv2D(8, (5,5), padding = 'valid', input_shape = (48, 48, 1)))
model.add(BatchNormalization(epsilon = 0.01, momentum = 0.99))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (5,5), strides = 1, padding = 'valid'))
model.add(Dropout(0.2))

model.add(Conv2D(16, (5,5), padding = 'valid'))
model.add(BatchNormalization(epsilon = 0.01, momentum = 0.99))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (5,5), strides = 2, padding = 'valid'))
model.add(Dropout(0.3))

#New Addition
model.add(Conv2D(32, (5,5), padding = 'valid'))
model.add(Conv2D(32, (5,5), padding = 'valid'))
model.add(Conv2D(16, (5,5), padding = 'valid'))

model.add(Flatten())
#New Addition
model.add(Dense(1000, bias_regularizer = regularizers.l2(0.01), activation = 'relu', kernel_regularizer = regularizers.l2( l2 = 0.01)))
model.add(Dropout(0.4))
model.add(Dense(1000, bias_regularizer=regularizers.l2(0.01),  activation = 'relu', kernel_regularizer = regularizers.l2( l2 = 0.01)))
model.add(Dense(7, activation = 'softmax'))

model.summary()

batch_size = 100
epochs = 100

#Train Model

optimizer = keras.optimizers.RMSprop(learning_rate = 0.0003, epsilon = 1e-6)

model.compile(loss = 'CategoricalCrossentropy', optimizer = optimizer, metrics = ['categorical_accuracy'])

history = model.fit(x_train, y_train, batch_size = batch_size, steps_per_epoch = int(len(x_train)/batch_size), epochs = epochs, validation_data = (x_val, y_val))

history

#Create Variables for Learning Curve
loss_train = history.history['loss']
accuracy_train = history.history['categorical_accuracy']
loss_val = history.history['val_loss']
accuracy_val = history.history['val_categorical_accuracy']
epoch_label = range(1, (epochs + 1))

#Plot Learning Curve
plt.plot(epoch_label, loss_train, 'g', label = 'Training Loss')
plt.plot(epoch_label, loss_val, 'r', label = 'Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Plot Accuracies for Training and CV by Epoch
plt.plot(epoch_label, accuracy_train, 'b', label = 'Training Accuracy')
plt.plot(epoch_label, accuracy_val, 'y', label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Run Model on Test Data
score = model.evaluate(x_test, y_test)
print('Score: ')
print(score)
