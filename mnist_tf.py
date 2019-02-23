import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

# Load MNIST data and cache locally using Keras
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# DEBUG: Show an example of digit
# image_index = 7777
# print(y_train[image_index])
# plt.imshow(x_train[image_index], cmap='Greys')
# plt.show()

# DEBUG: Print the dimensions of the data
# print(x_train.shape)

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_train.shape[1], x_train.shape[2], 1)
input_shape = (x_train.shape[1], x_train.shape[2], 1)

# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255

# DEBUG: Print metadata
# print('x_train shape:', x_train.shape)
# print('x_test shape:', x_test.shape)
# print('Number of images in x_train', x_train.shape[0])
# print('Number of images in x_test', x_test.shape[0])

# Creating a Sequential Keras Model and adding the layers
model = Sequential()
model.add(Conv2D(filters=28, kernel_size=(3, 3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(units=128, activation=tf.nn.relu))
model.add(Dropout(rate=0.2))
model.add(Dense(units=10, activation=tf.nn.softmax))  # units = 10 for having 10 digits

# Compile and fit model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x=x_train, y=y_train, epochs=10)

# Evaluate the model
scores = model.evaluate(x=x_test, y=y_test)
print(scores)
