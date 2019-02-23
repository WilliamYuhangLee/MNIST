import tensorflow as tf
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D


def build(train_data="out/preprocessed.npz", save_file="out/model.h5"):

    # Retrieve preprocessed data
    train_data = np.load(file=train_data)
    x_train, y_train = train_data["x_train"], train_data["y_train"]
    input_shape = train_data["input_shape"]

    # Creating a Sequential Keras Model and adding the layers
    model = Sequential()
    model.add(Conv2D(filters=28, kernel_size=(3, 3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
    model.add(Dense(units=128, activation=tf.nn.relu))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=10, activation=tf.nn.softmax))  # units = 10 for having 10 digits

    # Compile and fit model
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x=x_train, y=y_train, epochs=10)

    # Save the model
    model.save(filepath=save_file)
