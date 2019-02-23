import tensorflow as tf
import numpy as np


def preprocess(train_data="out/preprocessed_train.npz", test_data="out/preprocessed_test.npz"):
    # Load MNIST data and cache locally using Keras
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

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

    # Save preprocessed data locally
    np.savez(file=train_data, x_train=x_train, y_train=y_train, input_shape=input_shape)
    np.savez(file=test_data, x_test=x_test, y_test=y_test, input_shape=input_shape)
