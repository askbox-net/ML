# -*- coding:utf-8 -*-

import keras
from keras.models import load_model
from keras.datasets import mnist
from keras.models import model_from_json
import numpy as np

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    y_train = keras.utils.to_categorical(y_train, 10)
    #y_test = keras.utils.to_categorical(y_test, 10)

    with open('./model/mnist_model.json', 'r') as fp:
        model = model_from_json(fp.read())
        model.load_weights('./model/mnist_model.h5')
        predict_y = model.predict(x_test)
        predict_y = list(map(lambda x: x.argmax(), predict_y))
        for k, v in zip(y_test, predict_y):
            print(k, v, k==v)


