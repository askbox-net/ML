# -*- coding:utf-8 -*-

import keras
import os
from keras.optimizers import RMSprop
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf


class mnist_model(object):
    def __init__(self, model_filename='mnist_model', model_path='./model'):
        self.model_filename = model_filename
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.model = Sequential()
        self.model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
        print(self.model.summary())
        with open('%s/%s.json' % (self.model_path, self.model_filename), 'w') as fp:
            fp.write(self.model.to_json())

        h5_model = '%s/%s.h5' % (self.model_path, self.model_filename)
        if os.path.exists(h5_model):
            self.model.load_weights(h5_model)

        self.checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=h5_model,
                save_weights_only=True,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=0)

        self.early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                #monitor='loss,accuracy',
                #monitor='accuracy',
                #min_delta=0,
                patience=3
                )

    def fit(self, x_train, y_train, x_valid, y_valid):
        history = self.model.fit(x_train, y_train,
                    #batch_size=128,
                    batch_size=128,
                    epochs=2,
                    verbose=1,
                    callbacks=[self.checkpoint, self.early_stopping],
                    validation_data=(x_valid, y_valid))
        self.model.save_weights(self.model_filename)

    def evaluate(self, x_test, y_test):
        score = self.model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train)
    print(x_train.shape)
    x_train1, x_valid, y_train1, y_valid = train_test_split(x_train, y_train, test_size=0.2)

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_valid = x_valid.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    x_train = x_train.astype('float32') / 255
    x_valid = x_valid.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    y_train = keras.utils.to_categorical(y_train, 10)
    y_valid = keras.utils.to_categorical(y_valid, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model = mnist_model()
    model.fit(x_train, y_train, x_valid, y_valid)
    model.evaluate(x_test, y_test)
