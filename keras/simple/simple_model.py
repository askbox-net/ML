# -*- coding:utf-8 -*-

import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense
from keras.optimizers import RMSprop
import tensorflow as tf


class simple_model(object):
    def __init__(self, input_dim=2, units=3, model_filename='simple_model'):
        self.model_filename = model_filename
        self.model = Sequential()
        self.model.add(
                Dense(
                    activation = 'sigmoid',
                    input_dim = input_dim,
                    units = units
                    )
                )
        self.model.add(
                Dense(
                    units = 2,
                    activation = 'sigmoid'
                    )
                )
        self.model.compile(
                loss = 'mean_squared_error',
                optimizer = RMSprop(),
                metrics = ['accuracy']
                )
        with open('%s.json' % model_filename, 'w') as fp:
            fp.write(self.model.to_json())
        """
        self.checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath='%s.h5' % model_filename,
            save_weights_only=True,
            #monitor='val_accuracy',
            #mode='max',
            save_best_only=True)
        """
        self.checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath='%s.h5' % model_filename,
                save_weights_only=True,
                verbose=0)

        self.early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=0.0,
                patience=2
                )

    def fit(self, X, y):
        self.history = self.model.fit(
                X,
                y,
                batch_size = 4,
                epochs = 5000,
                #validation_data=(X, y),
                callbacks = [self.checkpoint, self.early_stopping]
                )
        self.model.save_weights('simple_model.h5')

    def evaluate(self, X, y):
        test_loss, test_acc = self.model.evaluate(X, y, verbose=0)
        print('test_loss:', test_loss)
        print('test_acc:', test_acc)
        predict_y = self.model.predict(X)
        threshold = 0.5
        print("thresholded predict_y:", (predict_y > threshold).astype(np.int))


if __name__ == '__main__':
    # OR
    x_or = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]])
    y_or = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0]])
    param = {
        'model_filename': 'or'
    }
    model = simple_model(**param)
    model.fit(x_or, y_or)
    model.evaluate(x_or, y_or)

    # AND
    x_and = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]])
    y_and = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0]])
    param = {
        'model_filename': 'and'
    }
    model = simple_model(**param)
    model.fit(x_and, y_and)
    model.evaluate(x_and, y_and)

    # XOR
    x_xor = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]])
    y_xor = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 0.0]])
    param = {
        'model_filename': 'xor'
    }
    model = simple_model(**param)
    model.fit(x_xor, y_xor)
    model.evaluate(x_xor, y_xor)
