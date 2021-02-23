# -*- coding:utf-8 -*-

from keras.models import load_model
from keras.models import model_from_json
import numpy as np

if __name__ == '__main__':
    x = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
        ])
    print('X:', x)
    with open('./or.json', 'r') as fp:
        model = model_from_json(fp.read())
        model.load_weights('./or.h5')
        predict_y = model.predict(x)
        print('OR:', list(map(lambda x: x.argmax(), predict_y)))

    with open('./and.json', 'r') as fp:
        model = model_from_json(fp.read())
        model.load_weights('./and.h5')
        predict_y = model.predict(x)
        print('AND:', list(map(lambda x: x.argmax(), predict_y)))

    with open('./xor.json', 'r') as fp:
        model = model_from_json(fp.read())
        model.load_weights('./xor.h5')
        predict_y = model.predict(x)
        print('XOR:', list(map(lambda x: x.argmax(), predict_y)))
