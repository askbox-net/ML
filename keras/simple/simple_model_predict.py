# -*- coding:utf-8 -*-

from keras.models import load_model
from keras.models import model_from_json
import numpy as np

if __name__ == '__main__':
    x = np.array([[0.0, 0.0],
                            [1.0, 0.0],
                            [0.0, 1.0],
                            [1.0, 1.0]])
    with open('./or.json', 'r') as fp:
        model = model_from_json(fp.read())
        model.load_weights('./or.h5')
        y = model.predict(x)
        threshold = 0.5
        print("predict y:", (y > threshold).astype(np.int))

    with open('./and.json', 'r') as fp:
        model = model_from_json(fp.read())
        model.load_weights('./and.h5')
        y = model.predict(x)
        threshold = 0.5
        print("predict y:", (y > threshold).astype(np.int))

    with open('./xor.json', 'r') as fp:
        model = model_from_json(fp.read())
        model.load_weights('./xor.h5')

        y = model.predict(x)
        threshold = 0.5
        print("predict y:", (y > threshold).astype(np.int))
