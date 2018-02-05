# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 22:15:24 2018

@author: Rrubaa Panchendrarajan
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

np.random.seed(2017)

def init():
    seed = 7
    np.random.seed(seed)

    train_data = pd.read_csv("./data/train.csv")
    test_data = pd.read_csv("./data/test.csv")
    
    Y = pd.get_dummies(train_data, columns=['Prediction']).filter(regex="Prediction")
    
    _test_size = .1
    X_train, X_test, y_train, y_test = train_test_split(train_data, Y, test_size = _test_size)
    X_train = X_train.drop(['Prediction','Id',"NextId","Position"],axis=1)
    X_test = X_test.drop(['Prediction','Id',"NextId","Position"],axis=1)
    X_out = test_data.drop(['Prediction','Id',"NextId","Position"],axis=1)
    
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    
    # define model
    model = simple_nn()
    # define optimizer
    sgd = SGD(lr=0.2)
    model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
    model.fit(np.array(X_train), np.array(y_train), batch_size=250, \
                       nb_epoch=3000, verbose=2, validation_split=0.2)
    # Final evaluation of the model
    scores = model.evaluate(np.array(X_test), np.array(y_test), verbose=0)
    print("Baseline Error: %.2f%%" % (100-scores[1]*100))
    
    predictions = model.predict(X_out)
    classes = [np.argmax(pred) for pred in predictions]
    classes = [chr(97 + classes[x]) for x in classes]
    pd.DataFrame(classes, columns=['Prediction']).to_csv('Prediction_MLNN.csv')

def vectorized_result(j):
    e = np.zeros((26))
    e[ord(j) - 96 -1] = 1.0
    return e

# define baseline model
def simple_nn():
    model = Sequential()
    model.add(Dense(100, input_dim = 128))
    model.add(Activation('sigmoid'))
    model.add(Dense(26))
    model.add(Activation('softmax'))
    return model