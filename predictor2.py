# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 22:15:24 2018

@author: Rrubaa Panchendrarajan
"""


import time
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

np.random.seed(2017)

def init():
    seed = 7
    np.random.seed(seed)

    training_data_set = np.genfromtxt('./data/train.csv',delimiter=',', dtype='unicode',  skip_header=1)
    test_data_set = np.genfromtxt('./data/test.csv',delimiter=',', dtype='unicode',  skip_header=1)
    
    training_data_input = training_data_set[0:,4:].astype(np.float64)
    training_data_ouput = training_data_set[:,1]
    test_data_input = test_data_set[0:,4:].astype(np.float64)
    test_data_ouput = test_data_set[:,1]
    
    training_inputs = [np.reshape(x,128) for x in training_data_input]
    training_results = [(vectorized_result(y)) for y in training_data_ouput]
    
    test_inputs = [np.reshape(x,128) for x in test_data_input]
    test_results = [(vectorized_result(y)) for y in test_data_ouput]

    print(np.shape(training_inputs))
    print(np.shape(training_results))
    print(np.shape(test_inputs))
    print(np.shape(test_results))

    # define model
    model = simple_nn()
    # define optimizer
    sgd = SGD(lr=0.01)
    model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
    model.fit(np.array(training_inputs), np.array(training_results), batch_size=64, \
                       nb_epoch=10, verbose=2, validation_split=0.2)
    # Final evaluation of the model
    scores = model.evaluate(np.array(test_inputs), np.array(test_results), verbose=0)
    print("Baseline Error: %.2f%%" % (100-scores[1]*100))

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