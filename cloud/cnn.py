# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 00:35:20 2018

@author: Rrubaa Panchendrarajan
Convolutional Neural Network
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
K.set_image_dim_ordering('th')
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

def init():
    seed = 7
    np.random.seed(seed)
    
    #read test and training data sets
    train_data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")
    
    Y = pd.get_dummies(train_data, columns=['Prediction']).filter(regex="Prediction")
    
    #preprocess the data sets    
    _test_size = .1
    X_train, X_test, y_train, y_test = train_test_split(train_data, Y, test_size = _test_size)
    X_train = X_train.drop(['Prediction','Id',"NextId","Position"],axis=1).as_matrix()
    X_test = X_test.drop(['Prediction','Id',"NextId","Position"],axis=1).as_matrix()
    X_out = test_data.drop(['Prediction','Id',"NextId","Position"],axis=1).as_matrix()
    
    training_inputs = np.array([np.reshape(x, (1, 16, 8)) for x in X_train])
    training_results = np.array(y_train.as_matrix())
    
    test_inputs = np.array([np.reshape(x, (1, 16, 8)) for x in X_test])
    test_results = np.array(y_test.as_matrix())
    
    out_input = [np.reshape(x, (1, 16, 8)) for x in X_out]
    
    datagen = ImageDataGenerator(width_shift_range=.2, 
                             height_shift_range=.2,
                             fill_mode='nearest')
    
    datagen.fit(training_inputs)
    
    #train the model
    model = baseline_model()
    #model.fit(np.array(training_inputs), np.array(training_results), validation_data=(np.array(test_inputs), np.array(test_results)), epochs=28, batch_size=250, verbose=2)

    model.fit_generator(datagen.flow(training_inputs, training_results, batch_size=250), epochs=50, validation_data=(test_inputs, test_results))
    #predict using the train model for test data
    predictions = model.predict(np.array(out_input))
    classes = [(chr(np.argmax(pred) + 97)) for pred in predictions]
    pd.DataFrame(classes, columns=['Prediction']).to_csv('prediction_cnn.csv')
    

def baseline_model():
    #convolutional neural network model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(1, 16, 8), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(26, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

init()