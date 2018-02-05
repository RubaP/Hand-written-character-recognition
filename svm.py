# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 12:06:23 2018

@author: Rrubaa Panchendrarajan
SVM
"""

import pandas as pd
import numpy as np

from sklearn import svm

def load_data():    
    train = pd.read_csv("./data/train.csv")
    test = pd.read_csv("./data/test.csv")
    
    training_data_input = train.drop(['Prediction','Id',"NextId","Position"], axis=1).as_matrix()
    training_data_ouput = train.as_matrix(columns=['Prediction'])
    test_data_input = test.drop(['Prediction','Id',"NextId","Position"], axis=1).as_matrix()
    
    training_inputs = [np.reshape(image,128) for image in training_data_input]
    training_results = np.squeeze(training_data_ouput, axis=1)
    
    test_inputs = [np.reshape(image,128) for image in test_data_input]
    
    clf = svm.SVC()
    clf.fit(training_inputs, training_results);
    predictions = [a for a in clf.predict(test_inputs)]
    
    dfO = pd.DataFrame(predictions)
    dfO.to_csv('out.csv', index=False, header=False)
    
    return ""



