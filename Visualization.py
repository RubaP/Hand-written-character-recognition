# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 14:38:26 2018

@author: user
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def display_sample(dataset, train_np, rows=4, columns=8):
    indx = 1
    print(dataset[:20].shape)
    for image in dataset[:rows*columns]:
        img = np.reshape(image, [16, 8])
        plt.subplot(rows, columns, indx)
        plt.axis('off')
        plt.title(train_np[indx-1])
        plt.tight_layout()
        plt.imshow(img)
        plt
        indx += 1
        
train = pd.read_csv("./data/test.csv")
print(train.shape)
print("Each characters are 16 by 8 images.")

train_np = train.as_matrix(columns=['Prediction'])
print(train_np.shape)
dataset = train.drop(['Prediction','Id',"NextId","Position"], axis=1).as_matrix()
display_sample(dataset, train_np)