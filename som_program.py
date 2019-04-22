# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 21:54:45 2019

@author: tdpco
"""

# importing the libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show

# importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

# Training the SOM
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(X, num_iteration=100)

# Visualizing the result
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['red', 'blue']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0]+0.5,
         w[1]+0.5,
         markers[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)
show()

# Finding the Fraud
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8, 3)], mappings[(6, 4)]), axis=0)
frauds = sc.inverse_transform(frauds)
