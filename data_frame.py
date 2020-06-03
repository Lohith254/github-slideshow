#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 12:27:16 2020

@author: lohith
"""


values = [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,6,7,29,30,31,34,40,47,62,62,74,82,
          100,114,129,143,169,194,249,332,396,499,536,657,727,887,987,1024,
          1251,1397,1998,2543,3059,3588,4289,4778,5351,5916,6725,7600,8446,
          9205,10453,11487,12370,13430,14352,16365,17615,18539,20080,21370,
          23039,24447,26283,27890,29451,31324,33062,34863,37257,39699,42505,
          46437,49400,52987,56351,59695,62808,67161,70768,74292,78055,81997,
          85784,90648,95698,100328,106475,112028,118226,124794,131423,138536,
          144950]

date = ["Feb 15","Feb 16","Feb 17","Feb 18","Feb 19","Feb 20","Feb 21",
        "Feb 22","Feb 23","Feb 24","Feb 25","Feb 26","Feb 27","Feb 28",
        "Feb 29","Mar 01","Mar 02","Mar 03","Mar 04","Mar 05","Mar 06",
        "Mar 07","Mar 08","Mar 09","Mar 10","Mar 11","Mar 12","Mar 13",
        "Mar 14","Mar 15","Mar 16","Mar 17","Mar 18","Mar 19","Mar 20",
        "Mar 21","Mar 22","Mar 23","Mar 24","Mar 25","Mar 26","Mar 27",
        "Mar 28","Mar 29","Mar 30","Mar 31","Apr 01","Apr 02","Apr 03",
        "Apr 04","Apr 05","Apr 06","Apr 07","Apr 08","Apr 09","Apr 10",
        "Apr 11","Apr 12","Apr 13","Apr 14","Apr 15","Apr 16","Apr 17",
        "Apr 18","Apr 19","Apr 20","Apr 21","Apr 22","Apr 23","Apr 24",
        "Apr 25","Apr 26","Apr 27","Apr 28","Apr 29","Apr 30","May 01",
        "May 02","May 03","May 04","May 05","May 06","May 07","May 08",
        "May 09","May 10","May 11","May 12","May 13","May 14","May 15",
        "May 16","May 17","May 18","May 19","May 20","May 21","May 22",
        "May 23","May 24","May 25"]

deaths = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,2,2,2,3,3,4,
          5,5,7,10,10,12,20,20,24,27,32,35,58,72,86,99,118,136,160,178,227,249,
          288,331,358,393,422,448,486,521,559,592,645,681,721,780,825,881,939,
          1008,1079,1154,1223,1323,1391,1566,1693,1785,1889,1985,2101,2212,
          2294,2415,2551,2649,2753,2871,3025,3156,3302,3434,3584,3726,3868,
          4024,4172] 

recovered = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,6,0,3,0,
             1,0,6,3,0,1,10,6,3,2,28,11,11,7,21,25,43,1,37,99,47,93,38,129,139,
             195,111,101,178,149,260,273,422,391,419,702,395,642,484,443,584,
             614,610,690,631,939,812,956,1072,1295,1189,1445,1111,1414,1668,
             1580,1871,1980,1569,2289,3966,2571,2438,3076,3113,3131,3271,2561,
             3307,3014] 

Active = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,4,26,27,28,31,37,43,58,58,69,70,88,
          99,114,126,152,170,221,304,365,455,486,602,662,794,879,902,1117,1239,
          1792,2280,2781,3260,3843,4267,4723,5232,5863,6577,7189,7794,8914,
          9735,10440,11214,11825,13381,14202,14674,15460,16319,17306,18171,
          19519,20486,21375,22569,23546,24641,26027,27557,29339,32024,33565,
          35871,37686,39823,41406,43980,45925,47457,49104,51379,52773,53553,
          55878,57939,60864,63172,66089,69244,73170,76820,80072]

import numpy as np
import pandas as pd
corona_original_india_data = pd.DataFrame()
corona_original_india_data['date'] = date
corona_original_india_data['No of Total Cases'] = values 
corona_original_india_data['No of Deaths'] = deaths
corona_original_india_data['Recovered Cases'] = recovered
corona_original_india_data['Active cases'] = Active

predicted_values = []


corona_data_india = pd.DataFrame()
corona_data_india['Values'] = values

# Creating Training Data Set
training_data_set = corona_data_india.iloc[:, 0:1].values

# Scaling data with Normalisation
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1), copy = True)
scaled_training_data = sc.fit_transform(training_data_set)

# Converting scaled data to training set and test set (Taking 20 days data to predict 21st day)
X_train = []
y_train = []
for i in range(45, len(values)):
    X_train.append(scaled_training_data[i - 45: i, 0])
    y_train.append(scaled_training_data[i, 0])
    
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshapping the array X_train

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

"""
Neural Network
"""
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

# Initialising RNN variable
rnn = Sequential()

# Adding the first LSTM layer
rnn.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
rnn.add(Dropout(p = 0.2))

# Adding the second LSTM layer
rnn.add(LSTM(units = 50, return_sequences = True))
rnn.add(Dropout(p = 0.2))

# Adding the Third LSTM Layer
rnn.add(LSTM(units = 50, return_sequences = True))
rnn.add(Dropout(p = 0.2))

# Adding the Fourth Layer
rnn.add(LSTM(units = 50))
rnn.add(Dropout(p = 0.2))

# Adding the Output Layer
rnn.add(Dense(units = 1))

# Compiling the sequential Neural Network
rnn.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the trained network
rnn.fit(X_train, y_train, batch_size = 32, epochs = 200) 

"""
Prdicting the values
"""

for x in range(10):
    corona_data_india = pd.DataFrame()
    corona_data_india['Values'] = values
    
    # Creating Training Data Set
    training_data_set = corona_data_india.iloc[:, 0:1].values
    
    # Scaling data with Normalisation
    sc = MinMaxScaler(feature_range = (0, 1), copy = True)
    scaled_training_data = sc.fit_transform(training_data_set)
    
    # Converting scaled data to training set and test set (Taking 20 days data to predict 21st day)
    X_train = []
    y_train = []
    for i in range(45, len(values)):
        X_train.append(scaled_training_data[i - 45: i, 0])
        y_train.append(scaled_training_data[i, 0])
        
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # Reshapping the array X_train
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    X_test = corona_data_india.iloc[-45:, 0:1]
    X_test = sc.transform(X_test)
    
    X = []
    for i in range(0,45):
        X.append(X_test[0 : 45, 0])
    
    X = np.array(X)
    
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    predictions = rnn.predict(X)
    
    predictions = sc.inverse_transform(predictions)
    
    predictions = list(predictions)
    
    predictions = predictions[0]
    
    complete_data_set = values
    
    complete_data_set.append(int(predictions[0]))
    predicted_values.append(int(predictions[0]))
    
    values = complete_data_set


import matplotlib.pyplot as plt
