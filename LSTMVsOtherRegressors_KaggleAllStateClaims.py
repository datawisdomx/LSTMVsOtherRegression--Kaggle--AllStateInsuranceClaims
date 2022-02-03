#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 11:56:29 2022

@author: nitinsinghal
"""
# LSTM ANN vs RF XGB for regression - All state claims data Kaggle competition
# LSTM gave worst results. Not good for time series/regression problems
# Data can be downloaded from here https://www.kaggle.com/c/allstate-claims-severity/data
# Test your results against global ranking by submitting in Kaggle (5 year old competition)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras import layers, models, losses, metrics, optimizers, regularizers
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import datetime
import warnings
warnings.filterwarnings('ignore')


# Load the data
train_data = pd.read_csv('/allstate-claims-severity/train.csv')
test_data = pd.read_csv('/allstate-claims-severity/test.csv')

# Perform EDA - see the data types, content, statistical properties
print(train_data.describe())
print(train_data.info())
print(train_data.head(5))
print(train_data.dtypes)
      
print(test_data.describe())
print(test_data.info())
print(test_data.head(5))
print(test_data.dtypes)

# Perform data wrangling - remove duplicate values and set null values to 0
train_data.drop_duplicates(inplace=True)
test_data.drop_duplicates(inplace=True)
train_data.drop(['id'], axis=1, inplace=True)

# Only used fillna=0. Dropna not used as other columns/row shave useful data
train_data.fillna(0, inplace=True)
test_data.fillna(0, inplace=True)

# Split categorical data for one hot encoding
train_data_cat = train_data.select_dtypes(exclude=['int64','float64'])
train_data_num = train_data.select_dtypes(include=['int64','float64'])

test_data_cat = test_data.select_dtypes(exclude=['int64','float64'])
test_data_num = test_data.select_dtypes(include=['int64','float64'])

# Encode the categorical features using OneHotEncoder. Use the same encoder for train and test set
ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
ohe.fit(train_data_cat)
train_data_cat = pd.DataFrame(ohe.transform(train_data_cat))
test_data_cat = pd.DataFrame(ohe.transform(test_data_cat))

# Merge encoded categorical data with mueric data
train_data_ohe = train_data_num.join(train_data_cat)
test_data_ohe = test_data_num.join(test_data_cat)

# Setup the traing and test X, y datasets
y_train = train_data_ohe.loc[:,'loss'].values
train_data_ohe.drop(['loss'], axis=1, inplace=True)
X_train = train_data_ohe.iloc[:,:-1].values
X_test = test_data_ohe.iloc[:,1:-1].values

# Scale all the data as some features have larger range compared to the rest
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#### LSTM ###
input_dim = X_train.shape[1]
batch_size = 32
units = 50
output_size = 1

lstmmodel = models.Sequential()

lstmmodel.add(layers.LSTM(units, input_shape=(input_dim,1), return_sequences=True))
lstmmodel.add(layers.BatchNormalization())
#lstmmodel.add(layers.LSTM(units, return_sequences=True))
#lstmmodel.add(layers.BatchNormalization())
lstmmodel.add(layers.LSTM(units))
lstmmodel.add(layers.BatchNormalization())
#lstmmodel.add(layers.Dropout(rate=0.2))

lstmmodel.add(layers.Dense(20, activation='relu'))
lstmmodel.add(layers.BatchNormalization())
#lstmmodel.add(layers.Dropout(rate=0.2))
lstmmodel.add(layers.Dense(output_size, activation='relu'))

lstmmodel.summary()

lstmmodel.compile(loss=losses.MeanSquaredError(),
              optimizer=optimizers.Adam(),
              metrics=[metrics.RootMeanSquaredError()])

history = lstmmodel.fit(X_train, y_train, batch_size=batch_size, epochs=2)

print(history.params)
print(history.history.keys())

# Plot the train/test accuracy to see marginal improvement
plt.plot(history.history['root_mean_squared_error'], label='root_mean_squared_error')
plt.plot(history.history['val_root_mean_squared_error'], label = 'val_root_mean_squared_error')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.title('LSTM Train Vs Test')
plt.show()

# Evaluate the lstmmodel using the test set
# test_loss, test_acc = lstmmodel.evaluate(X_test, y_test, verbose=1)
# print('Evaluate Acc: ', test_acc)

y_pred = lstmmodel.predict(X_test, verbose=1)
print('LSTM Pred Prob: ', y_pred)

####### ANN ##########

neurons = 200
annmodel = models.Sequential()
annmodel.add(layers.Dense(neurons, activation='relu', kernel_regularizer='l1',
    bias_regularizer='l2', activity_regularizer='l2'))
annmodel.add(layers.Dense(neurons, activation='relu', kernel_regularizer='l1',
    bias_regularizer='l2', activity_regularizer='l2'))
annmodel.add(layers.Dense(neurons, activation='relu', kernel_regularizer='l1',
    bias_regularizer='l2', activity_regularizer='l2'))
annmodel.add(layers.Dense(1, activation='relu', kernel_regularizer='l1',
    bias_regularizer='l2', activity_regularizer='l2'))

# STEPS_PER_EPOCH = 20
# lr_schedule = optimizers.schedules.InverseTimeDecay(
#   0.001,
#   decay_steps=STEPS_PER_EPOCH*1000,
#   decay_rate=1,
#   staircase=False)


annmodel.compile(loss=losses.MeanSquaredError(),
              optimizer=optimizers.Adam(),
              metrics=[metrics.RootMeanSquaredError()])

history = annmodel.fit(X_train, y_train, batch_size=60, epochs=10)

print(history.params)
print(history.history.keys())

# Plot the train/test accuracy to see marginal improvement
plt.plot(history.history['root_mean_squared_error'], label='root_mean_squared_error')
# plt.plot(history.history['val_root_mean_squared_error'], label = 'val_root_mean_squared_error')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('ANN Train Vs Test')
plt.show()

y_pred = annmodel.predict(X_test, verbose=1)

# Accuracy score
# mse = mean_squared_error(y_test, y_pred)
# print('ANN MSE: ', mse)
# print('ANN RMSE: ', np.sqrt(mse))

# Output predicted y values using actula test data into csv file, submit in kaggle competition and check score
df_result = pd.DataFrame()
df_result.index = test_data['id']
df_result['loss'] = y_pred
df_result.to_csv('/AllStateANN.csv')

###### Using Random Forest Regressor #######
print('Start time: ', datetime.datetime.now())
regressor = RandomForestRegressor()
regressor.fit(X_train,y_train)

# Make predictions using the input X test features
y_pred = regressor.predict(X_test)

# Output predicted y values using actula test data into csv file, submit in kaggle competition and check score
df_result['loss'] = y_pred
df_result.to_csv('/AllStateRF.csv')

print('End time: ', datetime.datetime.now())

###### Using XGBRegressor #######
print('Start time: ', datetime.datetime.now())
regressor = XGBRegressor()
regressor.fit(X_train,y_train)

# Make predictions using the input X test features
y_pred = regressor.predict(X_test)

# Output predicted y values using actula test data into csv file, submit in kaggle competition and check score
df_result = pd.DataFrame()
df_result.index = test_data['id']
df_result['loss'] = y_pred
df_result.to_csv('/AllStateXGB.csv')

print('End time: ', datetime.datetime.now())





