#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 14:18:30 2016

@author: carlosurdaneta
"""
from __future__ import print_function
import tensorflow as tf
import numpy
import pandas
import math
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# convert an array of values into a dataset matrix
def create_dataset(datasetx1, datasetx2, datasety, look_back=1):
    dataX, dataY= [], []
    for i in range(len(datasetx1)-look_back-1):
        a = numpy.append(datasetx1[i:(i+look_back),0],datasetx2[i:(i+look_back),0])        
        dataX.append(a)
        dataY.append(datasety[i+look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

# fix random seed for reproducibility
numpy.random.seed(7)

# load the datasets
trainfn = r'/Users/carlosurdaneta/Desktop/Data/TrainLead.csv'
traindataframex1 = pandas.read_csv(trainfn, sep=',', usecols=[0], skipinitialspace=True)
traindatasetx1 = traindataframex1.values
traindatasetx1 = traindatasetx1.astype('float32')
traindataframex2 = pandas.read_csv(trainfn, sep=',', usecols=[1], skipinitialspace=True)
traindatasetx2 = traindataframex2.values
traindatasetx2 = traindatasetx2.astype('float32')
traindataframey = pandas.read_csv(trainfn, sep=',', usecols=[2], skipinitialspace=True)
traindatasety = traindataframey.values
traindatasety = traindatasety.astype('float32')

testfn = r'/Users/carlosurdaneta/Desktop/Data/TestLead.csv'
testdataframex1 = pandas.read_csv(testfn, sep=',', usecols=[0], skipinitialspace=True)
testdatasetx1 = testdataframex1.values
testdatasetx1 = testdatasetx1.astype('float32')
testdataframex2 = pandas.read_csv(testfn, sep=',', usecols=[1], skipinitialspace=True)
testdatasetx2 = testdataframex2.values
testdatasetx2 = testdatasetx2.astype('float32')
testdataframey = pandas.read_csv(testfn, sep=',', usecols=[2], skipinitialspace=True)
testdatasety = testdataframey.values
testdatasety = testdatasety.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
traindatasetx1 = scaler.fit_transform(traindatasetx1)
traindatasetx2 = scaler.fit_transform(traindatasetx2)
traindatasety = scaler.fit_transform(traindatasety)
testdatasetx1 = scaler.fit_transform(testdatasetx1)
testdatasetx2 = scaler.fit_transform(testdatasetx2)
testdatasety = scaler.fit_transform(testdatasety)

# split into train and test sets
train_size = int(len(traindatasety))
test_size = int(len(testdatasety))
trainx1, testx1 = traindatasetx1, testdatasetx1
trainx2, testx2 = traindatasetx2, testdatasetx2
trainy, testy = traindatasety, testdatasety

# reshape into X=t and Y=t+1
look_back = 10
trainX, trainY = create_dataset(trainx1, trainx2, trainy, look_back)
testX, testY = create_dataset(testx1, testx2, testy, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
batch_size=1
model = Sequential()
#model.add(LSTM(1, batch_input_shape=(batch_size, 1, look_back*2), stateful=True, return_sequences=True))
#model.add(LSTM(4, batch_input_shape=(batch_size, 1, look_back*2), stateful=True, return_sequences=True))
model.add(LSTM(4, batch_input_shape=(batch_size, 1, look_back*2), stateful=True))
model.add(Dense(1))
adam=Adam(lr=0.2)
model.compile(loss='mean_squared_error', optimizer=adam)
#Change number of epochs below in range
for i in range(10):
    print('Epoch %f' % (i+1))
    model.fit(trainX, trainY, nb_epoch=1, batch_size=batch_size, verbose=2, shuffle=False)
    model.reset_states()

# make predictions
trainPredict = model.predict(trainX, batch_size=batch_size)
model.reset_states()
testPredict = model.predict(testX, batch_size=batch_size)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

#changed datase with dataset for plotting purpose
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(traindatasety)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :]= trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(testdatasety)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[look_back:len(testPredict)+look_back, :]= testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(traindatasety))
plt.plot(trainPredictPlot)
plt.show()
plt.plot(scaler.inverse_transform(testdatasety))
plt.plot(testPredictPlot)
plt.show()
