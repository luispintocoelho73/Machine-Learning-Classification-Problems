# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 10:44:12 2021

@author: Group 6- André Silva 90015 and Luis Coelho 90127
"""

# imports
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import balanced_accuracy_score as BACC
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential, Model, load_model
from keras.callbacks import EarlyStopping, Callback
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, ELU,GlobalAveragePooling2D, LeakyReLU, BatchNormalization
from keras.layers.convolutional import Convolution2D, Cropping2D, Conv1D, Conv2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
from keras.utils import np_utils
import keras_tuner as kt
import time
import time, glob

global inputShape,size

LOG_DIR = f"{int(time.time())}"

# this method was used in order to find a suiting network for the binary classification problem at hand.
# this method uses the keras-tuner package
def build_model(hp):
    model = Sequential()
    model.add(Conv2D(hp.Choice("input_units", [32, 64, 96, 128, 160, 192, 224, 256]), (3,3), input_shape=(50,50,1)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    for i in range(hp.Choice("n_layers", [1, 2])):
        model.add(Conv2D(hp.Choice('conv_units', [32, 64, 96, 128, 160, 192, 224, 256]), (3,3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))
        
    model.add(Flatten())
    model.add(Dense(hp.Choice('dense_units', [32, 64, 96, 128, 160, 192, 224, 256])))
    model.add(Activation("relu"))
    model.add(Dropout(hp.Choice('dropout_units', [0.1, 0.2, 0.3, 0.4, 0.5])))
    model.add(Dense(2))
    model.add(Activation("sigmoid"))
    
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    return model

# the best performing network had these parameters
def kerasModel():
    model = Sequential()
    model.add(Conv2D(64, (3,3), input_shape=(50, 50, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
              
    model.add(Conv2D(128, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
              
    model.add(Flatten())  
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))
    return model



# The data is loaded
X_allfolds = np.load('Xtrain_Classification_Part1.npy')
y_allfolds = np.load('Ytrain_Classification_Part1.npy')
X_test = np.load('Xtest_Classification_Part1.npy')

# The data is reshaped to be inserted in the models
X_allfolds_reshaped = X_allfolds.reshape(len(X_allfolds),50,50,1)
y_allfolds_reshaped = y_allfolds.reshape(len(y_allfolds),1)
X_test_reshaped = X_test.reshape(len(X_test),50,50,1)

# The data is normalized
X_test_reshaped = X_test_reshaped/255
X_allfolds_reshaped = X_allfolds_reshaped/255

# 5 folders were created: 4 of these folders will be used for training and the remaining folder will be used for validation
kf = KFold(n_splits=5, random_state=7, shuffle=True)

# split input data into folds
kf.split(y_allfolds_reshaped)

# this for cicle only runs for one interation
for train_index, validation_index in kf.split(X_allfolds_reshaped):
    X_train, X_validation = X_allfolds_reshaped[train_index], X_allfolds_reshaped[validation_index]
    y_train, y_validation = y_allfolds_reshaped[train_index], y_allfolds_reshaped[validation_index]

    #print(X_train.shape)
    #print(X_validation.shape)  
    
    #print("train shape X", X_train.shape)
    #print("train shape y", y_train.shape)
    
    

    # the y training and validation data was one hot encoded 
    train_labels = tf.keras.utils.to_categorical(y_train, num_classes=2)
    validation_labels = tf.keras.utils.to_categorical(y_validation, num_classes=2)
    
    # an early stopping method was defined in order to avoid overfitting
    early_stopping = [EarlyStopping(monitor = 'val_loss', min_delta =  0.001, patience = 5)]
    
    #### these lines of code were used in order to find the most suiting architectures for our binary classification models, through the kensor-tuner package
    #tuner = kt.RandomSearch(build_model, objective="val_accuracy", max_trials=5, executions_per_trial=2, directory=LOG_DIR)
    
    #tuner.search(x=X_train, y=train_labels, epochs=50, batch_size=64, callbacks=early_stopping, validation_data=(X_validation,validation_labels))
    
    ## information about the 
    #print(tuner.get_best_hyperparameters()[0].values)
    #print(tuner.results_summary())
    #print(tuner.get_best_models()[0].values)
    #####
    
    #print("train shape X", X_train.shape)
    #print("train shape y", y_train.shape)
    
    # the best model found through keras_tuner was the following 
    model = kerasModel()
    
    
    # the model was compiled
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    # the data was fit into the model for the training and validation procedures
    history = model.fit(X_train, train_labels, batch_size= 64, epochs=50, callbacks=early_stopping, validation_data=(X_validation, validation_labels))
    
    # the training accuracy for the model was the following
    metricsTrain = model.evaluate(X_train, train_labels)
    print("Training Accuracy: ",metricsTrain[1]*100,"%")
    
    print("")
    
    # the validation (testing) accuracy for the model was the following
    metricsTest = model.evaluate(X_validation,validation_labels)
    print("Testing Accuracy: ",metricsTest[1]*100,"%")
    
    # print("Saving model weights and configuration file")
    model.save('model.h5')
    print("Saved model to disk")
    
        
    break


aux = 0

# the three best trainings of the kerasModel() model were saved
model1 = load_model('88_254.h5')
model2 = load_model('88_25.h5')
model3 = load_model('88_33.h5')

# predictions of the three models were computed
y_predict1 = model1.predict(X_test_reshaped)
y_predict2 = model2.predict(X_test_reshaped)
y_predict3 = model3.predict(X_test_reshaped)

# the respective class for each data sample was computed
# [1 0] - class 0
# [0 1] - class 1
y_predict1 = [np.argmax(x) for x in y_predict1]
y_predict2 = [np.argmax(x) for x in y_predict2]
y_predict3 = [np.argmax(x) for x in y_predict3]

y_predict_total = [0 for aux in range (len(y_predict1))] 

# for each data sample, in order to try to obtain a better prediction accuracy,
#   the predictions from the 3 different models were compared.
# if the majority of the models predicted a data sample as belonging to class 1,
#   then the data sample is classified as belonging to class 1. However, if the majority
#   classifies a data sample as belonging to class 0, then it is classified as being a class 0 sample
for i in range(len(y_predict1)):
    if y_predict1[i] == 1 and y_predict2[i] == 1  and y_predict3[i] == 1 :
        y_predict_total[i] = 1
    elif y_predict1[i] == 1 and y_predict2[i] == 1:
        y_predict_total[i] = 1
    elif y_predict2[i] == 1 and y_predict3[i] == 1:
        y_predict_total[i] = 1
    elif y_predict1[i] == 1 and y_predict3[i] == 1:
        y_predict_total[i] = 1
    else:
        y_predict_total[i] = 0


# the final prediction array is converted into an array
y_predict_total_test = np.array(y_predict_total)

