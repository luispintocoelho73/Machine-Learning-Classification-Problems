# -*- coding: utf-8 -*-
"""
Created on Tue Nov 9 10:20:12 2021

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
from keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, ELU,GlobalAveragePooling2D, LeakyReLU, BatchNormalization
from keras.layers.convolutional import Convolution2D, Cropping2D, Conv1D, Conv2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
from keras.utils import np_utils
import keras_tuner as kt
import time
import time, glob
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
global inputShape,size

LOG_DIR = f"{int(time.time())}"

def scores(y_real,y_pred,mode):
    ###y_real - ground truth vector 
    ###y_pred - vector of predictions, must have the same shape as y_real
    ###mode   - if evaluating regression ('r') or classification ('c')

    if y_real.shape != y_pred.shape:
        print('confirm that both of your inputs have the same shape')
    else:
        if mode == 'r':
            mse = MSE(y_real,y_pred)
            print('The Mean Square Error is', mse)
            return mse

        elif mode == 'c':
            bacc = BACC(y_real,y_pred)
            print('The Balanced Accuracy is', bacc)
            return bacc

        else:
            print('You must define the mode input.')

# this method was used in order to find a suiting network for the binary classification problem at hand.
# this method uses the keras-tuner package
def build_model(hp):
    model = Sequential()
    model.add(Conv2D(hp.Choice("input_units", [32, 64, 96, 128, 160, 192, 224, 256]), (3,3), input_shape=(50,50,1)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    for i in range(hp.Choice("n_layers", [1, 2, 3, 4])):
        model.add(Conv2D(hp.Choice('conv_units', [32, 64, 96, 128, 160, 192, 224, 256]), (3,3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(hp.Choice('drop_units', [0.1, 0.2, 0.3, 0.4, 0.5])))
        
    model.add(Flatten())
    model.add(Dense(hp.Choice('dense_units', [32, 64, 96, 128, 160, 192, 224, 256])))
    model.add(Activation("relu"))
    model.add(Dropout(hp.Choice('dropout_units', [0.1, 0.2, 0.3, 0.4, 0.5])))
    model.add(Dense(4))
    model.add(Activation("softmax"))
    
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    return model

# the best performing network had these parameters
def kerasModel():
    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape=(50, 50, 1)))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
              
    model.add(Conv2D(64, (3,3)))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    '''
    model.add(Conv2D(128, (3,3)))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(256, (3,3)))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    '''
    model.add(Flatten())  
    model.add(Dense(128))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(4))
    #model.add(BatchNormalization())
    model.add(Activation('softmax'))
    return model


# The data is loaded
X_allfolds = np.load('Xtrain_Classification_Part2.npy')
y_allfolds = np.load('Ytrain_Classification_Part2.npy')
X_test = np.load('Xtest_Classification_Part2.npy')

# The data is reshaped to be inserted in the models
X_allfolds_reshaped = X_allfolds.reshape(len(X_allfolds),50,50,1)
y_allfolds_reshaped = y_allfolds.reshape(len(y_allfolds),1)
X_test_reshaped = X_test.reshape(len(X_test),50,50,1)

# The data is normalized
#X_test_reshaped = X_test_reshaped/255
#X_allfolds_reshaped = X_allfolds_reshaped/255

# create image data augmentation generator
datagen = ImageDataGenerator(rotation_range=20, zoom_range=[0.96,1.0], horizontal_flip=True)
# prepare iterator
it = datagen.flow(X_allfolds_reshaped, batch_size=1, shuffle=False)

X_allfolds_aux = np.empty((2*len(X_allfolds_reshaped),50,50,1))
y_allfolds_aux = np.empty((2*len(y_allfolds_reshaped),1))

for i in range(len(X_allfolds_reshaped)):
    X_allfolds_aux[i] = np.copy(X_allfolds_reshaped[i])
    y_allfolds_aux[i] = y_allfolds_reshaped[i]
    

for i in range(len(X_allfolds_reshaped)):
    # define subplot
    # generate batch of images
    batch = it.next()
    X_allfolds_aux[len(X_allfolds_reshaped)+i] = batch[0].astype('float64')
    y_allfolds_aux[len(X_allfolds_reshaped)+i] = y_allfolds_reshaped[i]
	# convert to unsigned integers for viewing

#plt.imshow(X_allfolds_aux[1])


# The data is normalized
X_test_reshaped = X_test_reshaped/255
X_allfolds_aux = X_allfolds_aux/255

# 5 folders were created: 4 of these folders will be used for training and the remaining folder will be used for validation
kf = KFold(n_splits=5, random_state=7, shuffle=True)

# split input data into folds
kf.split(y_allfolds_aux)

# this for cicle only runs for one interation
for train_index, validation_index in kf.split(X_allfolds_aux):
    X_train, X_validation = X_allfolds_aux[train_index], X_allfolds_aux[validation_index]
    y_train, y_validation = y_allfolds_aux[train_index], y_allfolds_aux[validation_index]

    print(X_train.shape)
    print(X_validation.shape)  
    
    print("train shape X", X_train.shape)
    print("train shape y", y_train.shape)
    
    

    # the y training and validation data was one hot encoded 
    train_labels = tf.keras.utils.to_categorical(y_train, num_classes=4)
    validation_labels = tf.keras.utils.to_categorical(y_validation, num_classes=4)
    
    # an early stopping method was defined in order to avoid overfitting
    early_stopping = [EarlyStopping(monitor = 'val_loss', min_delta =  0.001, patience = 5)]
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=5, min_lr=0.000002)
    #### these lines of code were used in order to find the most suiting architectures for our multiclass classification models, through the kensor-tuner package
    #tuner = kt.RandomSearch(build_model, objective="val_accuracy", max_trials=10, executions_per_trial=2, directory=LOG_DIR)
    
    #tuner.search(x=X_train, y=train_labels, epochs=25, batch_size=64, callbacks=early_stopping, validation_data=(X_validation,validation_labels))

    ## information about the various models 
    #print(tuner.get_best_hyperparameters()[0].values)
    #print(tuner.results_summary())
    #print(tuner.get_best_models()[0].values)
    #####
    
    print("train shape X", X_train.shape)
    print("train shape y", y_train.shape)
    
    # the best model found through keras_tuner was the following 
    #model = kerasModel()
    
    
    # the model was compiled
    #model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    # the data was fit into the model for the training and validation procedures
    #history = model.fit(X_train, train_labels, batch_size= 64, epochs=100, callbacks=[reduce_lr, early_stopping], validation_data=(X_validation, validation_labels))
    
    # the training accuracy for the model was the following
    #metricsTrain = model.evaluate(X_train, train_labels)
    #print("Training Accuracy: ",metricsTrain[1]*100,"%")
    
    print("")
    
    # the validation (testing) accuracy for the model was the following
    #metricsTest = model.evaluate(X_validation,validation_labels)
    #print("Testing Accuracy: ",metricsTest[1]*100,"%")
    
    print("Saving model weights and configuration file")
    #model.save('model.h5')
    print("Saved model to disk")
    
        
    break


aux = 0

# the three best trainings of the kerasModel() model were saved
model1 = load_model('87_478.h5')
model2 = load_model('87.275.h5')
model3 = load_model('86.868.h5')
model4 = load_model('85_617.h5')
model5 = load_model('85.918.h5')
model6 = load_model('dataAug_85.95.h5')
model7 = load_model('85_210.h5')


'''

isto é um comment



'''
'''
# predictions of the three models were computed
y_predict1 = model1.predict(X_test_reshaped)
y_predict2 = model2.predict(X_test_reshaped)
y_predict3 = model3.predict(X_test_reshaped)
y_predict4 = model1.predict(X_test_reshaped)
y_predict5 = model2.predict(X_test_reshaped)
y_predict6 = model3.predict(X_test_reshaped)
y_predict7 = model1.predict(X_test_reshaped)
'''
y_predict1 = model1.predict(X_validation)
y_predict2 = model2.predict(X_validation)
y_predict3 = model3.predict(X_validation)
y_predict4 = model1.predict(X_validation)
y_predict5 = model2.predict(X_validation)
y_predict6 = model3.predict(X_validation)
y_predict7 = model1.predict(X_validation)
# the respective class for each data sample was computed
# [1 0] - class 0
# [0 1] - class 1
y_predict1 = [np.argmax(x) for x in y_predict1]
y_predict2 = [np.argmax(x) for x in y_predict2]
y_predict3 = [np.argmax(x) for x in y_predict3]
y_predict4 = [np.argmax(x) for x in y_predict4]
y_predict5 = [np.argmax(x) for x in y_predict5]
y_predict6 = [np.argmax(x) for x in y_predict6]
y_predict7 = [np.argmax(x) for x in y_predict7]

y_predict_total = [0 for aux in range (len(y_predict1))] 

# for each data sample, in order to try to obtain a better prediction accuracy,
#   the predictions from the 3 different models were compared.
# if the majority of the models predicted a data sample as belonging to class 1,
#   then the data sample is classified as belonging to class 1. However, if the majority
#   classifies a data sample as belonging to class 0, then it is classified as being a class 0 sample
y_predict_total_aux = [0 for aux in range (4)] 
y_predict_total_aux = [0,0,0,0] # (0, 1 , 2, 3) y_predict1[i] = (0, 1, 2, 3)
predicted_class = 0
max_occurrences = 0 

for i in range(len(y_predict1)):
    
    y_predict_total_aux = [0,0,0,0]
    y_predict_total_aux[y_predict1[i]] = y_predict_total_aux[y_predict1[i]] + 1
    y_predict_total_aux[y_predict2[i]] = y_predict_total_aux[y_predict2[i]] + 1
    y_predict_total_aux[y_predict3[i]] = y_predict_total_aux[y_predict3[i]] + 1
    y_predict_total_aux[y_predict4[i]] = y_predict_total_aux[y_predict4[i]] + 1
    y_predict_total_aux[y_predict5[i]] = y_predict_total_aux[y_predict1[i]] + 1
    y_predict_total_aux[y_predict6[i]] = y_predict_total_aux[y_predict1[i]] + 1
    y_predict_total_aux[y_predict7[i]] = y_predict_total_aux[y_predict1[i]] + 1
    
    # the most 
    if y_predict_total_aux[0] >= y_predict_total_aux[1]:
        max_occurrences = y_predict_total_aux[0]
        predicted_class = 0
    if y_predict_total_aux[2] >= max_occurrences:
        max_occurrences = y_predict_total_aux[2]
        predicted_class = 2
    if y_predict_total_aux[3] >= max_occurrences:
        max_occurrences = y_predict_total_aux[3]
        predicted_class = 3     
    y_predict_total[i] = predicted_class
    

# the final prediction array is converted into an array
scores(np.array(y_predict1).reshape(len(y_predict1),1),y_validation,'c')
print("1")

scores(np.array(y_predict2).reshape(len(y_predict2),1),y_validation,'c')
print("2")

scores(np.array(y_predict3).reshape(len(y_predict3),1),y_validation,'c')
print("3")

scores(np.array(y_predict4).reshape(len(y_predict4),1),y_validation,'c')
print("4")

scores(np.array(y_predict5).reshape(len(y_predict5),1),y_validation,'c')
print("5")

scores(np.array(y_predict6).reshape(len(y_predict6),1),y_validation,'c')
print("6")

scores(np.array(y_predict7).reshape(len(y_predict7),1),y_validation,'c')
print("7")

y_predict_total_test = np.array(y_predict_total)
scores(y_predict_total_test.reshape(len(y_predict_total_test),1),y_validation,'c')
print("Democracy")


model4 = load_model('85_210.h5')
y_predict4 = model4.predict(X_validation)
y_predict4 = [np.argmax(x) for x in y_predict4]
y_predict4 = np.array(y_predict4)
scores(y_predict4.reshape(len(y_predict4),1),y_validation,'c')

best_model = load_model("85_210.h5")

y_predict = best_model.predict(X_test_reshaped)


y_predict = [np.argmax(x) for x in y_predict]

y_predict = np.array(y_predict)

y_predict=y_predict.reshape(len(y_predict),1)

import matplotlib.pyplot as plt
plt.imshow(X_allfolds_reshaped[0])