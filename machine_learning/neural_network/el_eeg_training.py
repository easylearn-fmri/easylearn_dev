# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 12:07:18 2021

@author: Li Chao
Email: lichao19870617@163.com
"""

import os
import pandas as pd
import numpy as np
import keras
import netron
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from eslearn.model_evaluator import ModelEvaluator
import matplotlib.pyplot as plt
meval = ModelEvaluator()


img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 3)
num_classes = 2

class Training():
    def __init__(self, workingDir="./"):
        self.workingDir = workingDir

    def prep_data(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4,
                                                            shuffle=True, 
                                                            random_state=666)

        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5,
                                                            shuffle=True, 
                                                            random_state=666)

        # input image dimensions
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        self.y_train = np_utils.to_categorical(y_train, num_classes)
        self.y_val = np_utils.to_categorical(y_val, num_classes)
        self.y_test = np_utils.to_categorical(y_test, num_classes)

        self.x_train = x_train.astype('float32')
        self.x_val = x_val.astype('float32')
        self.x_test = x_test.astype('float32')
        return self

    def train(self, 
            batch_size=128, 
            epochs=5,
            lr=0.0005,
            decay=1e-6):

        #%% Model
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(10))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        #%%
        from keras import backend as K
         
        def f1(y_true, y_pred):
            def recall(y_true, y_pred):
                """Recall metric.
                Only computes a batch-wise average of recall.
                Computes the recall, a metric for multi-label classification of
                how many relevant items are selected.
                """
                
                
                true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
                recall = true_positives / (possible_positives + K.epsilon())
                return recall
         
            def precision(y_true, y_pred):
                """Precision metric.
                Only computes a batch-wise average of precision.
                Computes the precision, a metric for multi-label classification of
                how many selected items are relevant.
                """
                true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
                precision = true_positives / (predicted_positives + K.epsilon())
                return precision
            precision = precision(y_true, y_pred)
            recall = recall(y_true, y_pred)
            return 2*((precision*recall)/(precision+recall+K.epsilon()))
         

        # initiate RMSprop optimizer
        opt = keras.optimizers.RMSprop(lr=0.0005, decay=1e-6)

        # Let's train the model using RMSprop
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=[f1])


        #%% Fit
        history = model.fit(self.x_train, self.y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(self.x_val, self.y_val),
                  shuffle=True)

        return model, history

    def save(self, model, history, 
             modelSaveName="eegModel.h5", 
             historySaveName="trainHistoryDict.txt",
             lossSaveName = "loss.pdf"):
        

        #%% Save and Vis loss
        self.modelSaveName = os.path.join(self.workingDir, modelSaveName)
        historySaveName = os.path.join(self.workingDir, historySaveName)
        lossSaveName = os.path.join(self.workingDir, lossSaveName)

        model.save(modelSaveName)
        with open(historySaveName, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        # with open(historySaveName,'rb') as file_pi:
        #     history=pickle.load(file_pi)
        
        history = history.history
        epochs = len(history['loss'])
        plt.plot(range(epochs), history['loss'], label='Training loss')
        plt.plot(range(epochs), history['val_loss'], label='Validation loss')
        plt.legend()

        ax = plt.gca()
        ax.set_ylabel("Loss", fontsize=15)
        ax.set_xlabel("Epoch", fontsize=15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        plt.show()
        plt.savefig(lossSaveName)

        return self

    #%% Eval
    def eval(self):
        from keras.models import load_model
        model = load_model(self.modelSaveName, compile=False)

        all_predict_label = np.argmax(model.predict(self.x_train), axis=-1)
        all_predict_prob = model.predict_proba(self.x_train)[:,1]
        meval.binary_evaluator(self.y_train[:,1], all_predict_label, all_predict_prob, 
                            is_savefig=True, out_name=os.path.join(self.workingDir, "performancesOfTraining.pdf"),
                            legend1="Negative", legend2="Positive")

        all_predict_label = np.argmax(model.predict(self.x_val), axis=-1)
        all_predict_prob = model.predict_proba(self.x_val)[:,1]
        meval.binary_evaluator(self.y_val[:,1], all_predict_label, all_predict_prob, 
                            is_savefig=True, out_name=os.path.join(self.workingDir, "performancesOfVal.pdf"),
                            legend1="Negative", legend2="Positive")


        all_predict_label = np.argmax(model.predict(self.x_test), axis=-1)
        all_predict_prob = model.predict_proba(self.x_test)[:,1]
        meval.binary_evaluator(self.y_test[:,1], all_predict_label, all_predict_prob, 
                            is_savefig=True, out_name=os.path.join(self.workingDir, "performancesOfTest.pdf"),
                            legend1="Negative", legend2="Positive")


if __name__ == "__main__":
    trainer = Training()
    data = np.load(r"D:\software\miniconda\conda\Lib\site-packages\eslearn\machine_learning\neural_network/eegData.npz")
    x, y = data['x'],  data['y']
    trainer.prep_data(x, y)
    model, history = trainer.train(batch_size=128, 
                                    epochs=5, 
                                    lr=0.0005,
                                    decay=1e-6)
    trainer.save(model, history, 
                 modelSaveName="eegModel.h5", 
                 historySaveName="trainHistoryDict.txt",
                 lossSaveName = "loss.pdf")
    
    trainer.eval()
