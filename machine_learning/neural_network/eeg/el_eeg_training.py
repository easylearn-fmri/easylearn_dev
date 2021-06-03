# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 12:07:18 2021

@author: Li Chao
Email: lichao19870617@163.com
"""

import os
import numpy as np
import netron
import keras
import tensorflow
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
import json
from eslearn.machine_learning.neural_network.eeg.el_eeg_prep_data import parse_configuration
from keras import backend as K
from keras.models import load_model
meval = ModelEvaluator()



class Trainer():
    def __init__(self, out_dir=None):
        self.out_dir = out_dir
        self._model_file = "eegModel.h5"
        self._modelSaveName = os.path.join(out_dir, self._model_file)

    def prep_data(self, x, y, num_classes):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4,
                                                            shuffle=True, 
                                                            random_state=666)

        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5,
                                                            shuffle=True, 
                                                            random_state=666)

        # input image dimensions
        print('x_train shape:', x_train.shape)
        print('x_validation shape', x_val.shape[0])
        print('x_test shape', x_test.shape[0])

        # convert class vectors to binary class matrices
        self.y_train = np_utils.to_categorical(y_train, num_classes)
        self.y_val = np_utils.to_categorical(y_val, num_classes)
        self.y_test = np_utils.to_categorical(y_test, num_classes)

        self.x_train = x_train.astype('float32')
        self.x_val = x_val.astype('float32')
        self.x_test = x_test.astype('float32')
        return self

    def train(self, 
            input_shape=None,
            num_classes=None,
            batch_size=None, 
            epochs=None,
            lr=None,
            decay=None):

        np.random.seed(666)
        tensorflow.random.set_seed(666)

        # Model
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding='same',input_shape=input_shape))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(10))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes))
        self.model.add(Activation('softmax'))

        # initiate RMSprop optimizer
        self.opt = keras.optimizers.RMSprop(lr=lr, decay=decay)

        # Let's train the self.model using RMSprop
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=self.opt,
                      metrics=[f1])


        #%% Fit
        self.history = self.model.fit(self.x_train, self.y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(self.x_val, self.y_val),
                  shuffle=False)

        return self

    def train_with_pretrained_model(self,
                                    batch_size=None,
                                    epochs=None):

        np.random.seed(666)
        tensorflow.random.set_seed(666)

        # Load pre-trained model
        self.model = load_model(self._modelSaveName, compile=False)
        
        # Comp
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=self.opt,
                      metrics=[f1])
        
        # Fit
        self.history = self.model.fit(self.x_train, self.y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(self.x_val, self.y_val),
                  shuffle=False)

        return self

    def save_model_and_loss(self, 
             historySaveName="trainHistoryDict.txt",
             lossSaveName = "loss.pdf"):
        
        # Save and Vis loss
        historySaveName = os.path.join(self.out_dir, historySaveName)
        lossSaveName = os.path.join(self.out_dir, lossSaveName)

        self.model.save(self._modelSaveName)
        with open(historySaveName, 'wb') as file_pi:
            pickle.dump(self.history.history, file_pi)

        # with open(historySaveName,'rb') as file_pi:
        #     history=pickle.load(file_pi)
        
        # Visualize loss
        history = self.history.history
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
    
    def vis_net(self):
        # Vis model
        netron.start(self._modelSaveName, address=None, browse=True)
        return self

    #%% Eval
    def eval(self):
        # self.model = load_model(self._modelSaveName, compile=False)
        all_predict_label = np.argmax(self.model.predict(self.x_train), axis=-1)
        all_predict_prob = self.model.predict_proba(self.x_train)[:,1]
        meval.binary_evaluator(self.y_train[:,1], all_predict_label, all_predict_prob, 
                            is_savefig=True, out_name=os.path.join(self.out_dir, "performancesOfTraining.pdf"),
                            legend1="Negative", legend2="Positive")
        plt.suptitle("Training data")
        
        all_predict_label = np.argmax(self.model.predict(self.x_val), axis=-1)
        all_predict_prob = self.model.predict_proba(self.x_val)[:,1]
        meval.binary_evaluator(self.y_val[:,1], all_predict_label, all_predict_prob, 
                            is_savefig=True, out_name=os.path.join(self.out_dir, "performancesOfVal.pdf"),
                            legend1="Negative", legend2="Positive")
        plt.suptitle("Validation data")
        
    def test(self):
        # self.model = load_model(self._modelSaveName, compile=False)

        all_predict_label = np.argmax(self.model.predict(self.x_test), axis=-1)
        all_predict_prob = self.model.predict_proba(self.x_test)[:,1]
        (accuracy, sensitivity, specificity, auc, confusion_matrix_values) = \
            meval.binary_evaluator(self.y_test[:,1], all_predict_label, all_predict_prob, 
                            is_savefig=True, out_name=os.path.join(self.out_dir, "performancesOfTest.pdf"),
                            legend1="Negative", legend2="Positive")
    
        plt.suptitle("Test data")
        
        n = all_predict_label.shape[0]
        return n, accuracy


#%%
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

if __name__ == "__main__":
    (data_file, frequency, theta, alpha, beta,image_size, frame_duration, overlap, locs_2d,
    out_dir, num_classes, batch_size, epochs, lr, decay) =\
             parse_configuration("./config.json")

    input_shape = (image_size, image_size, 3)

    trainer = Trainer(out_dir=out_dir)
    data = np.load(r"D:\software\miniconda\conda\Lib\site-packages\eslearn\machine_learning\neural_network\eeg/eegData.npz")
    x, y = data['x'],  data['y']
    trainer.prep_data(x, y, num_classes)
    trainer.train(num_classes=num_classes,
                                    input_shape=input_shape,
                                    batch_size=batch_size, 
                                    epochs=epochs, 
                                    lr=lr,
                                    decay=decay)

    trainer.save_model_and_loss( 
                 modelSaveName="eegModel.h5", 
                 historySaveName="trainHistoryDict.txt",
                 lossSaveName = "loss.pdf")
    
    trainer.eval()
