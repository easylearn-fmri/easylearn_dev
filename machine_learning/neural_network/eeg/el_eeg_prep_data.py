# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 12:07:18 2021

@author: Li Chao
Email: lichao19870617@163.com
"""

import time
import os
import scipy.io as sio
import numpy as np
from numpy import genfromtxt
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
from eslearn.machine_learning.neural_network.eeg.eeg_learn_functions import *
from eslearn.base import BaseMachineLearning, DataLoader
import json


def parse_configuration(config_file=None):
    with open(config_file, 'r', encoding='utf-8') as config:
            configuration = config.read()
            configuration = json.loads(configuration)
    
    config_eegclf = configuration.get("machine_learning", None).get("Deep learning", None).get("EEGClassifier")
    
    coordinate = config_eegclf.get("coordinate", None).get("value", None)
    frequency = np.float32(config_eegclf.get("frequency", None).get("value", None))
    image_size = np.int32(config_eegclf.get("image_size", None).get("value", None))
    frame_duration = np.float32(config_eegclf.get("frame_duration", None).get("value", None))
    overlap = np.float32(config_eegclf.get("overlap", None).get("value", None))
    num_classes = np.int32(config_eegclf.get("num_classes", None).get("value", None))
    batch_size = np.int32(config_eegclf.get("batch_size", None).get("value", None))
    epochs = np.int32(config_eegclf.get("epochs", None).get("value", None))
    lr = np.float32(config_eegclf.get("learning_rate", None).get("value", None))
    decay = np.float32(config_eegclf.get("decay", None).get("value", None))

    # (data_file, coordinate, frequency,image_size, frame_duration, overlap,
    # save_path, num_classes, batch_size, epochs, lr, decay) = \
    # (configuration["data_file"], configuration["coordinate"], configuration["frequency"], 
    # configuration["image_size"], configuration["frame_duration"], configuration["overlap"],

    # configuration["save_path"], configuration["num_classes"], 
    # configuration["batch_size"], configuration["epochs"], configuration["lr"], configuration["decay"])
    locs_2d = pd.read_csv(coordinate)
    

    return (frequency,image_size, frame_duration, overlap, locs_2d,
            num_classes, batch_size, epochs, lr, decay)

class DataLoader_(DataLoader):
    def __init__(self, data_file):
        self.data_file = data_file
        super(DataLoader_, self).__init__(self.data_file)
        
    def load_data(self):
        self.get_configuration_()
        loaded_data = self.configuration.get('data_loading', None)
        
        #%% ==========================================Check datasets=================================
        # NOTE.: That check whether the feature dimensions of the same modalities in different groups are equal
        # is placed in the next section.
        targets = {}
        self.covariates_ = {}
        for i, gk in enumerate(loaded_data.keys()):
            
            # Check the number of modality across all group is equal
            if i == 0:
                n_mod = len(loaded_data.get(gk).get("modalities").keys())
            else:
                if n_mod != len(loaded_data.get(gk).get("modalities").keys()):
                    raise ValueError("The number of modalities in each group is not equal, check your inputs")
                    return
                n_mod = len(loaded_data.get(gk).get("modalities").keys())
                
            # Get targets
            targets_input = loaded_data.get(gk).get("targets")
            targets[gk] = self.read_targets(targets_input)  

    
            # Get covariates
            covariates_input = loaded_data.get(gk).get("covariates")
            if (isinstance(covariates_input, str) and
                covariates_input.strip() != "" and
                (not os.path.isfile(covariates_input))):  # Easylearn only supports file input for covariates
                raise ValueError("Easylearn only supports file input for covariates, check your covariates for {gk}")
            self.covariates_[gk] = self.base_read(covariates_input)
            
            # Check the number of files in each modalities in the same group is equal
            for j, mk in enumerate(loaded_data.get(gk).get("modalities").keys()):
                modality = loaded_data.get(gk).get("modalities").get(mk)
                
                # Filses
                input_files = modality.get("file")
                if j == 0:
                    n_file = self.get_file_len(input_files)  # Initialize n_file
                else:
                    if n_file != self.get_file_len(input_files):  # Left is previous, right is current loop
                        raise ValueError(f"The number of files in each modalities in {gk} is not equal, check your inputs")
                        return
                n_file = self.get_file_len(input_files)  # Update n_file

                # Check the number of targets in each modalities is equal to the number of files          
                # If the type of targets is list, and number of files are not equal to targets, then raise error
                if (isinstance(targets[gk],list)) and (n_file != len(targets[gk])):
                    raise ValueError(f"The number of files in {mk} of {gk} is not equal to the number of targets, check your inputs")
                    return
        
                # Check the number of lines of covariates in each modalities is equal to the number of files
                # If covariates is not int (0), and number of files are not equal to covariates, then raise error
                if (not isinstance(self.covariates_[gk],int)) and (n_file != len(self.covariates_[gk])):
                    raise ValueError(f"The number of files in {mk} of {gk} is not equal to its' number of covariates, check your inputs")
                    return
            
            if i == 0:
                input_files_all = input_files.copy()
            else:
                input_files_all.extend(input_files)
                
        #%% ==========================================Get selected datasets =================================
        shape_of_data = {}
        feature_applied_mask_all = {}
        feature_applied_mask_and_add_otherinfo = {}
        col_drop = {}
        self.mask_ = {}
        self.data_format_ = {}
        self.affine_ =  {}

        for gi, gk in enumerate(loaded_data.keys()): 
            col_drop[gk] = ["__Targets__"]
            shape_of_data[gk] = {}
            feature_applied_mask_and_add_otherinfo[gk] = {}
            feature_applied_mask_all[gk] = {}
            self.mask_[gk] = {}
            self.data_format_[gk] = {}
            self.affine_[gk] = {}
            
            for jm, mk in enumerate(loaded_data.get(gk).get("modalities").keys()):
                modality = loaded_data.get(gk).get("modalities").get(mk)
               
                # Get files
                # If only input one file for one modality, 
                # then I think the file contained multiple cases' data
                input_files = modality.get("file")
                n_file = self.get_file_len(input_files)
                if len(input_files) == 1:
                    one_file_per_modality = True
                else:
                    one_file_per_modality = False
                
                # Get features' format and affine for each modality
                # I think all files in on modality are in the same format
                # So I take the first file in corresponding modality as example file
                # TODO: other situations
                self.data_format_[gk][mk], self.affine_[gk][mk] = self.get_data_format(input_files[0])

                # Get Features
                all_features = self.read_file(input_files, False)
                if one_file_per_modality:
                    all_features_ = list(all_features)[0]
                else:
                    all_features_ = False

                # Get cases' name (unique ID) in this modality
                # If one_file_per_modality = False, then each file name must contain r'.*(sub.?[0-9].*).*'
                # If one_file_per_modality = True and all_features_ is DataFrame, 
                # then the DataFrame must have header of "__ID__" which contain the unique_identifier,
                # otherwise easylearn will take the first column as "__ID__".
                if isinstance(all_features_, pd.core.frame.DataFrame) and ("__ID__" not in all_features_.columns):
                    # raise ValueError(f"The dataset of {input_files} did not have '__ID__' column, check your dataset")
                    unique_identifier = all_features_.iloc[:,0]  # Take the first column as __ID__
                    print(f"The dataset of {input_files} did not have '__ID__' column, easylearn take the first column as ID\n")
                elif isinstance(all_features_, pd.core.frame.DataFrame) and ("__ID__" in all_features_.columns):
                    unique_identifier = pd.DataFrame(all_features_["__ID__"])
                    all_features_.drop("__ID__", axis=1, inplace=True)
                    all_features = [all_features_]
                elif isinstance(all_features_, np.ndarray):
                    all_features_ = pd.DataFrame(all_features_)
                    all_features = [all_features_]
                    unique_identifier = pd.DataFrame(all_features_.iloc[:,0], dtype=np.str) # Take the first column as __ID__
                    unique_identifier.columns = ["__ID__"]
                    all_features = [all_features_.iloc[:,1:]]
                else:
                    autogen = True if (isinstance(targets[gk],int)) else False
                    unique_identifier = self.extract_id(input_files, autogen)  # Multiple files
                        
            #%% =====================Match targets and covariates with unique_identifier_==============================
            # NOTE. subj-name is come from the first modality due to the second and later modality are sorted 
            # according with the first one using pd.merge method 

            # Sort targets and check
            if (isinstance(targets[gk],int)):
                targets[gk] = [targets[gk] for ifile in range(n_file)]
                targets[gk] = pd.DataFrame(targets[gk])
                targets[gk]["__ID__"] = unique_identifier
                targets[gk].rename(columns={0: "__Targets__"}, inplace=True)
            elif isinstance(targets[gk], pd.core.frame.DataFrame) and ("__ID__" not in targets[gk].columns):
                # raise ValueError(f"The targets of {gk} did not have '__ID__' column, check your targets") 
                print(f"The targets of {gk} did not have '__ID__' column, easylearn take the first column as ID\n") 
                # Take the first column as __ID__, and the second column as __Targets__
                targets[gk].columns = ["__ID__", "__Targets__"] 
            elif isinstance(targets[gk], np.ndarray):
                targets[gk] = pd.DataFrame(targets[gk])
                # Take the first column as __ID__, and the second as __Targets__
                targets[gk].rename(columns={0:"__ID__", 1:"__Targets__"}, inplace=True)
            
            targets[gk] = pd.merge(unique_identifier, targets[gk], left_on="__ID__", right_on="__ID__", how='inner')
            if targets[gk].shape[0] != n_file:
                    raise ValueError(f"The subjects' ID in targets is not totally matched with its' data file name in {mk} of {gk} , check your ID in targets or check your data file name")

            # Check whether the feature dimensions of the same modalities in different groups are equal
            # shape_of_data[gk][mk] = feature_applied_mask_all[gk].shape
            # if gi == 0:
            #     gk_pre = gk
            # else:
            #     if shape_of_data[gk_pre][mk][-1] != shape_of_data[gk][mk][-1]:
            #         raise ValueError(f"Feature dimension of {mk} in {gk_pre} is {shape_of_data[gk_pre][mk][-1]} which is not equal to {mk} in {gk}: {shape_of_data[gk][mk][-1]}, check your inputs")
             

            # Concat datasets across different group
            # unique_identifier_ = pd.DataFrame([f"{gk}_{ui}" for ui in unique_identifier])
            if gi == 0:
                self.id_ = unique_identifier
                self.targets_ = targets[gk]["__Targets__"]
            else:
                self.id_ = pd.concat([self.id_, unique_identifier])
                self.targets_ = pd.concat([self.targets_,  targets[gk]["__Targets__"]])

        self.id_ = self.id_.values
        self.targets_ = np.float64(self.targets_.values)
        self.input_files = input_files_all
        return self


def get_fft(snippet, frequency):
    #Ts = len(snippet)/frequency/frequency; # sampling interval
    snippet_time = len(snippet)/frequency
    Ts = 1.0/frequency; # sampling interval
    t = np.arange(0,snippet_time,Ts) # time vector

    # ff = 5;   # frequency of the signal
    # y = np.sin(2*np.pi*ff*t)
    y = snippet
#     print('Ts: ',Ts)
#     print(t)
#     print(y.shape)
    n = len(y) # length of the signal
    k = np.arange(n)
    T = n/frequency
    frq = k/T # two sides frequency range
    frq = frq[range(n//2)] # one side frequency range

    Y = np.fft.fft(y)/n # fft computing and normalization
    Y = Y[range(n//2)]
    #Added in: (To remove bias.)
    #Y[0] = 0
    return frq,abs(Y)

def theta_alpha_beta_averages(f,Y):
    theta_range = (4,8)
    alpha_range = (8,12)
    beta_range = (12,40)
    theta = Y[(f>theta_range[0]) & (f<=theta_range[1])].mean()
    alpha = Y[(f>alpha_range[0]) & (f<=alpha_range[1])].mean()
    beta = Y[(f>beta_range[0]) & (f<=beta_range[1])].mean()
    return theta, alpha, beta

def theta_alpha_beta_all_channels(signal, frequency):
    f_Y = [get_fft(signal[i,:], frequency) for i in range(signal.shape[0])]
    power = [theta_alpha_beta_averages(f_,Y_) for f_, Y_ in f_Y]
    return np.array(power)

def make_steps(samples,frame_duration,overlap, frequency):
    '''
    in:
    samples - number of samples in the session
    frame_duration - frame duration in seconds
    overlap - float fraction of frame to overlap in range (0,1)

    out: list of tuple ranges
    '''
    #steps = np.arange(0,len(df),frame_length)
    i = 0
    intervals = []
    samples_per_frame = frequency * frame_duration
    while i+samples_per_frame <= samples:
        intervals.append((i,i+samples_per_frame))
        i = i + samples_per_frame - int(samples_per_frame*overlap)
    return intervals       

def make_frames(df,frame_duration, overlap, frequency):
    '''
    in: dataframe or array with all channels, frame duration in seconds
    out: array of theta, alpha, beta averages for each probe for each time step
        shape: (n-frames,m-probes,k-brainwave bands)
    '''
    
    frame_length = frequency*frame_duration
    frames = []
    steps = make_steps(len(df),frame_duration,overlap, frequency)
    for i,_ in enumerate(steps):
        frame = []
        # if i == 0:
        #     continue
        # else:
        for channel in df.columns:
            snippet = np.array(df.loc[steps[i][0]:steps[i][1],int(channel)])
            f,Y =  get_fft(snippet, frequency)
            theta, alpha, beta = theta_alpha_beta_averages(f,Y)
            frame.append([theta, alpha, beta])

        frames.append(frame)
    return np.array(frames)

def make_data_pipeline(file_names,labels,image_size,frame_duration, overlap, locs_2d, frequency):
    '''
    IN:
    file_names - list of strings for each input file (one for each subject)
    labels - list of labels for each
    image_size - int size of output images in form (x, x)
    frame_duration - time length of each frame (seconds)
    overlap - float fraction of frame to overlap in range (0,1)

    OUT:
    X: np array of frames (unshuffled)
    y: np array of label for each frame (1 or 0)
    '''

    frame_length = frequency * frame_duration

    print('Generating training data...')


    for i, file in enumerate(file_names):
        print ('Processing session: ',file, '. (',i+1,' of ',len(file_names),')')
        data = genfromtxt(file, delimiter=',').T
        nChannles = data.shape[1]
        df = pd.DataFrame(data)

        X_0 = make_frames(df,frame_duration, overlap, frequency)
        #steps = np.arange(0,len(df),frame_length)
        X_1 = X_0.reshape(len(X_0),nChannles*3)

        images = gen_images(np.array(locs_2d),X_1, image_size, normalize=False)
        images = np.swapaxes(images, 1, 3)
        print(len(images), ' frames generated with label ', labels[i], '.')
        print('\n')
        if i == 0:
            X = images
            y = np.ones(len(images))*labels[0]
        else:
            X = np.concatenate((X,images),axis = 0)
            y = np.concatenate((y,np.ones(len(images))*labels[i]),axis = 0)


    return X,np.array(y)


if __name__ == "__main__":
    st = time.time()

    (frequency,image_size, frame_duration, overlap, locs_2d,
            save_path, num_classes, batch_size, epochs, lr, decay) = \
        parse_configuration(config_file="eeg.json")

    data_loader = DataLoader_(data_file="eeg.json")
    data_loader.load_data()
    x, y = make_data_pipeline(data_loader.input_files,
                       data_loader.targets_,
                       image_size,
                       frame_duration,
                       overlap,
                       locs_2d,
                       frequency)
    et = time.time()
    print(et-st)



