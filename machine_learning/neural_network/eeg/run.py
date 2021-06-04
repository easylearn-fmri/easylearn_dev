# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 09:29:09 2021

@author: Li Chao
"""

import os
import numpy as np
from  eslearn.machine_learning.neural_network.eeg.el_eeg_prep_data import parse_configuration, make_data_pipeline, DataLoader_
from eslearn.machine_learning.neural_network.eeg.el_eeg_training import Trainer
from eslearn.statistical_analysis.el_binomialtest import binomialtest
import matplotlib.pyplot as plt

class EEGClassifier():

  def __init__(self, configuration_file=None, out_dir=None):
    self.configuration_file = configuration_file
    self.out_dir = out_dir
    self.data_path = os.path.join(out_dir, "eegclfData.npz")
    
  def parse_configuration(self):
    """parse configuration"""

    (self.frequency, self.image_size, self.frame_duration, self.overlap, self.locs_2d,
        self.num_classes, self.batch_size, self.epochs, self.lr, self.decay) =\
                 parse_configuration(self.configuration_file)

    return self

  def prepare_data(self):
    """ Prepare data"""
    self.parse_configuration()
    data_loader = DataLoader_(self.configuration_file)
    data_loader.load_data()
    x, y = make_data_pipeline(data_loader.input_files,
                    data_loader.targets_,
                    self.image_size,
                    self.frame_duration,
                    self.overlap,
                    self.locs_2d,
                    self.frequency)

    np.savez(self.data_path, x=x, y=y)
    print("=="*30)
    return self

  def train(self):
    #%% Train
    
    print("#"*50)
    print("Training...")
    input_shape = (self.image_size, self.image_size, 3)

    self.trainer = Trainer(out_dir=self.out_dir)
    data = np.load(self.data_path)
    x, y = data['x'],  data['y']
    self.trainer.prep_data(x, y, self.num_classes)
    self.trainer.train(num_classes=self.num_classes,
                                    input_shape=input_shape,
                                    batch_size=self.batch_size, 
                                    epochs=self.epochs, 
                                    lr=self.lr,
                                    decay=self.decay)
    print("=="*30)
    return self
  
  def train_with_pretrained_model(self):
     self.trainer.train_with_pretrained_model(batch_size=self.batch_size, 
                                             epochs=self.epochs)
     print("=="*30)
     return self

  def eval(self):
    self.trainer.eval()
    print("=="*30)
    return self

  def save_model_and_loss(self):
    self.trainer.save_model_and_loss()

    self.trainer.vis_net()
    print("=="*30)
    return self

  def test(self):
    self.n, self.accuracy = self.trainer.test()
    self.run_statistical_analysis()
    plt.suptitle(f"Test data (Pvalue={self.pv})")
    print("=="*30)
    return self

  def run_statistical_analysis(self):
    # stat
    self.pv = binomialtest(self.n, 
                      k=np.int32(self.n * self.accuracy), 
                      p=0.5)
    print(f"P value of accuracy = {self.pv}")
    print("=="*30)
    return self


if __name__ == "__main__":
  eegclf = EEGClassifier(configuration_file=r"D:\work\lichao\test_eegclf/eegclf.json",
                         out_dir=r"D:\work\lichao\test_eegclf")
  eegclf.parse_configuration()
  # eegclf.prepare_data()
  eegclf.train()
  eegclf.save_model_and_loss()
  eegclf.train_with_pretrained_model()
  eegclf.test()

