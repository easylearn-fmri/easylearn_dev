from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from utils import *
from models import DSGCNN
from tensorflow import set_random_seed
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy
from scipy.sparse import csr_matrix, lil_matrix
import numpy as np



def myaccalc(pred,yhat):
    return np.sum(np.argmax(pred,1)==np.argmax(yhat,1)) 


# random seed for reproducability
seed = 32 
np.random.seed(seed)
tf.set_random_seed(seed)  



# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')  
flags.DEFINE_integer('epochs', 500, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 320, 'Number of units in hidden graph conv layer 1.') 
flags.DEFINE_integer('hidden2', 100, 'Number of units in hidden graph conv layer 2.')
flags.DEFINE_integer('dense', 100, 'Number of units in hidden dense layer.')  
flags.DEFINE_float('dropout', 0.10, 'Dropout rate (1 - keep probability).')   
flags.DEFINE_float('weight_decay', 0.0, 'Weight for L2 loss on embedding matrix.') 
flags.DEFINE_integer('nkernel', 3, 'number of kernels')



nkernel=flags.FLAGS.nkernel
# how many times do you want to update parameters over one epoch. batchsize=trainsize/bsize
bsize=3

# read data
a=sio.loadmat('enzymes.mat')
# list of adjacency matrix
A=a['A'][0]
# list of features
F=a['F'][0]
# label of graphs
Y=a['Y'][0]
# test train index for 10-fold test
TRid=a['tr']
TSid=a['ts']


 
# max number of nodes
nmax=0
for i in range(0,len(A)):
    nmax=max(nmax,A[i].shape[0])


# number of node per graph
ND=np.zeros((len(A),1)) 
# node feature matrix
FF=np.zeros((len(A),nmax,3))
# one-hot coding output matrix 
YY=np.zeros((len(A),6))
# Convolution kernels, supports
SP=np.zeros((len(A),nkernel,nmax,nmax))


# prepare inputs, outputs, convolution kernels for each graph
for i in range(0,len(A)):  
    # number of node in graph
    n=F[i].shape[0]
    ND[i,0]=n

    # feature matrix
    FF[i,0:n,:]= F[i]

    # one-hot coding output matrix
    YY[i,Y[i]]=1

    # set kernels
    chebnet = chebyshev_polynomials(A[i], nkernel-1)
    for j in range(0,nkernel):
        SP[i,j,0:n,0:n]=chebnet[j].toarray() 

    ## GCN convolution kernel
    #gcn= (normalize_adj(A[i] + sp.eye(A[i].shape[0]))).toarray()    
    #SP[i,0,0:n,0:n]=gcn

    ## MLP convolution kernel
    #mlp=np.eye(n)
    #SP[i,0,0:n,0:n]=gcn

    ## A and I convolution kernel
    # SP[i,0,0:n,0:n]=np.eye(n)
    # SP[i,1,0:n,0:n]=A[i]

NB=np.zeros((FLAGS.epochs,10))


for fold in range(0,10):   

    # train and test ids
    trid=TRid[fold]
    tsid=TSid[fold]
    
    placeholders = {        
            'support': tf.placeholder(tf.float32, shape=(None,nkernel,nmax,nmax)),            
            'features': tf.placeholder(tf.float32, shape=(None,nmax, FF.shape[2])),
            'labels': tf.placeholder(tf.float32, shape=(None, 6)),  
            'nnodes': tf.placeholder(tf.float32, shape=(None, 1)),               
            'dropout': tf.placeholder_with_default(0., shape=()),        
    }

    model = DSGCNN(placeholders, input_dim=FF.shape[2],nkernel=nkernel,logging=True,agg='mean')  

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # train data placeholders
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: YY[trid,:]})    
    feed_dict.update({placeholders['features']: FF[trid,:,:]})
    feed_dict.update({placeholders['support']: SP[trid,:,:,:]}) 
    feed_dict.update({placeholders['nnodes']: ND[trid,]})    
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # test data placeholders
    feed_dictT = dict()
    feed_dictT.update({placeholders['labels']: YY[tsid,:]})    
    feed_dictT.update({placeholders['features']: FF[tsid,:,:]})
    feed_dictT.update({placeholders['support']: SP[tsid,:,:,:]})    
    feed_dictT.update({placeholders['nnodes']: ND[tsid,]})     
    feed_dictT.update({placeholders['dropout']: 0})        
    
    ind=np.round(np.linspace(0,len(trid),bsize+1))    
    
    for epoch in range(FLAGS.epochs): 
               
        np.random.shuffle(trid)
        for i in range(0,bsize): # batch training
            feed_dictB = dict()
            bid=trid[int(ind[i]):int(ind[i+1])]
            feed_dictB.update({placeholders['labels']: YY[bid,:]})    
            feed_dictB.update({placeholders['features']: FF[bid,:,:]})
            feed_dictB.update({placeholders['support']: SP[bid,:,:,:]})            
            feed_dictB.update({placeholders['nnodes']: ND[bid,]})            
            feed_dictB.update({placeholders['dropout']: FLAGS.dropout})
            # train for batch data
            outs = sess.run([model.opt_op], feed_dict=feed_dictB)

        # check performance for all train sample
        outs = sess.run([model.accuracy, model.loss, model.entropy,model.outputs], feed_dict=feed_dict)
        # check performance for all test sample
        outsT = sess.run([model.accuracy, model.loss, model.entropy,model.outputs], feed_dict=feed_dictT) 

        # number of true classified test graph
        vtest=myaccalc(outsT[3],YY[tsid,:])

        NB[epoch,fold]=vtest
        if np.mod(epoch + 1,1)==0 or epoch==0:
            print(fold," Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),"train_xent=", "{:.5f}".format(outs[2]),"train_acc=", "{:.5f}".format(outs[0]),"test_loss=", "{:.5f}".format(outsT[1]), 
            "test_xent=", "{:.5f}".format(outsT[2]), "test_acc=", "{:.5f}".format(outsT[0]), " ntrue=", "{:.0f}".format(vtest))


import pandas as pd
pd.DataFrame(NB).to_csv('testresultsoverepoch.csv') 
