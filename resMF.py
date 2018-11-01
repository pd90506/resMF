### This model mimic a resNet style connection between embedding layer and GMF layer
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

from res_gmf import res_gmf_model

# args initiation
class Args(object):
    """A simulator of parser in jupyter notebook"""
    def __init__(self):
        self.path = 'Data/'
        self.dataset = '100k'
        self.epochs = 100
        self.batch_size = 256
        self.num_factors = 8
        self.layers = [64,32,16,8]
        self.reg_mf = 0.0001
        self.reg_layers = [0.0001,0.0001,0.0001,0.0001]
        self.num_neg = 4
        self.lr = 0.001
        self.learner = 'adam'
        self.verbose = 1
        self.out = 1
        self.mf_pretrain = ''
        self.mlp_pretrain= ''

        self._init_nums()
        

    def _init_nums(self):
        if self.dataset=='1m':
            self.num_users = 6040
            self.num_items = 3706
        elif self.dataset=='100k':
            self.num_users = 671
            self.num_items = 9125

if __name__ == '__main__':
    # init args
    args = Args()
    # creating model
    model = res_gmf_model(args.num_users, args.num_items, factor=args.num_factors, layers=args.layers, reg_mf=args.reg_mf, reg_layers=args.reg_layers)
    # get input dataset
    user_input = np.random.randint(args.num_users, size=100, dtype='int32')
    item_input = np.random.randint(args.num_items, size=100, dtype='int32')
    ratings = np.random.randint(2, size=100, dtype='int32')
    model.compile(optimizer=Adam(lr=args.lr), loss='binary_crossentropy')

    # training
    hist = hist = model.fit([user_input, item_input], #input
                         ratings, # labels 
                         batch_size=args.batch_size, epochs=1000, verbose=args.verbose, shuffle=True)




