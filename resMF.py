### This model mimic a resNet style connection between embedding layer and GMF layer
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K

from tensorflow.keras import initializers

# args initiation
class Args(object):
    """A simulator of parser in jupyter notebook"""
    def __init__(self):
        self.path = 'Data/'
        self.dataset = '100k'
        self.epochs = 100
        self.batch_size = 256
        self.num_factors = 16
        self.layers = '[64,32,16,8]'
        self.reg_mf = '[0,0]'
        self.reg_layers = '[0,0,0,0]'
        self.num_neg = 4
        self.lr = 0.001
        self.learner = 'adam'
        self.verbose = 0
        self.out = 1
        self.mf_pretrain = ''
        self.mlp_pretrain= ''

def init_normal(shape=[0,0.05], seed=None):
    """ A easy initializer"""
    mean, stddev = shape
    return initializers.RandomNormal(mean=mean, stddev=stddev, seed=seed)




