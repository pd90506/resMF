import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Input, Flatten, Embedding, Dense
from tensorflow.keras.regularizers import l2
from mlp import mlp_model
from utils import init_normal


def aux_mlp_model(num_items, factor=10, layers = [20,10], emb_reg=0, reg_layers=[0,0]):
    """res_mlp_model, the first and last layer of mlp should be of the same size!"""
    assert len(layers) == len(reg_layers)
    #assert factor == layers[-1] # this must hold because a residual is added
    # define inputs
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')

    # define components
    embedding_item = Embedding(input_dim = num_items, output_dim = int(factor), name = 'item_embedding',
                                  embeddings_initializer = init_normal(), embeddings_regularizer = l2(emb_reg), input_length=1)  
    mlp = mlp_model(factor, layers, reg_layers)
    
    # construct network
    latent = embedding_item(item_input)
    latent = Flatten()(latent)
    vector = mlp(latent)
    model = Model(inputs=[item_input], outputs=vector)
    return model


if __name__ == '__main__':
    NUM_ITEMS = 1000
    model = aux_mlp_model(NUM_ITEMS)
   # mlp.compile('rmsprop', 'mse')
    item_input = tf.random_uniform(shape=(100,1), maxval=1000, dtype=tf.int32)
    output = model(item_input)
    print(output)
