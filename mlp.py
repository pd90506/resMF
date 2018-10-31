import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input, Flatten, Embedding, Dense
from tensorflow.keras.regularizers import l2
from utils import init_normal



def mlp_model(input_dim=10, layers=[10,10], reg_layers=[0,0]):
    """mlp model with integer (to be one-hot encoded) inputs and output a vector of size of last layer"""
    assert len(layers) == len(reg_layers)
    num_layer = len(layers) #Number of layers in the MLP
    # Input variables
    vector_input = Input(shape=(input_dim,), dtype='float32', name = 'mlp_input')
    vector = Dense(layers[0], kernel_regularizer= l2(reg_layers[0]), activation='relu', name = 'layer%d' %0)(vector_input)
    # MLP layers
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer= l2(reg_layers[idx]), activation='relu', name = 'layer%d' %idx)
        vector = layer(vector)
        
    # the final layer as output
    model = Model(inputs=[vector_input], 
                  outputs=vector)
    
    return model


if __name__ == '__main__':
    #NUM_ITEMS = 1000
    mlp = mlp_model()
   # mlp.compile('rmsprop', 'mse')
    item_input = tf.random_uniform(shape=(100,10), dtype=tf.float32)
    output = mlp(item_input)
    print(output)