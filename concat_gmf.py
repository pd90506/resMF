import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Input, Multiply, Dense
from tensorflow.keras.regularizers import l2
from gmf import gmf_model
from concat_mlp import concat_mlp_model
from utils import init_normal

def concat_gmf_model(num_users, num_items, factor=10, layers=[10], reg_mf=0, emb_reg=0, reg_layers=[0]):
    """
    Args: 
        num_users: number of users
        num_items: number of items
        factor: embedding factor
        layers: mlp layers
        reg_mf: gmf regularization coef
        emb_reg: embedding regularization
        reg_layers: mlp layers regularizations
    """
    assert len(layers) == len(reg_layers)    

    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')

    # define model components
    # the user and item factor can be different
    user_concat_mlp = concat_mlp_model(num_users, factor=factor, layers=layers, emb_reg=emb_reg, reg_layers=reg_layers)
    item_concat_mlp = concat_mlp_model(num_items, factor=factor, layers=layers, emb_reg=emb_reg, reg_layers=reg_layers)
    gmf = gmf_model(input_dim=int(2*layers[-1]), reg=reg_mf)

    # construct model
    user_vector = user_concat_mlp(user_input)
    item_vector = item_concat_mlp(item_input)
    
    prediction = gmf([user_vector, item_vector])

    model = Model(inputs=[user_input, item_input], outputs=prediction)
    return model

if __name__ == '__main__':
    NUM_ITEMS = 1000
    NUM_USERS = 500
    model = concat_gmf_model(NUM_USERS, NUM_USERS, factor=10, layers=[10,10], reg_mf=0, emb_reg=0, reg_layers=[0,0])
   # mlp.compile('rmsprop', 'mse')
    user_input = tf.random_uniform(shape=(100,1), maxval=1000, dtype=tf.int32)
    item_input = tf.random_uniform(shape=(100,1), maxval=1000, dtype=tf.int32)
    output = model([user_input, item_input])
    print(output)
