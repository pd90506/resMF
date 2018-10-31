import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Input, Multiply, Dense
from tensorflow.keras.regularizers import l2
from utils import init_normal

def gmf_model(input_dim=10, reg=0):
    # Input variables
    user_input = Input(shape=(input_dim,), dtype='float32', name = 'user_input')
    item_input = Input(shape=(input_dim,), dtype='float32', name = 'item_input')

    # Element-wise product of user and item embeddings 
    predict_vector = Multiply()([user_input, item_input])

    # Prediction layer
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', kernel_regularizer= l2(reg), name = 'prediction')(predict_vector)

    model = Model(inputs=[user_input, item_input], outputs=[prediction])

    return model

if __name__ == '__main__':
    #NUM_ITEMS = 1000
    model = gmf_model(input_dim=10)
    # mlp.compile('rmsprop', 'mse')
    user_input = tf.random_uniform(shape=(100,10), dtype=tf.float32)
    item_input = tf.random_uniform(shape=(100,10), dtype=tf.float32)
    output = model([user_input,item_input])
    print(output)