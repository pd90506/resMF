import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Input, Multiply, Dense, Flatten, Embedding
from tensorflow.keras.regularizers import l2
from utils import init_normal

def gmf_embed_model(num_users, num_items, factor=10, reg=0):
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')

    embedding_user = Embedding(input_dim = num_users, output_dim = int(factor), name = 'user_embedding',
                                  embeddings_initializer = init_normal(), embeddings_regularizer = l2(reg), input_length=1)  
    embedding_item = Embedding(input_dim = num_items, output_dim = int(factor), name = 'item_embedding',
                                  embeddings_initializer = init_normal(), embeddings_regularizer = l2(reg), input_length=1)  
                        
    embedding_user = Flatten()(embedding_user(user_input))
    embedding_item = Flatten()(embedding_item(item_input))                   

    # Element-wise product of user and item embeddings 
    predict_vector = Multiply()([embedding_user, embedding_item])

    # Prediction layer
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', kernel_regularizer= l2(reg), name = 'prediction')(predict_vector)

    model = Model(inputs=[user_input, item_input], outputs=[prediction])

    return model

if __name__ == '__main__':
    #NUM_ITEMS = 1000
    model = gmf_embed_model(num_users=100, num_items=100)
    # mlp.compile('rmsprop', 'mse')
    user_input = tf.random_uniform(shape=(100,1), maxval=100, dtype=tf.int32)
    item_input = tf.random_uniform(shape=(100,1), maxval=100, dtype=tf.int32)
    output = model([user_input,item_input])
    print(output)