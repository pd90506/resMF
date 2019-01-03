### This model mimic a resNet style connection between embedding layer and GMF layer
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

from concat_gmf import concat_gmf_model
from time import time
from Dataset import Dataset
from evaluate import evaluate_model
import multiprocessing as mp

# args initiation
class Args(object):
    """A simulator of parser in jupyter notebook"""
    def __init__(self):
        self.path = 'Data/'
        self.dataset = 'ml-1m'
        self.epochs = 30
        self.batch_size = 256
        self.num_factors = 8
        self.layers = [64,32,16,8]
        self.emb_reg = 0
        self.reg_mf = 0
        self.reg_layers = [0,0,0,8]
        self.num_neg = 4
        self.lr = 0.001
        self.learner = 'adam'
        self.verbose = 1
        self.out = 1
        self.mf_pretrain = ''
        self.mlp_pretrain= ''

        self._init_nums()
        

    def _init_nums(self):
        if self.dataset=='ml-1m':
            self.num_users = 6040
            self.num_items = 3706
        elif self.dataset=='ml-100k':
            self.num_users = 671
            self.num_items = 9125

def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [],[],[]
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while ((u,j) in train.keys()):
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

if __name__ == '__main__':
    # init args
    args = Args()

    # load dataset
    topK = 10
    evaluation_threads = 1 #mp.cpu_count()
    print("concat_gmf arguments: %s" %(args))
    #model_out_file = 'Pretrain/%s_resMF_%d_%d.h5' %(args.dataset, num_factors, time())

    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))

    # creating model
    model = concat_gmf_model(args.num_users, args.num_items, factor=args.num_factors, layers=args.layers, reg_mf=args.reg_mf, emb_reg=args.emb_reg, reg_layers=args.reg_layers)
    model.compile(optimizer=Adam(lr=args.lr), loss='binary_crossentropy')

    # initial performance
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (hr, ndcg, time()-t1))    


    # Train model
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in range(args.epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train, args.num_neg)
        
        # Training
        hist = model.fit([np.array(user_input), np.array(item_input)], #input
                         np.array(labels), # labels 
                         batch_size=args.batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time()
        
        # Evaluation
        if epoch %1 == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                #if args.out > 0:
                #    model.save_weights(model_out_file, overwrite=True)

    print("End concat_gmf. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
