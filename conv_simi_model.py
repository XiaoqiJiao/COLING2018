from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse

import theano
import numpy as np
import lasagne
import cPickle
import sys
import time
from theano import tensor as T
from theano import config

from lasagne_mask_layer import lasagne_mask_layer
from lasagne_sum_pooling_layer import lasagne_sum_pooling_layer

def check_quarter(idx, n):
    if idx == round(n / 4.) or idx == round(n / 2.) or idx == round(3 * n / 4.):
        return True
    return False

class conv_simi_model(object):

    def prepare_data(self, list_of_seqs):
        lengths = [len(s) for s in list_of_seqs]
        n_samples = len(list_of_seqs)
        maxlen = np.max(lengths)
        x = np.zeros((n_samples, maxlen)).astype('int32')
        x_mask = np.zeros((n_samples, maxlen)).astype(theano.config.floatX)
        for idx, s in enumerate(list_of_seqs):
            x[idx, :lengths[idx]] = s
            x_mask[idx, :lengths[idx]] = 1.
        x_mask = np.asarray(x_mask, dtype=config.floatX)
        return x, x_mask
    
    def get_minibatches_idx(self, n, minibatch_size, shuffle=False):
        idx_list = np.arange(n, dtype="int32")

        if shuffle:
            np.random.shuffle(idx_list)

        minibatches = []
        minibatch_start = 0
        for i in range(n // minibatch_size):
            minibatches.append(idx_list[minibatch_start:
            minibatch_start + minibatch_size])
            minibatch_start += minibatch_size

        if (minibatch_start != n):
            # Make a minibatch out of what is left
            minibatches.append(idx_list[minibatch_start:])

        return zip(range(len(minibatches)), minibatches)

    def getDataSim(self, batch, nout=5):
        g1 = []
        g2 = []
        for i in batch:
            g1.append(i[0].embeddings)
            g2.append(i[1].embeddings)
            
        g1x, g1mask = self.prepare_data(g1)
        g2x, g2mask = self.prepare_data(g2)

        scores = []
        for i in batch:
            temp = np.zeros(nout)
            score = float(i[2])
            ceil, fl = int(np.ceil(score)), int(np.floor(score))
            if ceil == fl:
                temp[fl - 1] = 1
            else:
                temp[fl - 1] = ceil - score
                temp[ceil - 1] = score - fl
            scores.append(temp)
        scores = np.matrix(scores) + 0.000001
        scores = np.asarray(scores, dtype=config.floatX)
        return (scores, g1x, g1mask, g2x, g2mask)
    
    def getEvalData(self, batch):
        g1 = []
        g2 = []
        golds = []
        for i in batch:
            g1.append(i[0].embeddings)
            g2.append(i[1].embeddings)
            golds.append(float(i[2]))
        g1x, g1mask = self.prepare_data(g1)
        g2x, g2mask = self.prepare_data(g2)

        return (golds, g1x, g1mask, g2x, g2mask)
    
    def evaluate_sick(self, evl_data, words, batchsize=64):
        kf = self.get_minibatches_idx(len(evl_data), batchsize, shuffle=False)
        
        full_golds = []
        full_preds = []
        
        for _, index in kf:
            batch = [evl_data[t] for t in index]
            for i in batch:
                i[0].populate_embeddings(words)
                i[1].populate_embeddings(words)
                
            (golds, g1x, g1mask, g2x, g2mask) = self.getEvalData(batch)
        
            scores = self.scoring_function(g1x,g2x,g1mask,g2mask)        
            preds = np.squeeze(scores)
            
            full_golds.extend(golds)
            full_preds.extend(list(preds))
            
            for i in batch:
                i[0].representation = None
                i[1].representation = None
                i[0].unpopulate_embeddings()
                i[1].unpopulate_embeddings()
                
        return pearsonr(full_preds,full_golds)[0],spearmanr(full_preds,full_golds)[0],mse(full_preds, full_golds)
 
    
    def __init__(self, We_initial, params):

        #initial_We = theano.shared(np.asarray(We_initial, dtype=config.floatX))
        We = theano.shared(np.asarray(We_initial, dtype=config.floatX))

        g1batchindices = T.imatrix(); g2batchindices = T.imatrix();
        g1mask = T.matrix(); g2mask = T.matrix();
        scores = T.matrix()

        l_in = lasagne.layers.InputLayer((None, None))
        l_mask = lasagne.layers.InputLayer(shape=(None, None))
        l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=We.get_value().shape[0],
                                              output_size=We.get_value().shape[1], W=We)
                 
        l_masked_emb = lasagne_mask_layer([l_emb, l_mask])
        l_reshape = lasagne.layers.ReshapeLayer(l_masked_emb, ([0], 1, [1], [2]))
            
        l_conv1 = lasagne.layers.Conv2DLayer(l_reshape, 
                                             num_filters=params.dim/3, 
                                             filter_size=(1, 300), 
                                             nonlinearity=lasagne.nonlinearities.linear)
        l_conv1_mean = lasagne_sum_pooling_layer(l_conv1)
        l_conv1_out = lasagne.layers.ReshapeLayer(l_conv1_mean, ([0], [1]))
            
        l_conv2 = lasagne.layers.Conv2DLayer(l_reshape, 
                                             num_filters=params.dim/3, 
                                             filter_size=(2, 300), 
                                             nonlinearity=lasagne.nonlinearities.linear)
        l_conv2_mean = lasagne_sum_pooling_layer(l_conv2)
        l_conv2_out = lasagne.layers.ReshapeLayer(l_conv2_mean, ([0], [1]))
            
        l_conv3 = lasagne.layers.Conv2DLayer(l_reshape, 
                                             num_filters=params.dim/3, 
                                             filter_size=(3, 300),
                                             nonlinearity=lasagne.nonlinearities.linear)
        l_conv3_mean = lasagne_sum_pooling_layer(l_conv3)
        l_conv3_out = lasagne.layers.ReshapeLayer(l_conv3_mean, ([0], [1]))
            
        l_out = lasagne.layers.ConcatLayer([l_conv1_out, l_conv2_out, l_conv3_out], axis=1)

        self.final_layer = l_out

        embg1 = lasagne.layers.get_output(l_out, {l_in: g1batchindices, l_mask: g1mask})
        embg2 = lasagne.layers.get_output(l_out, {l_in: g2batchindices, l_mask: g2mask})
        
        def fix(x):
            return x*(x > 0) + 1E-10*(x <= 0)
        
        embg1 = embg1/T.sqrt(fix(T.sum(embg1 ** 2, axis=1, keepdims=True)))
        embg2 = embg2/T.sqrt(fix(T.sum(embg2 ** 2, axis=1, keepdims=True)))
        
        g1_dot_g2 = embg1*embg2
        g1_abs_g2 = abs(embg1-embg2)
        
        lin_dot = lasagne.layers.InputLayer((None, We.get_value().shape[1]))
        lin_abs = lasagne.layers.InputLayer((None, We.get_value().shape[1]))
        
        l_sum = lasagne.layers.ConcatLayer([lin_dot, lin_abs])
        #50 is the hidden layer size, we can tune, but in here for the whole frame.
        l_sigmoid = lasagne.layers.DenseLayer(l_sum, params.memsize, nonlinearity=lasagne.nonlinearities.sigmoid)
        #5 is the last layer size
        l_softmax = lasagne.layers.DenseLayer(l_sigmoid, 5, nonlinearity=T.nnet.softmax)
        X = lasagne.layers.get_output(l_softmax, {lin_dot:g1_dot_g2, lin_abs:g1_abs_g2})
        Y = T.log(X)

        cost = scores*(T.log(scores) - Y)
        cost = cost.sum(axis=1)/(float(5.0))
        
        prediction = 0.
        i = 1                                   #minval : 1
        while i<= 5:                            #maxval : 5
            prediction = prediction + i*X[:,i-1]
            i += 1
        
        ref_params = cPickle.load(open(params.loadmodel, 'rb'))
        lasagne.layers.set_all_param_values(self.final_layer, ref_params)
        
        tmp = lasagne.layers.get_all_params(l_out, trainable=True)
        
        reg = 0.5*params.LW*lasagne.regularization.l2(We-ref_params[0])
        reg += 0.5*params.LC*( lasagne.regularization.l2(tmp[1]-ref_params[1]) )
        reg += 0.5*params.LC*( lasagne.regularization.l2(tmp[2]-ref_params[2]) )
        reg += 0.5*params.LC*( lasagne.regularization.l2(tmp[3]-ref_params[3]) )
        reg += 0.5*params.LC*( lasagne.regularization.l2(tmp[4]-ref_params[4]) )
        reg += 0.5*params.LC*( lasagne.regularization.l2(tmp[5]-ref_params[5]) )
        reg += 0.5*params.LC*( lasagne.regularization.l2(tmp[6]-ref_params[6]) )
        '''
        tmp = lasagne.layers.get_all_params(l_out, trainable=True)
        
        reg = 0.5*params.LW*lasagne.regularization.l2(We-initial_We)
        reg += 0.5*params.LC*( lasagne.regularization.l2(tmp[1]) )
        reg += 0.5*params.LC*( lasagne.regularization.l2(tmp[2]) )
        reg += 0.5*params.LC*( lasagne.regularization.l2(tmp[3]) )
        reg += 0.5*params.LC*( lasagne.regularization.l2(tmp[4]) )
        reg += 0.5*params.LC*( lasagne.regularization.l2(tmp[5]) )
        reg += 0.5*params.LC*( lasagne.regularization.l2(tmp[6]) )
        '''
        
        tmp = lasagne.layers.get_all_params(l_softmax, trainable=True)
        reg += 0.5*params.L2*sum(lasagne.regularization.l2(x) for x in tmp)
        
        cost = T.mean(cost) + reg
        
        self.all_params = lasagne.layers.get_all_params(l_out, trainable=True) + lasagne.layers.get_all_params(l_softmax, trainable=True)

        self.feedforward_function = theano.function([g1batchindices,g1mask], embg1)
        
        self.scoring_function = theano.function([g1batchindices, g2batchindices, g1mask, g2mask], prediction)
        
        self.cost_function = theano.function([scores, g1batchindices, g2batchindices, g1mask, g2mask], cost)
        
        grads = theano.gradient.grad(cost, self.all_params)
        updates = params.learner(grads, self.all_params, params.eta)

        self.train_function = theano.function([scores, g1batchindices, g2batchindices,
                             g1mask, g2mask], cost, updates=updates)
                
        print "Num Params:", lasagne.layers.count_params(self.final_layer) + lasagne.layers.count_params(l_softmax)

    def train(self, data, words, params):

        start_time = time.time()
        
        train_data, dev_data, test_data = data[0], data[1], data[2]
        
        pearsonr,spearmanr,mse = self.evaluate_sick(train_data, words)
        print 'train pearsonr: ', pearsonr, 'train spearmanr: ', spearmanr , 'train mse: ', mse       
        pearsonr,spearmanr,mse = self.evaluate_sick(dev_data, words)
        print 'dev pearsonr: ', pearsonr, 'dev spearmanr: ', spearmanr, 'dev mse: ', mse        
        pearsonr,spearmanr,mse = self.evaluate_sick(test_data, words)
        print 'test pearsonr: ', pearsonr, 'test spearmanr: ', spearmanr, 'test mse: ', mse                        
        print '\n'
        
        try:
            for eidx in xrange(params.epochs):
                kf = self.get_minibatches_idx(len(train_data), params.batchsize, shuffle=True)
                uidx = 0
                for _, train_index in kf:

                    uidx += 1
                    batch = [train_data[t] for t in train_index]

                    for i in batch:
                        i[0].populate_embeddings(words)
                        i[1].populate_embeddings(words)

                    (scores, g1x, g1mask, g2x, g2mask) = self.getDataSim(batch)
                    cost = self.train_function(scores, g1x, g2x, g1mask, g2mask)

                    if np.isnan(cost) or np.isinf(cost):
                        print 'NaN detected. Exiting.'
                        sys.exit(0)

                    if (check_quarter(uidx, len(kf))):
                        print 'Eidx:', eidx, '    Uidx:', uidx, '    Cost:', cost
                        pearsonr,spearmanr,mse = self.evaluate_sick(train_data, words)
                        print 'train pearsonr: ', pearsonr, 'train spearmanr: ', spearmanr , 'train mse: ', mse       
                        pearsonr,spearmanr,mse = self.evaluate_sick(dev_data, words)
                        print 'dev pearsonr: ', pearsonr, 'dev spearmanr: ', spearmanr, 'dev mse: ', mse        
                        pearsonr,spearmanr,mse = self.evaluate_sick(test_data, words)
                        print 'test pearsonr: ', pearsonr, 'test spearmanr: ', spearmanr, 'test mse: ', mse                    
                        print '\n'
                    #undo batch to save RAM
                    for i in batch:
                        i[0].representation = None
                        i[1].representation = None
                        i[0].unpopulate_embeddings()
                        i[1].unpopulate_embeddings()
                    
                print 'Eidx:', eidx, '    Uidx:', uidx, '    Cost:', cost
                pearsonr,spearmanr,mse = self.evaluate_sick(train_data, words)
                print 'train pearsonr: ', pearsonr, 'train spearmanr: ', spearmanr , 'train mse: ', mse       
                pearsonr,spearmanr,mse = self.evaluate_sick(dev_data, words)
                print 'dev pearsonr: ', pearsonr, 'dev spearmanr: ', spearmanr, 'dev mse: ', mse        
                pearsonr,spearmanr,mse = self.evaluate_sick(test_data, words)
                print 'test pearsonr: ', pearsonr, 'test spearmanr: ', spearmanr, 'test mse: ', mse                        
                print '\n'
        except KeyboardInterrupt:
            print "Training interupted"

        end_time = time.time()
        print "total time:", (end_time - start_time)
