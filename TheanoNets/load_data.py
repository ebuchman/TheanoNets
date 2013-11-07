__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T




def load_data(dataset, target_is_int = False):
    print '... loading the data %s'%dataset
    start_time = time.clock()
    f = open(dataset, 'rb')
    data = cPickle.load(f)
    f.close()    

    train_set, valid_set, test_set = data
            

    def shared_dataset(data_set):
        data_shared = []
	for d in data_set:
		this_shared = theano.shared(np.asarray(d, dtype = theano.config.floatX))
		data_shared.append(this_shared)

 
        if target_is_int:
            data_shared[-1] = T.cast(data_shared[-1], 'int32')

	return data_shared

    for set_ in [train_set, valid_set, test_set]:
	shared_set = shared_dataset(set_)

    test_shared = shared_dataset(test_set)
    valid_shared = shared_dataset(valid_set)
    train_shared = shared_dataset(train_set)
    
    return [train_shared, valid_shared, test_shared]

def load_data2(dataset, target_is_int = False):
    print '... loading the data %s'%dataset
    start_time = time.clock()
    f = open(dataset, 'rb')
    data = cPickle.load(f)
    f.close()    

    train_set, valid_set, test_set = data
            

    def shared_dataset(data_xy):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX))
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX))
                                               
        
        if target_is_int:
            return shared_x, T.cast(shared_y, 'int32')
        else:
            return shared_x, shared_y
        

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval
    

def load_dataSimple(dataset):
    print '... loading the data %s'%dataset
    start_time = time.clock()
    f = open(dataset, 'rb')
    data = cPickle.load(f)
    f.close()    

    train_set, valid_set, test_set = data
            
    end_time = time.clock()
    print '\t took %f seconds' % (end_time-start_time)

    def shared_dataset(data_x):
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX))

        return shared_x


    test_set_x = shared_dataset(test_set)
    valid_set_x = shared_dataset(valid_set)
    train_set_x = shared_dataset(train_set)
    rval = [train_set_x, valid_set_x, test_set_x]
    return rval


def load_test_id(dataset):
    start_time = time.clock()
    f = open(dataset, 'rb')
    data = cPickle.load(f)
    f.close()    

    test_set, id = data
            
    end_time = time.clock()
    print '\t took %f seconds' % (end_time-start_time)

    def shared_dataset(data_x):
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX))

        return shared_x


    test_set = shared_dataset(test_set)

    rval = [test_set, id]
    return rval


