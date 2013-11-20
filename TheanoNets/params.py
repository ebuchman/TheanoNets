import cPickle
import gzip
import os
import sys
import time
import datetime
import numpy as np
import theano
import theano.tensor as T

sys.path.append('src')
from util import get_data_path


ROOT = get_data_path()
LOAD_PARAMS = False
PARAM_PATH = None

def load_convpool_params(rng, img_size, n_in, n_kern, kern_size, n_hid, n_out, pool_ds):
    
    params = []
    new_size = img_size
    
    for i in xrange(len(n_kern)):
        shape = (n_kern[i], n_in, kern_size[i], kern_size[i])
        fan_in = np.prod(shape[1:])		
        fan_out = (shape[0] * np.prod(shape[2:]) / (pool_ds[i]**2 ))
        
        params.append(set_params_random(rng, fan_in, fan_out, shape))
        
        n_in = n_kern[i]
        
        new_size = (new_size - kern_size[i] + 1)/ pool_ds[i]
        
    params.append(set_params_random(rng, n_in*new_size**2, n_hid))
    params.append(set_params_random(rng, n_hid, n_out))
    
    return params
    
def saveModel(model, score, architecture, dir='./'):
    print '...saving model'
    f = open(os.path.join(ROOT, dir, 'model_%s_%f.pkl'%(architecture, score)), 'wb')
    cPickle.dump(model, f)
    f.close() 
    
def saveParams(params, details, score, architecture, dir='./'):
    print '...saving params'
    
    save_to = os.path.join(ROOT, dir, 'params_%s_%f.pkl'%(architecture, score))
    print '\t', save_to

    np_params = []
    for pp in params:
		these_p = []
		for p in pp:
			these_p.append(p.get_value())
		np_params.append(these_p)


    #params = [p.get_value() for p in params[i] for i in xrange(len(params))]
    f = open(save_to, 'wb')
    cPickle.dump([np_params, details], f)
    f.close()
    
    
def load_params(param_file = os.path.join(ROOT,'params_422_5~13_13~9_500H_40O.pkl')):
    print param_file
    f = open(param_file, 'rb')
    params = cPickle.load(f)
    f.close

    return params
    
def set_params_random (rng, fan_in, fan_out, mode = None, shape = None):    
    W_bound = 0.01*np.sqrt(6. / (fan_in + fan_out))
    
    if not shape == None and len(shape) == 4:
        # convpool params
        w_values = np.asarray(0.0001 * rng.uniform(size=(fan_in, fan_out)), 
                    dtype=theano.config.floatX)
        b_values = np.zeros((shape[0],), dtype=theano.config.floatX)
        g_values = np.ones((shape[0],), dtype=theano.config.floatX)
    else:
        # mlp params
        #w_values = np.asarray(0.01 * rng.standard_normal(size=(fan_in, fan_out)), dtype=theano.config.floatX)

        w_values = np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=(fan_in, fan_out)), dtype=theano.config.floatX)
        b_values = np.zeros((fan_out,), dtype = theano.config.floatX)
        #g_values = np.ones((fan_out,), dtype = theano.config.floatX)

    W = theano.shared(w_values, 'W')
    b = theano.shared(b_values, 'b')
    #gain = theano.shared(g_values, 'g')
    
    params = [W,b]

    if mode == 'dA':
        b_prime_values = np.zeros((fan_in,), dtype = theano.config.floatX)
        b_prime = theano.shared(b_prime_values, 'b_prime')
        params.append(b_prime)
    
    return params

def make_param_shared(param):
    if isinstance(param, np.ndarray) or isinstance(param, T.basic.TensorVariable):
        print 'claims to be ndarray or tensor var'

        return theano.shared(np.asarray(param, dtype = theano.config.floatX))
    else:
        print 'clean'
        return param
        
        
        
def make_params_shared(params):
    shared_params = []
    for i in xrange(len(params)):
        shared_params.append(theano.shared(np.asarray(params[i], dtype = theano.config.floatX)))
    
    return shared_params
