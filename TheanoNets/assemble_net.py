import cPickle
import gzip
import os
import sys
import time
import datetime

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.sandbox.rng_mrg import MRG_RandomStreams as  RandomStreams

from layers import HiddenLayer, OutputLayer
from layer_accessories import make_bad_outs, err_add_ons, cost_add_func, make_params_to_train


class Tracer(object):
    def __init__(self, rng=None, params=None, details = 
					{'n_context' : 33*33, 'n_trace' : 17*17, 'n_h' : [500, 100, 2000], 'n_recon' : 17*17, 'n_out' : 10, 'n_layers' : 7,
					'activations' : ['relu', 'relu', 'relu', ['sigmoid', None], ['softmax', 'argmax']],
					'recon_cost' : 'quadratic', 'pred_cost' : 'nll', 'error_func' : 'neq',
					'cost_add' : [None],
					'dropout' : True,
					'target_is_int' : True,
					'params_to_train' : None,
					'mode' : 'train'}):
	n_context = details['n_context']
	n_trace = details['n_trace']
        n_h = details['n_h']
	n_recon = details['n_recon']
        n_out = details['n_out']
        n_layers = details['n_layers']
        activations = details['activations']
        recon_cost = details['recon_cost']
	pred_cost = details['pred_cost']
        error_func = details['error_func']
        cost_add = details['cost_add']
        dropout = details['dropout']
        target_is_int = details['target_is_int']
        params_to_train = details['params_to_train']
        mode = details['mode']
	
	self.details = details


	self.__name__ = 'tracer'

	if not rng:
	    rng = np.random.RandomState(12345)
	
	if params == None:
 		params = [None]*n_layers
	else:
          while(len(params)<n_layers):
                params.append(None)
	
	if mode == 'test' and dropout == True and not params == None:
		dropout = False
		for i in xrange(len(n_h)):	
			params[i][0] *= 0.5

	str_nh = ''
        for i in xrange(len(n_h)):
          str_nh += str(n_h[i])+'_'
        self.architecture = 'nh_%s_nout_%d_dropout%s'%(str_nh, n_out, int(dropout))


        # allocate symbolic variables (defaults to floatX)
        self.x1 = T.matrix('x1') # context
	self.x2 = T.matrix('x2') # partial trace
        if target_is_int:
            self.y = T.ivector('y') # next pixel
        else:
            self.y = T.matrix('y')
	self.z = T.matrix('z') #reconstruction of trace (full trace)
    
	# make bad outputs
	bad_outputs = make_bad_outs(n_trace, self.x2)
    
        # assemble layers
        self.hidden_layers = []
        
	hidden_layer_1a = HiddenLayer(rng, input = self.x1, n_in = n_context, n_out = n_h[0], dropout = dropout, activation = activations[0], params = params[0])
	hidden_layer_1b = HiddenLayer(rng, input = self.x2, n_in = n_trace, n_out = n_h[1], dropout = dropout, activation = activations[1], params = params[1])

	input_to_h2 = T.concatenate([hidden_layer_1a.output, hidden_layer_1b.output], axis=1)

	hidden_layer_2 = HiddenLayer(rng, input = input_to_h2, n_in = n_h[0]+n_h[1], n_out = n_h[2], dropout = dropout, activation = activations[2], params = params[2])

	recon_layer = OutputLayer(rng, input=hidden_layer_2.output, n_in = n_h[2], n_out = n_recon, non_linearities = activations[3], params=params[3])


	if not n_out == 0:
		pred_layer = OutputLayer(rng, input=recon_layer.output, n_in = n_recon, n_out = n_out, bad_output = bad_outputs, non_linearities = activations[4], params = params[4])

	if n_out == 0:
		self.layers = [hidden_layer_1a, hidden_layer_1b, hidden_layer_2, recon_layer]
	else:	
		self.layers = [hidden_layer_1a, hidden_layer_1b, hidden_layer_2, recon_layer, pred_layer]

	self.params = []
	for layer in self.layers:
		self.params.append(layer.params)

	self.params_to_train = make_params_to_train(self.params, params_to_train)

	self.cost_add, self.dynamic_params = cost_add_func(cost_add, self.layers)

        # cost, error, prediction
        self.error_function = error_func
        self.cost_function = pred_cost
        
	if n_out == 0:
		self.cost = recon_layer.error_functions[recon_cost](self.z) + self.cost_add
		self.error = self.cost
		

	else:
		self.cost = recon_layer.error_functions[recon_cost](self.z) + pred_layer.error_functions[pred_cost](self.y) + self.cost_add
		self.error = pred_layer.error_functions[error_func](self.y)
		self.pred = pred_layer.prediction
		
	# external variables (useful for syncing with the data, ie these are the givens in theano functions in train.py)
	self.in_out = [self.x1, self.x2, self.z, self.y]



class MLP(object):
    def __init__(self, rng=None, params = None, details = 
			    {'n_in' : 2500, 'n_h' : [1000, 100], 'n_out' : 1, 'n_layers' : 3, 
                            'activations' : ['relu', 'relu', ['sigmoid', 'round']], 
                            'cost_func' : 'cross_entropy', 'error_func' : 'neq', 
                            'cost_add' : [None],
			    'dropout' : True, 
                            'target_is_int' : True,
			    'params_to_train' : None,
			    'mode' : 'train'}):

	n_in = details['n_in']
	n_h = details['n_h']
	n_out = details['n_out']
	n_layers = details['n_layers']
	activations = details['activations']
	cost_func = details['cost_func']
	error_func = details['error_func']
	cost_add = details['cost_add']
	dropout = details['dropout']
	target_is_int = details['target_is_int']
	params_to_train = details['params_to_train']
	mode = details['mode']

	self.details = details

	self.__name__ = 'mlp'


	if not rng:
	    rng = np.random.RandomState(12345)

	if params == None:
 		params = [None]*n_layers
	else:
          while(len(params)<n_layers):
                params.append(None)
 		params = [None]*n_layers

	if mode == 'test' and not params == None:
		dropout = False
		for i in xrange(len(n_h)):	
			params[i][0] *= 0.5
              
        str_nh = ''
        for i in xrange(len(n_h)):
          str_nh += str(n_h[i])+'_'
        self.architecture = 'nh_%s_nout_%d_dropout%s'%(str_nh, n_out, int(dropout))

        # allocate symbolic variables (defaults to floatX)
        self.x = T.matrix('x')
        if target_is_int:
            self.y = T.ivector('y')
        else:
            self.y = T.matrix('y')
    
        # assemble layers
        self.hidden_layers = []
        
        this_input = self.x
        this_nin = n_in
        for i in xrange(len(n_h)):
          hidden_layer = HiddenLayer(rng, input = this_input, n_in = this_nin, n_out = n_h[i], activation = activations[i],  dropout = dropout, params = params[i])
          this_input = hidden_layer.output
          this_nin = n_h[i]
          self.hidden_layers.append(hidden_layer)

        self.output_layer = OutputLayer(rng, input = self.hidden_layers[-1].output, n_in = this_nin, n_out = n_out, non_linearities = activations[-1], params = params[-1])
        
        self.layers = self.hidden_layers + [self.output_layer]  

	self.params = []
	for layer in self.layers:
		self.params.append(layer.params)
	print self.params
	self.params_to_train = make_params_to_train(self.params, params_to_train) 

	self.cost_add, self.dynamic_params = cost_add_func(cost_add, self.layers)

        # cost, error, prediction
        self.error_function = error_func
        self.cost_function = cost_func
        
        self.cost = self.output_layer.error_functions[cost_func](self.y) + self.cost_add
        self.error = self.output_layer.error_functions[error_func](self.y)
        self.pred = self.output_layer.prediction
        
        # external variables (useful for syncing with the data)
        self.in_out = [self.x, self.y]
        


# fix this up so its easy to train deep nets...
class AutoEncoder(object):
    def __init__(self, params=None, rng=None, theano_rng=None, 
                            details = { 
                            'n_in' : 2500, 'n_h' : 1000, 'n_layers' : 2,
                            'activations' : ['relu', ['sigmoid', None]], 
                            'cost_func' : 'quadratic', 'error_func' : 'quadratic',
                            'cost_add' : [['sparse1', 0.1], ['J', 0.2]],
                            'dropout' : True, 
                            'denoising' : 0.3,
			    'contractive' : False,
			    'tie_weights' : True,
			    'mode' : 'train'}):
	n_in = details['n_in']
	n_h = details['n_h']
	n_layers = details['n_layers']
	activations = details['activations']
	cost_func = details['cost_func']
	error_func = details['error_func']
	cost_add = details['cost_add']
	dropout = details['dropout']
	denoising = details['denoising']
	tie_weights = details['tie_weights']
	mode = details['mode']

	self.details = details
	self.__name__ = 'autoencoder'

	print details
	if not rng:
	    rng = np.random.RandomState(12345)
        if not theano_rng:                    
	    self.theano_rng = RandomStreams(rng.randint(100))

        
	if params == None:
 		params = [None]*n_layers
	else:
          while(len(params)<n_layers):
                params.append(None)
 		params = [None]*n_layers
	
        self.architecture = 'nh_%s_dropout%s'%(n_h, int(dropout))


	if mode == 'test' and not params == None:
		dropout = False
		params[0][0] *= 0.5


        # allocate symbolic variables (defaults to floatX)
        self.x = T.matrix('x')
        
	if not denoising == None:
            self.corrupted = self.get_corrupted_input(self.x, denoising) 
	else:
	    self.corrupted = self.x*T.ones_like(self.x)
	    	
        # assemble layers
        self.hidden_layer = HiddenLayer(rng, input = self.corrupted, n_in = n_in, n_out = n_h, activation = activations[0],  dropout = dropout, params = params[0])
		
	
	if tie_weights:
		if params[1] == None:
			params[1] = [self.hidden_layer.params[0].T, theano.shared(np.zeros(n_in, dtype=theano.config.floatX))]      
		else:
			params[1] = [self.hidden_layer.params[0].T, params[1][1]]      
	self.output_layer = OutputLayer(rng, input = self.hidden_layer.output, n_in = n_h, n_out = n_in, non_linearities = activations[-1], params = params[1])

        
        self.layers = [self.hidden_layer, self.output_layer]  


	self.params = self.layers[0].params + self.layers[-1].params
	self.params_to_train = make_params_to_train(self.params, params_to_train)
        
	self.cost_add = cost_add_func(cost_add, self.layers)
        
	# cost, error, prediction
        self.error_function = error_func
        self.cost_function = cost_func
        
        self.cost = self.output_layer.error_functions[cost_func](self.x) + self.cost_add
        self.error = self.output_layer.error_functions[error_func](self.x)
        
        # external variables, params
        self.in_out = [self.x]
        
    def get_corrupted_input(self, input, denoising):
	mode, corruption = denoising
	if mode == 'mask':
		return self.theano_rng.binomial(size=input.shape , n=1, p = 1 - corruption, dtype=theano.config.floatX) * input
	elif mode == 'gaussian':
		return self.theano_rng.normal(size=input.shape, avg=0., std=corruption) + input
