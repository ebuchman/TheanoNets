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



class Model(object):
	def __init__(self, rng=None, params=None, details=None):
		# all models must have at least these details

		self.n_layers = details['n_layers']
		self.activations = details['activations']
		self.cost_func = details['cost_func']
		self.error_func = details['error_func']
		self.cost_add = details['cost_add']
		self.dropout = details['dropout']
		self.params_to_train = details['params_to_train']
		self.mode = details['mode']
		
		self.details = details
		
		# model specific details
		self.model_details(details)	

		# name the architecture (useful for saving, refering)
		self.get_architecture()
		
		# all models have an rng
		self.rng = rng
		if not self.rng:
		    self.rng = np.random.RandomState(12345)

		# all models have parameters
		if params == None:
			self.params_init = [None]*self.n_layers
		else:
		  while(len(params)<self.n_layers):
			params.append(None)
		  self.params_init = params


		if self.mode == 'test' and not params == None:
			self.dropout = False
			for i in xrange(len(n_h)):	
				self.params_init[i][0] *= 0.5
		
		# all models have symbolic graph determining their inner structure
		self.build_symbolic_graph()

		# build list of parameters to train.
		self.get_params()

		# get cost, error, prediction, other monitoring signals
		self.get_training()

		# sync with data. in_out has all variables which must be provided in a givens to the theano function 
		self.init_in_out()


	def model_details(self, details):
		self.__name__ = Exception(NotImplemented)
		raise Exception(NotImplemented)

	def get_architecture(self):
		self.architecture = NotImplemented
		raise Exception(NotImplemented)

	def build_symbolic_graph(self):
		self.layers = NotImplemented
		raise Exception(NotImplemented)

	def get_params(self):
		self.params = []
		for layer in self.layers:
			self.params.append(layer.params)
		self.params_to_train = make_params_to_train(self.params, self.params_to_train) 
		
	def get_training(self):
		self.cost_add, self.dynamic_params = cost_add_func(self.cost_add, self.layers)

		# cost, error, prediction
		self.error_function = self.error_func
		self.cost_function = self.cost_func
		
		self.cost = NotImplemented #self.output_layer.error_functions[self.cost_func](self.y) + self.cost_add
		self.error = NotImplemented #self.output_layer.error_functions[self.error_func](self.y)
		self.pred = NotImplemented #self.output_layer.prediction
		raise NotImplemented
	

	def init_in_out(self):
		# external variables (useful for syncing with the data)
		self.in_out = NotImplemented
		raise NotImplemented


	def build_train_in_out(self, data, batch_size):
		train_set, valid_set, test_set = data

		index = T.lscalar('index')

		model_in_out_test = {self.in_out[i] : test_set[i][index * batch_size: (index + 1) * batch_size] for i in xrange(len(self.in_out))} 
		model_in_out_valid = {self.in_out[i] : valid_set[i][index * batch_size: (index + 1) * batch_size] for i in xrange(len(self.in_out))}
		model_in_out_train = {self.in_out[i] : train_set[i][index * batch_size: (index + 1) * batch_size] for i in xrange(len(self.in_out))}

		inputs = [index]
		outputs = [self.cost, self.error]
		

		model_in_out = [model_in_out_train, model_in_out_valid, model_in_out_test]

		return inputs, outputs,  model_in_out






# fix this up so its easy to train deep nets...
class AutoEncoder(Model):
	def __init__(self, params=None, rng=None,  details = { 
	                        'n_in' : 2500, 'n_h' : 1000, 'n_layers' : 2,
	                        'activations' : ['relu', ['sigmoid', None]], 
	                        'cost_func' : 'quadratic', 'error_func' : 'quadratic',
	                        'cost_add' : [['sparse1', 0.1], ['J', 0.2]],
	                        'dropout' : True, 
	                        'denoising' : 0.3,
							'contractive' : False,
							'tie_weights' : True,
							'mode' : 'train'}):
	
		super(AutoEncoder, self).__init__(rng, params, details)

	def model_details(self, details):
		self.n_in = details['n_in']
		self.n_h = details['n_h']
		self.denoising = details['denoising']
		self.tie_weights = details['tie_weights']
		
		self.__name__ = 'autoencoder'
		
		
	def get_architecture(self):
		self.architecture = 'nh_%s_dropout%s'%(self.n_h, int(self.dropout))
	
	
	def build_symbolic_graph(self):
		self.theano_rng = RandomStreams(self.rng.randint(100))	
		
		# allocate symbolic variables (defaults to floatX)
		self.x = T.matrix('x')
		
		if not self.denoising == None:
				self.corrupted = self.get_corrupted_input(self.x, self.denoising) 
		else:
			self.corrupted = self.x*T.ones_like(self.x)
				
		# assemble layers
		self.hidden_layer = HiddenLayer(self.rng, input = self.corrupted, n_in = self.n_in, n_out = self.n_h, activation = self.activations[0],  dropout = self.dropout, params = self.params_init[0])
			
		
		if self.tie_weights:
			if self.params_init[1] == None:
				self.params_init[1] = [self.hidden_layer.params[0].T, theano.shared(np.zeros(self.n_in, dtype=theano.config.floatX), name='b2')]      
			else:
				self.params_init[1] = [self.hidden_layer.params[0].T, self.params_init[1][1]]      
	
		self.output_layer = OutputLayer(self.rng, input = self.hidden_layer.output, n_in = self.n_h, n_out = self.n_in, non_linearities = self.activations[-1], params = self.params_init[1])

			
		self.layers = [self.hidden_layer, self.output_layer]  


	def get_training(self):

		self.cost_add = cost_add_func(self.cost_add, self.layers)
			
		# cost, error, prediction
		self.error_function = self.error_func
		self.cost_function = self.cost_func
		
		self.cost = self.output_layer.error_functions[self.cost_func](self.x)# + self.cost_add
		self.error = self.output_layer.error_functions[self.error_func](self.x)
	
	def init_in_out(self):	
		# external variables, params
		self.in_out = [self.x]
        
	def get_corrupted_input(self, input, denoising):
		mode, corruption = denoising
		if mode == 'mask':
			return self.theano_rng.binomial(size=input.shape , n=1, p = 1 - corruption, dtype=theano.config.floatX) * input
		elif mode == 'gaussian':
			return self.theano_rng.normal(size=input.shape, avg=0., std=corruption) + input




class MLP(Model):
	def __init__(self, rng=None, params = None, details = 
			    {'n_in' : 2500, 'n_h' : [1000, 100], 'n_out' : 1, 'n_layers' : 3, 
                            'activations' : ['relu', 'relu', ['sigmoid', 'round']], 
                            'cost_func' : 'cross_entropy', 'error_func' : 'neq', 
                            'cost_add' : [None],
			    'dropout' : True, 
                            'target_is_int' : True,
			    'params_to_train' : None,
			    'mode' : 'train'}):

		super(MLP, self).__init__(rng, params, details)
		print self.params_init
		print self.params		
    

	def model_details(self, details):
		self.n_in = details['n_in']
		self.n_h = details['n_h']
		self.n_out = details['n_out']
		self.target_is_int = details['target_is_int']
		self.__name__ = 'mlp'

	def get_architecture(self): 
		str_nh = ''
		for i in xrange(len(self.n_h)):
		  str_nh += str(self.n_h[i])+'_'
		self.architecture = 'nh_%s_nout_%d_dropout%s'%(str_nh, self.n_out, int(self.dropout))

	def build_symbolic_graph(self):
		# allocate symbolic variables (defaults to floatX)
		self.x = T.matrix('x')
		if self.target_is_int:
			self.y = T.ivector('y')
		else:
			self.y = T.matrix('y')
		
		# assemble layers
		self.hidden_layers = []
		
		this_input = self.x
		this_nin = self.n_in
		for i in xrange(len(self.n_h)):
		  hidden_layer = HiddenLayer(self.rng, input = this_input, n_in = this_nin, n_out = self.n_h[i], activation = self.activations[i],  dropout = self.dropout, params = self.params_init[i])
		  this_input = hidden_layer.output
		  this_nin = self.n_h[i]
		  self.hidden_layers.append(hidden_layer)

		self.output_layer = OutputLayer(self.rng, input = self.hidden_layers[-1].output, n_in = this_nin, n_out = self.n_out, non_linearities = self.activations[-1], params = self.params_init[-1])
		
		self.layers = self.hidden_layers + [self.output_layer]  

		
	def get_training(self):
		self.cost_add, self.dynamic_params = cost_add_func(self.cost_add, self.layers)

		# cost, error, prediction
		self.error_function = self.error_func
		self.cost_function = self.cost_func
		
		self.cost = self.output_layer.error_functions[self.cost_func](self.y) + self.cost_add
		self.error = self.output_layer.error_functions[self.error_func](self.y)
		self.pred = self.output_layer.prediction

	def init_in_out(self):
		# external variables (useful for syncing with the data)
		self.in_out = [self.x, self.y]



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

		super(Tracer, self).__init__(rng, params, details)

	def model_details(self):
		self.n_context = details['n_context']
		self.n_trace = details['n_trace']
		self.n_h = details['n_h']
		self.n_recon = details['n_recon']
		self.n_out = details['n_out']
		self.recon_cost = details['recon_cost']
		self.pred_cost = details['pred_cost']	
		self.target_is_int = details['target_is_int']
		self.__name__ = 'tracer'

	def get_architecture(self):
		str_nh = ''
		for i in xrange(len(self.n_h)):
			str_nh += str(self.n_h[i])+'_'
		self.architecture = 'nh_%s_nout_%d_dropout%s'%(str_nh, self.n_out, int(self.dropout))

	def build_symbolic_graph(self):
		# allocate symbolic variables (defaults to floatX)
		self.x1 = T.matrix('x1') # context
		self.x2 = T.matrix('x2') # partial trace
		if self.target_is_int:
		    self.y = T.ivector('y') # next pixel
		else:
		    self.y = T.matrix('y')
		self.z = T.matrix('z') #reconstruction of trace (full trace)
		
		# make bad outputs
		bad_outputs = make_bad_outs(self.n_trace, self.x2)
		
		# assemble layers
		self.hidden_layers = []
		
		hidden_layer_1a = HiddenLayer(self.rng, input = self.x1, n_in = self.n_context, n_out = self.n_h[0], dropout = self.dropout, activation = self.activations[0], params = self.params_init[0])
		hidden_layer_1b = HiddenLayer(self.rng, input = self.x2, n_in = self.n_trace, n_out = self.n_h[1], dropout = self.dropout, activation = self.activations[1], params = self.params_init[1])
		
		input_to_h2 = T.concatenate([hidden_layer_1a.output, hidden_layer_1b.output], axis=1)
		
		hidden_layer_2 = HiddenLayer(self.rng, input = input_to_h2, n_in = self.n_h[0]+self.n_h[1], n_out = self.n_h[2], dropout = self.dropout, activation = self.activations[2], params = self.params_init[2])
		
		recon_layer = OutputLayer(self.rng, input=hidden_layer_2.output, n_in = self.n_h[2], n_out = self.n_recon, non_linearities = self.activations[3], params=params_init[3])
		
		
		if not self.n_out == 0:
			pred_layer = OutputLayer(self.rng, input=recon_layer.output, n_in = self.n_recon, n_out = self.n_out, bad_output = bad_outputs, non_linearities = self.activations[4], params = self.params_init[4])
		
		
		
		if self.n_out == 0:
			self.layers = [hidden_layer_1a, hidden_layer_1b, hidden_layer_2, recon_layer]
		else:	
			self.layers = [hidden_layer_1a, hidden_layer_1b, hidden_layer_2, recon_layer, pred_layer]
		
		
		

	def get_training(self):

		self.cost_add, self.dynamic_params = cost_add_func(cost_add, self.layers)
		
		# cost, error, prediction
		self.error_function = error_func
		self.cost_function = pred_cost
		
		if self.n_out == 0:
			self.cost = recon_layer.error_functions[self.recon_cost](self.z) + self.cost_add
			self.error = self.cost
		else:
			self.cost = recon_layer.error_functions[self.recon_cost](self.z) + pred_layer.error_functions[self.pred_cost](self.y) + self.cost_add
			self.error = pred_layer.error_functions[self.error_func](self.y)
			self.pred = pred_layer.prediction
		
	def init_in_out(self):		
		# external variables (useful for syncing with the data, ie these are the givens in theano functions in train.py)
		self.in_out = [self.x1, self.x2, self.z, self.y]


