
import cPickle
import gzip
import os
import sys
import time
import datetime
import logging

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as  RandomStreams
import matplotlib.pyplot as plt
import random

sys.path.append('src')

from util import get_data_path, setupLogging
from load_data import load_data
from params import *


def save_run_details(architecture, details):
	logger.debug('Load parameters?  %s\n'%LOAD_PARAMS)
	logger.debug('Architecture:\n')
	logger.debug(architecture)
	#self.logger.debug('Activation Functions: %s'%str(activations))
	#logger.debug('Learning Params:\n')
	#logger.debug(learning)
	#self.logger.debug('Dropout: %s'%str(dropout))
	#self.logger.debug('L2: %f'%(L2_reg))
	#logger.debug('Batch Size: %d\n'%batch_size)
	#logger.debug('Dataset: %s\n'%dataname)

def record_keeping_info(model, details):

        architecture = model.architecture
        
        learning = 'lri_%f_lr_decay_%f_mom_i_%f_mom_f_%f_mom_tau_%f'%(details['learning_rate'], details['learning_rate_decay'], details['mom_i'], details['mom_f'], details['mom_tau'])
        learning_structure = 'LRi_%.5f_reg_%.2f'%(details['learning_rate'], details['regularizer'])
        
        now = datetime.datetime.now()
        year, month, day, hour, minute = now.year, now.month, now.day, now.hour, now.minute
        current = '%d_%d_%d_%d_%d'%(year, month, day, hour, minute)
    
        results_dir = os.path.join('results', dataname, architecture, learning_structure, current)

	return architecture, results_dir

def build_model_in_out(model, data, batch_size):

	name = model.__name__
	train_set, valid_set, test_set = data

	if name == 'dtw':
		index1, index2 = T.lscalars('index1', 'index2')
		dtw_indices = model.indices

		model_in_out_test = {model.in_out[0] : test_set[0][index1], model.in_out[1] : test_set[0][index2], model.in_out[2] : test_set[1][index1], model.in_out[3] : test_set[1][index2]  }
		model_in_out_valid = {model.in_out[0] : valid_set[0][index1], model.in_out[1] : valid_set[0][index2], model.in_out[2] : valid_set[1][index1], model.in_out[3] : valid_set[1][index2]  }
		model_in_out_train = {model.in_out[0] : train_set[0][index1], model.in_out[1] : train_set[0][index2], model.in_out[2] : train_set[1][index1], model.in_out[3] : train_set[1][index2]  }
		inputs = [index1, index2, dtw_indices]
	else:
		index = T.lscalar('index')

		model_in_out_test = {model.in_out[i] : test_set[i][index * batch_size: (index + 1) * batch_size] for i in xrange(len(model.in_out))} 
		model_in_out_valid = {model.in_out[i] : valid_set[i][index * batch_size: (index + 1) * batch_size] for i in xrange(len(model.in_out))}
		model_in_out_train = {model.in_out[i] : train_set[i][index * batch_size: (index + 1) * batch_size] for i in xrange(len(model.in_out))}

		inputs = [index]
	model_in_out = model_in_out_train, model_in_out_valid, model_in_out_test

	return inputs, model_in_out


def regularize_weights(regularizer, param_i, upd, upd_param):

	if regularizer > 0 and param_i.get_value(borrow=True).ndim==2:
	    squared_norms = T.sum((param_i + upd)**2, axis = 0)
	    scale = T.clip(T.sqrt(regularizer / squared_norms), 0., 1.)
	    upd_param *= scale

	return upd_param

def learning_updates(model, details, inputs):
	nesterov = details['nesterov']
	rmsprop = details['rmsprop']


	# create a list of all model parameters to be fit by gradient descent
	# note this is a list of lists, with a list for params in each layer
        params = [] 
        for i in xrange(len(model.params_to_train)):
		params += model.params[i]
	
	# list of gradients for all model parameters
        grads = T.grad(model.cost, params)

	# theano variables for input (learning hyperparameters)
        l_r, mom = T.scalars('l_r', 'mom') 

	dynamics = []
	current_values = []
	for dp in model.dynamic_params:
		dynamic_param, coef, dynamic = dp 

 		inputs.append(dynamic_param)
		current_values.append(coef)
		dynamics.append(dynamic)


        #initialize parameter updates for momentum
        param_updates = []
	for i in xrange(len(params)):
		param = params[i]
                init = np.zeros(param.get_value(borrow=True).shape, dtype=theano.config.floatX)
                param_updates.append(theano.shared(init))
                
        if nesterov:
		mom_updates = []
		for param_i, prev in zip(params, param_updates):
			upd = mom*prev
			upd_param = param_i + upd
			mom_updates.append((param_i, upd_param))
	
	if rmsprop:
		rmsprops = []
		for i in xrange(len(params)):
			rmsprops.append(theano.shared(np.ones(params[i].get_value().shape, dtype=theano.config.floatX)))	


	if rmsprop:
		for param_i, grad_i, prev, rms in zip(params, grads, param_updates, rmsprops):
			rms = 0.9*rms + 0.1 * (grad_i**2)
			grad_i = grad_i / (rms ** 0.5)
			if not nesterov:

				upd = mom * prev - l_r * grad_i
				upd_param = param_i + upd 
			  
				upd_param = regularize_weights(details['regularizer'], param_i, upd, upd_param)
				updates.append((param_i, upd_param))
				updates.append((prev, upd))
			else:
				upd = - l_r * grad_i
				upd_param = param_i + upd 
			  
				upd_param = regularize_weights(details['regularizer'], param_i, upd, upd_param) 
				updates.append((param_i, upd_param))
				updates.append((prev, upd + mom*prev))
	else:
		for param_i, grad_i, prev in zip(params, grads, param_updates):
			if not nesterov:
				upd = mom * prev - l_r * grad_i
				upd_param = param_i + upd 
			  
				upd_param = regularize_weights(details['regularizer'], param_i, upd, upd_param)
				updates.append((param_i, upd_param))
				updates.append((prev, upd))
			else:
				upd = - l_r * grad_i
				upd_param = param_i + upd
			  
				upd_param = regularize_weights(details['regularizer'], param_i, upd, upd_param)
				updates.append((param_i, upd_param))
				updates.append((prev, upd + mom*prev))

	return inputs, updates, 

def build_hyper_param_update_function(models, details, hyperparams):

	lr, mom, dp = hyperparams
        
	


	if nesterov:
		update_momentum = theano.function([mom], [], updates = mom_updates)

	def update_hyperparams(learning_rate):
		learning_rate 









