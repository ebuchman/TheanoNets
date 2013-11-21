
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


def save_run_details(architecture, details, logger):
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

def record_keeping_info(model, details, dataname):

        architecture = model.architecture
        
        learning = 'lri_%f_lr_decay_%f_mom_i_%f_mom_f_%f_mom_tau_%f'%(details['learning_rate'], details['learning_rate_decay'], details['mom_i'], details['mom_f'], details['mom_tau'])
        learning_structure = 'LRi_%.5f_reg_%.2f'%(details['learning_rate'], details['regularizer'])
        
        now = datetime.datetime.now()
        year, month, day, hour, minute = now.year, now.month, now.day, now.hour, now.minute
        current = '%d_%d_%d_%d_%d'%(year, month, day, hour, minute)
    
        results_dir = os.path.join('results', dataname, architecture, learning_structure, current)

	return architecture, results_dir, learning_structure


def regularize_weights(regularizer, param_i, upd, upd_param):

	if regularizer > 0 and param_i.get_value(borrow=True).ndim==2:
	    squared_norms = T.sum((param_i + upd)**2, axis = 0)
	    scale = T.clip(T.sqrt(regularizer / squared_norms), 0., 1.)
	    upd_param *= scale

	return upd_param



def learning_updates(model, details, inputs):


	# create a list of all model parameters to be fit by gradient descent
	params = []

	for i in xrange(len(model.params_to_train)):
		for j in model.params_to_train[i]:	
			params.append(model.params[i][j])

	updates = []

	# list of gradients for all model parameters
	grads = T.grad(model.cost, params)

	# theano variables for input (learning hyperparameters)
	l_r, mom = T.scalars('l_r', 'mom') 


	#initialize parameter updates for momentum
	param_updates = []
	for i in xrange(len(params)):
		param = params[i]
		init = np.zeros(param.get_value(borrow=True).shape, dtype=theano.config.floatX)
		param_updates.append(theano.shared(init))
					
	# build updates pairs
	for param_i, grad_i, prev in zip(params, grads, param_updates):
		upd = mom * prev - l_r * grad_i
		upd_param = param_i + upd 
	  
		upd_param = regularize_weights(details['regularizer'], param_i, upd, upd_param)
		updates.append((param_i, upd_param))


	inputs += [l_r, mom]

	
	return inputs, params, updates 


def learning_updates2(model, details, inputs):
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









