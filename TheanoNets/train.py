#####################################################################################################

#   @'Ethan Buchman' 2012
#
#   most of the code is adapted (but modified heavily!) from deeplearning.net/tutorials
#   dropout code is adapted from github.com/mdenil/dropout
#   learning algorithm code from gwtaylor, and from Hinton's dropout paper (also mdenil)
#####################################################################################################
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



def train_net(data, dataname, model, 
                                n_epochs = 5, 
                                learning_rate = 0.01, learning_rate_decay = 0.95, 
                                mom_i = 0.1, mom_f = 0.99, mom_tau = 200, mom_switch =100, 
                                regularizer = 0, 
				rmsprop = True,
                                nesterov = True,
				batch_size=10,
                                save_many_params = False):
        
  
        architecture = model.architecture
        num_layers = len(model.layers)
        
        learning = 'lri_%f_lr_decay_%f_mom_i_%f_mom_f_%f_mom_tau_%f'%(learning_rate, learning_rate_decay, mom_i, mom_f, mom_tau)
        learning_structure = 'LRi_%.5f_reg_%.2f'%(learning_rate, regularizer)
        momentum = mom_i
        
        now = datetime.datetime.now()
        year, month, day, hour, minute = now.year, now.month, now.day, now.hour, now.minute
        current = '%d_%d_%d_%d_%d'%(year, month, day, hour, minute)
    
        results_dir = os.path.join('results', dataname, architecture, learning_structure, current)
 
	logger = setupLogging(results_dir)

        index = T.lscalar()  # index to a [mini]batch
        l_r, mom, falcon_punch = T.scalars('l_r', 'mom', 'falcon_punch') # learning rate and 

        train_set, valid_set, test_set = data

	theano_rng = RandomStreams(1234)
        
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set[0].get_value(borrow=True).shape[0]
        n_valid_batches = valid_set[0].get_value(borrow=True).shape[0]
        n_test_batches = test_set[0].get_value(borrow=True).shape[0]
        n_train_batches /= batch_size
        n_valid_batches /= batch_size
        n_test_batches /= batch_size

        ######################
        ## THEANO FUNCTIONS ##
        ######################        
        logger.info('... compiling theano functions for backprop, training, testing')
   
	inputs = [index]
	dynamics = []
	current_values = []
	for dp in model.dynamic_params:
		dynamic_param, coef, dynamic = dp 

 		inputs.append(dynamic_param)
		current_values.append(coef)
		dynamics.append(dynamic)

        # create function to test current model error
        test_model = theano.function(inputs, model.error, givens = {model.in_out[i] : test_set[i][index * batch_size: (index + 1) * batch_size] for i in xrange(len(model.in_out))}, on_unused_input = 'ignore')
        validate_model = theano.function(inputs, model.error, givens = {model.in_out[i] : valid_set[i][index * batch_size: (index + 1) * batch_size] for i in xrange(len(model.in_out))}, on_unused_input = 'ignore')
                
        # create a list of all model parameters to be fit by gradient descent
	# note this is a list of lists, with a list for params in each layer
        params = [] 
        for i in xrange(len(model.params_to_train)):
		params += model.params[i]
	
	if rmsprop:
		rmsprops = []
		for i in xrange(len(params)):
			rmsprops.append(theano.shared(np.ones(params[i].get_value().shape, dtype=theano.config.floatX)))	

        
	# create a list of gradients for all model parameters
        #grads = T.grad(dropout_cost if dropout else cost, params)
        grads = T.grad(model.cost, params)

        #mom = ifelse(epoch < self.mom_tau, self.mom_i*(1. - epoch/self.mom_tau) + self.mom_f*(epoch/self.mom_tau), self.mom_f)

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
	
        
        # Create updates dictionary by automatically looping over all
        # (params[i],grads[i]) pairs.
        updates = []
	if rmsprop:
		if not nesterov:
			for param_i, grad_i, prev, rms in zip(params, grads, param_updates, rmsprops):
				rms = 0.9*rms + 0.1 * (grad_i**2)
				grad_i = grad_i / (rms ** 0.5)
				upd = mom * prev - l_r * grad_i
				upd_param = param_i + upd #+ falcon_punch*theano_rng.uniform(size=param_i.shape, low=-1, high=1) 
			  
				#regularize weights for each hidden unit.  Right now its also regularizing output
				if regularizer > 0 and param_i.get_value(borrow=True).ndim==2:
				    squared_norms = T.sum((param_i + upd)**2, axis = 0)
				    scale = T.clip(T.sqrt(regularizer / squared_norms), 0., 1.)
				    upd_param *= scale
				  
				updates.append((param_i, upd_param))
				updates.append((prev, upd))
		else:
			for param_i, grad_i, prev, rms in zip(params, grads, param_updates, rmsprops):
				rms = 0.9*rms + 0.1 * (grad_i**2)
				grad_i = grad_i / (rms ** 0.5)
				upd = - l_r * grad_i
				upd_param = param_i + upd #+ falcon_punch*theano_rng.uniform(size=param_i.shape, low=-1, high=1) 
			  
				#regularize weights for each hidden unit.  Right now its also regularizing output
				if regularizer > 0 and param_i.get_value(borrow=True).ndim==2:
				    squared_norms = T.sum((param_i + upd)**2, axis = 0)
				    scale = T.clip(T.sqrt(regularizer / squared_norms), 0., 1.)
				    upd_param *= scale
				  
				updates.append((param_i, upd_param))
				updates.append((prev, upd + mom*prev))
        # create function to output current model error and update parameters
	else:
		if not nesterov:
			for param_i, grad_i, prev in zip(params, grads, param_updates):
				upd = mom * prev - l_r * grad_i
				upd_param = param_i + upd #+ falcon_punch*theano_rng.uniform(size=param_i.shape, low=-1, high=1) 
			  
				#regularize weights for each hidden unit.  Right now its also regularizing output
				if regularizer > 0 and param_i.get_value(borrow=True).ndim==2:
				    squared_norms = T.sum((param_i + upd)**2, axis = 0)
				    scale = T.clip(T.sqrt(regularizer / squared_norms), 0., 1.)
				    upd_param *= scale
				  
				updates.append((param_i, upd_param))
				updates.append((prev, upd))
		else:
			for param_i, grad_i, prev in zip(params, grads, param_updates):
				upd = - l_r * grad_i
				upd_param = param_i + upd #+ falcon_punch*theano_rng.uniform(size=param_i.shape, low=-1, high=1) 
		 	  
				#regularize weights for each hidden unit.  Right now its also regularizing output
				if regularizer > 0 and param_i.get_value(borrow=True).ndim==2:
				    squared_norms = T.sum((param_i + upd)**2, axis = 0)
				    scale = T.clip(T.sqrt(regularizer / squared_norms), 0., 1.)
				    upd_param *= scale
				  
				updates.append((param_i, upd_param))
				updates.append((prev, upd + mom*prev))
        if nesterov:
		update_momentum = theano.function([mom], [], updates = mom_updates)

	inputs.insert(1, l_r)
	inputs.insert(2, mom)
	print inputs
	train_model = theano.function(inputs, [model.cost, model.error, model.layers[0].output], updates = updates, givens = {model.in_out[i] : train_set[i][index * batch_size: (index + 1) * batch_size] for i in xrange(len(model.in_out))}, on_unused_input = 'ignore')

        ###################
        ### Log Details ### Find better way to implement
        ###################

        def save_run_details():
                logger.debug('Load parameters?  %s\n'%LOAD_PARAMS)
                logger.debug('Architecture:\n')
                logger.debug(architecture)
                #self.logger.debug('Activation Functions: %s'%str(activations))
                logger.debug('Learning Params:\n')
                logger.debug(learning)
                #self.logger.debug('Dropout: %s'%str(dropout))
                #self.logger.debug('L2: %f'%(L2_reg))
                logger.debug('Batch Size: %d\n'%batch_size)
                logger.debug('Dataset: %s\n'%dataname)
        save_run_details()

        ###############
        # TRAIN MODEL #
        ###############
        logger.info('... training')
        # early-stopping parameters
        patience = 10000  # look at this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                           # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
        validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

        best_params = None
        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0.
        start_time = time.clock()

        epoch = 0
        done_looping = False
        best_params = params  
   

        error_function = model.error_function
        cost_function = model.cost_function                                            
                
	last_improved = 0
	lr_orig = learning_rate
        punch = 0                                                                        
        while (epoch < n_epochs): #and (not done_looping):
                epoch = epoch + 1
    
                avg_error = 0
                if last_improved >= 10: 
			print 'falcon punch!'
			#learning_rate = lr_orig
			#punch = 1
			last_improved = 0
		for minibatch_index in xrange(n_train_batches):
                    iter = epoch * n_train_batches + minibatch_index
    		    if nesterov:
			update_momentum(momentum)

		    # these are for dynamic hyperparameters (in the loss function)
		    input_values = []
		    for v in current_values:
			input_values.append(v)

                    cost_ij = train_model(minibatch_index, learning_rate, momentum, *input_values)
		    punch = 0
		    avg_error+=cost_ij[1]
      
                    if (iter + 1) % validation_frequency == 0:

                        # compute zero-one loss on validation set
                        validation_losses = [validate_model(i, *input_values) for i
                                             in xrange(n_valid_batches)]
                        this_validation_loss = np.mean(validation_losses)
                        if not error_function == 'quadratic':
                            this_validation_loss*=100
                        statement = 'epoch %i, minibatch %i/%i, validation error %f' %(epoch, minibatch_index + 1, n_train_batches, this_validation_loss)
                        logger.info(statement)
                        
                        
                        # if we got the best validation score until now
                        if (this_validation_loss < best_validation_loss):
                            last_improved = 0
			    #improve patience if loss improvement is good enough
                            if this_validation_loss < best_validation_loss *  \
                               improvement_threshold:
                                patience = max(patience, iter * patience_increase)

                            # save best validation score and iteration number
                            best_validation_loss = this_validation_loss
                            best_iter = iter
                            
                            best_params = model.params
                            
			    #save params
                            if save_many_params == True:
                                # should remove all other folders in the dir first ...
                                filelist = [ f for f in os.listdir(os.path.join(ROOT,results_dir)) if f.endswith(".pkl")]
				for f in filelist:
                                  os.remove(os.path.join(ROOT, results_dir, f))

                                saveParams(model.params, model.details, best_validation_loss, dataname+'_'+architecture+'_'+learning_structure, dir = results_dir)

                            # test it on the test set
                            test_losses = [test_model(i, *input_values) for i in xrange(n_test_batches)]
                            test_score = np.mean(test_losses)
                            if not error_function == 'quadratic':
                                test_score*=100

                            statement = (('     epoch %i, minibatch %i/%i, test error of best '
                                   'model %f') %
                                  (epoch, minibatch_index + 1, n_train_batches,
                                   test_score))
                            logger.info(statement)
			else:
			    last_improved += 1

                learning_rate *= learning_rate_decay
                if epoch == mom_switch:
                    momentum = mom_f
		
                for i in xrange(len(current_values)):
			current_values[i] = np.cast['float32'](current_values[i] * dynamics[i])
		#if epoch < mom_tau:
                #    momentum = mom_f*epoch/mom_tau +  (1 - epoch/mom_tau)*mom_i
                #self.logger.info('\tlearning rate: %f\tmomentum: %f'%(learning_rate, momentum))
                
                avg_error /= (n_train_batches*batch_size)
                if not error_function == 'quadratic':
                    avg_error*=100

                statement = '\t avg training error over epoch was %f'%(avg_error)
                logger.info(statement)
      

        model.params = best_params
        end_time = time.clock()
	if not save_many_params: 
        	saveParams(best_params, model.details,  best_validation_loss, dataname+'_'+architecture+'_'+learning_structure, dir = results_dir)
        print('Optimization complete.')
        print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f' %
          (best_validation_loss, best_iter, test_score))
        print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

        #close logger
        x = list(logger.handlers)
        for i in x:
            logger.removeHandler(i)
        i.flush()
        i.close()

