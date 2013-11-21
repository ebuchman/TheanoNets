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

from train_accessories import record_keeping_info, save_run_details, learning_updates
from util import get_data_path, setupLogging
from load_data import load_data
from params import *



def train_net(data, dataname, model, details = { 
                                'n_epochs' : 5, 
                                'learning_rate' : 0.01, 'learning_rate_decay' : 0.95, 
                                'mom_i' : 0.1, 'mom_f' : 0.99, 'mom_tau' : 200, 'mom_switch' : 100, 
                                'regularizer' : 0, 
								'rmsprop' : True,
                                'nesterov' : True,
								'batch_size' : 10,
                                'save_many_params' : False}):
	
	# this mostly shouldnt be necessary
	# parameters for training
	n_epochs = details['n_epochs']
	learning_rate = details['learning_rate']
	learning_rate_decay = details['learning_rate_decay']
	momentum = details['mom_i']
	mom_f = details['mom_f']
	mom_switch = details['mom_switch']
	batch_size = details['batch_size']
	save_many_params = details['save_many_params']

	###########################
	## 	   TO DO 	 ##
	###########################

	# integrate in hyperparams as shared variables and give them updates!
	# improve 'learning_updates' to restore all that golden functionality...
	# reimplemetn validation/test scheme

        



	##########################
	##  record keeping info ##
	##########################      
	architecture, results_dir, learning_structure = record_keeping_info(model, details, dataname)
 
	num_layers = len(model.layers)

	logger = setupLogging(results_dir)

	##########
	## DATA	##
	##########
	train_set, valid_set, test_set = data

	# compute number of minibatches for training, validation and testing
	batch_size = details['batch_size']
	n_train_batches = train_set[0].get_value(borrow=True).shape[0] / batch_size
	n_valid_batches = valid_set[0].get_value(borrow=True).shape[0] / batch_size
	n_test_batches = test_set[0].get_value(borrow=True).shape[0] /batch_size

	###########################
	## THEANO Symbolic Graph ##
	###########################        
	logger.info('... compiling theano functions for backprop, training, testing')

	theano_rng = RandomStreams(1234)

	# these are provided as givens to theano functions
	# they allow seemless integration and a single train function for all models
	inputs, outputs, model_in_out = model.build_train_in_out(data, batch_size)

	model_in_out_train, model_in_out_valid, model_in_out_test = model_in_out

	# Create updates dictionary accordnig to learning parameters
	inputs, params, updates = learning_updates(model, details, inputs)


	######################
	## Theano Functions ##
	######################

	train_model = theano.function(inputs, outputs, updates = updates, givens = model_in_out_train, on_unused_input = 'ignore') 

	test_model = theano.function(inputs, model.error, givens = model_in_out_test, on_unused_input = 'ignore') 

	validate_model = theano.function(inputs, model.error, givens = model_in_out_valid, on_unused_input = 'ignore') 

	###################
	### Log Details ### Find better way to implement
	###################
	# this function is a mess right now and maybe unnecessary due to details dictionary
	save_run_details(architecture, details, logger)


	###############
	# TRAIN MODEL #
	###############
	logger.info('... training')

	# early-stopping parameters - gah!
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

	error_function = model.error_function
	cost_function = model.cost_function                                            
                
	last_improved = 0
	lr_orig = details['learning_rate']
	
	while (epoch < n_epochs): #and (not done_looping):
		epoch = epoch + 1

		avg_error = 0

	
		for minibatch_index in xrange(n_train_batches):
			iter = epoch * n_train_batches + minibatch_index
		    
			input_values = [minibatch_index]
		    
			if model.__name__ == 'dtw':
				batch_index2 = np.random.randint(n_train_batches)
				input_values.append(batch_index2)
				ind = []
				for i in xrange(128):
					for j in xrange(np.maximum(0, i-2), np.minimum(i+2, 128)):
						ind.append([i,j])
				ind = np.asarray(ind, dtype = 'int32')
				
				input_values.append(ind)		    

			input_values.append(learning_rate)
			input_values.append(momentum)

			cost_ij = train_model(*input_values)

			avg_error+=cost_ij[1]
			''' 
                    if (iter + 1) % validation_frequency == 0:

                        # compute zero-one loss on validation set
                        validation_losses = [validate_model(i, batch_index2) for i
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
			'''
		learning_rate *= learning_rate_decay
		if epoch == mom_switch:
			momentum = mom_f

		'''	
		for i in xrange(len(current_values)):
	current_values[i] = np.cast['float32'](current_values[i] * dynamics[i])
		'''
		#if epoch < mom_tau:
			#    momentum = mom_f*epoch/mom_tau +  (1 - epoch/mom_tau)*mom_i
			#self.logger.info('\tlearning rate: %f\tmomentum: %f'%(learning_rate, momentum))
			
		avg_error /= (n_train_batches*batch_size)
		if not error_function == 'quadratic':
			avg_error*=100

		statement = '\t avg training error over epoch was %f'%(avg_error)
		logger.info(statement)
  

		end_time = time.clock()
		if not save_many_params: 
			saveParams(model.params, model.details,  best_validation_loss, dataname+'_'+architecture+'_'+learning_structure, dir = results_dir)
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

