#####################################################################################################
#   @'Ethan Buchman' 2013
#
#####################################################################################################
import cPickle as pickle
import gzip
import os
import sys
import time
import datetime
import logging

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import random

sys.path.append('src')

from assemble_net import Tracer, MLP, AutoEncoder
from run_net_recursively import run_net_recursively
from view import view_reconstruction
from train import train_net
from test_model import Tester, test_autoencoder
from util import get_data_path, setupLogging
from load_data import load_data, load_test_id
from params import *


ROOT = get_data_path() #glob=True if data is in the global datasets directory



# load data and params
dataname = 'tracer_data_multi_full_33_17_17_nC20_3360_switch0.60_2718_noise'
#dataname = 'tracer_data_multi_full_33_17_17_nC20_17424_switch0.60_2718_noise'
#dataname = 'tracer_data_multi_full_33_17_17_nC20_87544_switch0.60_2718_noise'
#dataname = 'tracer_data_expert_test_norm'

try:
	datasets
except:
	dataset = os.path.join(ROOT, dataname+'.pkl')
	datasets = load_data(dataset, target_is_int=True)

# set rng
rng = np.random.RandomState(23455)

# train first for reconstruction, then fix weights and train for classification

# train tracer for reconstruction only:
tracer_details = {
		'n_context' : 33*33, 'n_trace' : 17*17, 'n_h' : [500, 100, 2000], 'n_recon' : 17*17, 'n_out' : 0, 'n_layers' : 4,
		'activations' : ['relu', 'relu', 'relu', ['sigmoid', None], ['softmax', 'argmax']],
		'recon_cost' : 'quadratic', 'pred_cost' : 'nll', 'error_func' : 'neq',
		'cost_add' : [['L2', 0.001, None], ['S', 0.01, None]],
		'dropout' : True,
		'target_is_int' : True,
		'params_to_train' : "all",
		'mode' : 'train',
		'dataname' : dataname} 
	
recon_tracer = Tracer(rng=rng,  details = tracer_details) 



train_net(datasets, dataname, recon_tracer, 
			n_epochs = 50, 
			learning_rate = 0.1, learning_rate_decay = 0.9, 
			mom_i = 0.5, mom_f = 0.9, mom_tau = 200, mom_switch =10, 
			regularizer = 0, 
			rmsprop = True,
			batch_size=100,
			save_many_params =  True)



view_reconstruction(params=recon_tracer.params, details=tracer_details)



# train tracer for classification only, using weights learnt for reconstruction


tracer_details = {
		'n_context' : 33*33, 'n_trace' : 17*17, 'n_h' : [500, 100, 2000], 'n_recon' : 17*17, 'n_out' : 9, 'n_layers' : 5,
		'activations' : ['relu', 'relu', 'relu', ['sigmoid', None], ['softmax', 'argmax']],
		'recon_cost' : None, 'pred_cost' : 'nll', 'error_func' : 'neq',
		'cost_add' : [['L2', 0.0001, [4]]],
		'dropout' : False,
		'target_is_int' : True,
		'params_to_train' : [None, None, None, None, "all"],
		'mode' : 'train',
		'dataname' : dataname} 
	
class_tracer = Tracer(rng=rng,  params = recon_tracer.params, details = tracer_details) 

train_net(datasets, dataname, class_tracer, 
			n_epochs = 20, 
			learning_rate = 0.1, learning_rate_decay = 0.9, 
			mom_i = 0.5, mom_f = 0.9, mom_tau = 200, mom_switch =3, 
			regularizer = 0, 
			rmsprop = True,
			batch_size=20,
			save_many_params =  True)

      
      
run_net_recursively(params=class_tracer.params, details = class_tracer.details)
      
