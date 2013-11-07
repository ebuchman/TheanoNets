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

from util import get_data_path, setupLogging, tile_raster_images
from load_data import load_data
from params import *

from assemble_net import AutoEncoder, MLP

class Tester():
  def __init__(self):
  
    self.PARAM_PATH = 'data/results/evergreen_txt/nh_100_dropout1/LRi_0.01000_reg_0.00/2013_10_6_0_16/params_evergreen_txt_nh_100_dropout1_LRi_0.01000_reg_0.00_1209.485394.pkl'
  
  def test_net(self, data, params, details):
    test_set, id = data

    details['mode'] = 'test'
    model = MLP(params=params, details = details)
    test_model = theano.function([], model.pred, 
        givens = { model.x: test_set})
    
    preds = test_model()
    
    results = np.zeros((len(preds),), dtype=('i4, i4'))

    results[:] = zip(np.cast['int'](id), np.cast['int'](preds))
    #results[:, 0] = np.cast[np.int8](id)
    #results[:, 1] = np.cast[np.int8](preds)
    print results.dtype
    results = np.sort(results.view('i4, i4'), order=['f0'], axis=0)
    print results

    np.savetxt('submission.csv', results, fmt='%d', delimiter=',')




def test_autoencoder(data, params, details):
	details['mode'] = 'test'	 

	model = AutoEncoder(params=params, details = details)
  
  	test_model = theano.function([], [model.layers[-1].output, model.layers[0].output, model.corrupted, model.layers[0].params[0]],  givens={model.x: data})
  	recon, hidden, cor, w = test_model()
	
  	data = data.get_value()

	hidden_l = details['n_h']
	widths = [10]
	for ww in widths:
		if hidden_l % ww == 0:
			hidden_shape = (ww, hidden_l/ww)
			break

	width = int(np.sqrt(len(recon[0])))

  	for i in xrange(100):
	    h = hidden[i]

            active_unit_indices = [j for j,v in enumerate(h >= 0.1) if v]
	    active_features = w[:, active_unit_indices].T
	    n_units = len(active_unit_indices)
	    tile_l = int(np.sqrt(n_units)) + 1
	    features = tile_raster_images(active_features, (width,width), (tile_l, tile_l))   
	    print width, type(cor[i]), len(cor[i])
	    print cor[i].shape	    
            plt.figure()
	    plt.subplot(311)
	    plt.imshow(cor[i].reshape(width,width))
	    plt.subplot(312)
	    plt.imshow(hidden[i].reshape(hidden_shape))
	    plt.subplot(313)
	    plt.imshow(recon[i].reshape(width,width))
	    
	    plt.figure()
	    plt.imshow(features)
	    plt.show()


def make_params_shared(params):
	shared_params = []
	for p in params:
		p = theano.shared(np.asarray(p, dtype=theano.config.floatX))
		shared_params.append(p)
	return shared_params#[[shared_params[0], shared_params[1]], [shared_params[0].T, shared_params[2]]]

if __name__ == '__main__':

	ROOT = get_data_path()
	param_file = os.path.join('data', 'results', 'tracer_data_expert_test/nh_2000_dropout1/LRi_0.00100_reg_0.00/2013_10_11_18_50/params_tracer_data_expert_test_nh_2000_dropout1_LRi_0.00100_reg_0.00_73.450010.pkl')
	data_file = os.path.join(ROOT, 'tracer_data_expert_test.pkl')

	f = open(param_file, 'rb')
	params, details = pickle.load(f)
	f.close()
	train, valid, test = load_data(data_file, True)

	details['mode'] = 'test'	
	params = make_params_shared(params)
	test_set_x = test[0]
	
	test_autoencoder(test_set_x, params, details)





