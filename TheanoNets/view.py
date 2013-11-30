# simple code for viewing net's reconstruction of trace from context and partial trace
# no tracing

import cPickle, gzip, os, sys, time, datetime

import cmath
import numpy as np
import numpy.random as r
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

sys.path.append('src')

from util import get_data_path, tile_raster_images
from params import *



DATA_PATH = 'tracer_data_expert_test_norm_ext.pkl'

def view_weights(params=None, size=33):
	if params == None:
		params, details = load_params(PARAM_PATH)	
		W = params[0][0]
	else:
		W = params[0][0].get_value()

	img = tile_raster_images(W.T, (size,size), (25,25))

	plt.imshow(img)
	plt.show()

if __name__ == '__main__':
    	view_weights()
    
    
    
    

    
