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

from assemble_net import Tracer 
from util import get_data_path, tile_raster_images
from params import *

SAVE_FOLDER = 150

NUM_CURVES = 1

SEED = 2453
LENGTH = 200
SIZE = 250


CONTEXT_SIZE = 33
TRACE_SIZE = 17
ROOT = get_data_path()
print ROOT

PARAM_PATH = os.path.join('data', 'good_tracer_weights/params_tracer_data_multi_full_33_17_17_nC20_87544_switch0.60_2718_noise_nh_500_100_2000__nout_9_dropout1_LRi_0.10000_reg_0.00_23.725191.pkl')

#'results/tracer_data_expert_test_norm/nh_500_100_2000__nout_9_dropout0/LRi_0.10000_reg_0.00/2013_10_22_16_18/params_tracer_data_expert_test_norm_nh_500_100_2000__nout_9_dropout0_LRi_0.10000_reg_0.00_26.859903.pkl')

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

def view_reconstruction(params=None, details=None, data=None):
    
    print '...loading params'
    if params == None:
	params, details = load_params(PARAM_PATH)
    ''' 
    if LOAD_PARAMS:
        params_all_together = load_params(PARAM_PATH)
        params = [[params_all_together[2*i], params_all_together[2*i+1]] for i in xrange(num_layers)]
    else:
        params = [None]*num_layers
    '''

    rng = np.random.RandomState(12)

    details['mode'] = 'test'

    print '...loading data'
    if data == None:
	    f = open(os.path.join(ROOT, DATA_PATH), 'rb')
	    data = cPickle.load(f)
	    f.close()
    train, valid, test = data
    in1, in2, out, target = test

    print '...building model'
    tracer = Tracer(rng=rng, details=details, params=params)
 
    if details['n_out']:  
    	outputF = theano.function([tracer.x1, tracer.x2], outputs = [tracer.layers[-2].output, tracer.layers[-1].output])
    else:
    	outputF = theano.function([tracer.x1, tracer.x2], outputs = tracer.layers[-1].output)

    for i in xrange(len(in1)):
      if details['n_out']:	
      	recon, pred = outputF(in1[i].reshape(1, 1089), in2[i].reshape(1, 289))
      else:
      	recon = outputF(in1[i].reshape(1, 1089), in2[i].reshape(1, 289))
      
      plt.subplot(411)
      plt.imshow(in1[i].reshape(33,33), cmap='gray')
      plt.subplot(412)
      plt.imshow(in2[i].reshape(17,17), cmap='gray')
      plt.subplot(413)
      plt.imshow(out[i].reshape(17,17), cmap='gray')
      plt.subplot(414)
      plt.imshow(recon.reshape(17,17), cmap='gray')

      if details['n_out']:
     	print pred
      	print np.argmax(pred), target[i]
      
      plt.show()
   

if __name__ == '__main__':
	#view_reconstruction()
    	view_weights()
    
    
    
    

    
