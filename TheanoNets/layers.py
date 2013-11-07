__docformat__ = 'restructedtext en'

import numpy as np
import theano
import theano.tensor as T

from layer_accessories import convolve, function_options, dropout_from_layer
from theano.tensor.signal import downsample
from params import load_params, set_params_random, make_params_shared, make_param_shared



class ConvPoolLayer(object):
        def __init__(self, rng, input, image_shape, filter_shape, non_linearities, params = None):
		#param filter_shape: (number of filters, num input feature maps,filter height,filter width)
        #param image_shape: (batch size, num input feature maps, image height, image width)

            assert (image_shape[1] == filter_shape[1])
            
            self.input = input
            
            self.W = params[0]
            self.b = params[1]
            
            self.image_shape = image_shape
            self.filter_shape = filter_shape

            # there are "num input feature maps * filter height * filter width" inputs to each hidden unit
            # each unit in the lower layer receives a gradient from: "num output feature maps * filter height * filter width" pooling size
            # fan_in = np.prod(filter_shape[1:])		
            # fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))

            # Non-Linearities
            self.pool_mode = non_linearities[0]
            self.lcn_mode = non_linearities[2]
            
            self.rectifier = function_options(non_linearities[1])
            self.activation = function_options(non_linearities[3])
        
            # convolve input feature maps with filters
            self.conv_out = convolve(input, self.W, self.image_shape, self.filter_shape)
            # outputs (4D) a list of stacks of feature maps, one stack per 'example'
            
            # rectify output
            self.rectified_out = self.rectifier(self.conv_out)
            #(rectify(self.conv_out, self.rectify_mode) if self.rectify_mode is not None else self.conv_out)

            # local contrast normalization
            self.lcn_out = (lcn_2d(self.rectified_out) if self.lcn_mode is True else self.rectified_out)

            # pooling, subsampling
            self.pooled_out = (downsample.max_pool_2d(input=self.lcn_out, ds=(self.pool_mode[2], self.pool_mode[2]), ignore_border=True) if self.pool_mode is not None else self.lcn_out)
            #(pooling(lcn_out, pool_mode) if pool_mode is not None else lcn_out)

            # add the bias (vector) and apply activation function
            self.biased_out = self.pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
            self.output =  self.activation(self.biased_out)
            
            self.params = [self.W, self.b]


class HiddenLayer(object):
        def __init__(self, rng, input, n_in, n_out, activation, dropout = False, params = None):
                if params == None:
                    params = set_params_random(rng, n_in, n_out)

                #else:
                #    self.W = make_param_shared(params[0])
                #    self.b = make_param_shared(params[1])
                self.n_in = n_in
		self.n_out = n_out
 
                self.W = params[0]
                self.b = params[1]
                                                    
                self.input = input
            
		self.act_name = activation    
                self.activation = function_options(activation)                
                
                self.lin_output = T.dot(self.input, self.W) + self.b
                
                self.output =  self.activation(self.lin_output)
                
                if dropout == True:
                  p = 0.5
                  srng = T.shared_randomstreams.RandomStreams(rng.randint(99999))
                  mask = srng.binomial(n=1, p=1-p, size=self.output.shape)
                  self.output *= T.cast(mask, theano.config.floatX)

                #####(lin_output if activation is None else 1.7159*activation((2.0/3.0)*(lin_output)))
                                           
                self.params = [self.W, self.b]

class OutputLayer(object):

    def __init__(self, rng, input, n_in, n_out, non_linearities, bad_output=None, params=None):
        if params == None:
            params = set_params_random(rng, n_in, n_out)

        #else:
        #    print type(params[0]), type(params[1])
        #    self.W = make_param_shared(params[0])
        #   self.b = make_param_shared(params[1])
	self.n_in = n_in
	self.n_out = n_out

        self.W = params[0]
        self.b = params[1]

	self.input = input

        self.activation = function_options(non_linearities[0])
        
        self.predictionF = function_options(non_linearities[1])

        # linear output
        self.lin_output = T.dot(input, self.W) + self.b
    
        # activated output
        self.full_output = self.activation(self.lin_output)
        
        if bad_output:
            # some outputs set to 0 (unavailable choices)
            self.dropout_output = self.full_output * bad_output
        
            # rescaled to give probabilities.  This is equivalent to running the softmax on a smaller lin_output vector
            self.output = self.dropout_output/T.sum(self.dropout_output)
        else:
            self.output = self.full_output
        
        if not non_linearities[1] in (None , 'real'):
          if non_linearities[1] in 'round':
            pred = self.predictionF(self.output)
            self.prediction = T.sum(pred, axis=1) # trick to make it (n,) instead of (n, 1)       
          else:
            self.prediction = self.predictionF(self.output, axis = 1)
        
        self.params = [self.W, self.b]
        
		
        self.error_functions = {None : lambda x:0, 'quadratic': self.quadratic_error, 'cross_entropy': self.cross_entropy, 'nll': self.negative_log_likelihood, 'neq': self.not_equal_error}
        
        
    def quadratic_error(self, y):
        #Return the mean of the sum of the squared differences between output and target values
        squared_difference = (0.5)*(y-self.output)**2
        sum_sq_diff = T.sum(squared_difference, axis=1)
        mean_sqd_difference = T.mean(sum_sq_diff)
        return mean_sqd_difference

    def cross_entropy(self, y):
        eps = np.finfo(np.double).eps
	out = T.sum(self.output, axis=1) # trick to reduce matrix to vector
        return -(y * T.log(out + eps) + (1-y)*T.log(1-out + eps)).mean()
	#return T.nnet.binary_crossentropy(out, y).mean()

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.output)[T.arange(y.shape[0]), y])
        
    def not_equal_error(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch
        """

        # check if y has same dimension of y_pred
        #if y.ndim != self.prediction.ndim:
        #    raise TypeError('y should have the same shape as self.y_pred',
        #        ('y', y.type, 'y_pred', self.prediction.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.prediction, y))
        else:
            raise NotImplementedError()
