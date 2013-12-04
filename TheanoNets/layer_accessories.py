import cPickle as pickle
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image

import theano
import theano.tensor as T

from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

# library of functions for activation functions and rectifiers.
def function_options(key):
    if key == 'tanh':
        return T.tanh
    elif key == 'sigmoid':
        return T.nnet.sigmoid
    elif key == 'relu':
        def relu(x): return T.maximum(0, x)
        return relu
    elif key == 'step':
        def step(x): return T.gt(x, T.fill(x,0.5))
        return step
    elif key == 'softmax':    
        return T.nnet.softmax
    elif key == 'argmax':
        return T.argmax
    elif key == 'round':
        return T.round
    elif key == 'abs':
        return T.abs_
    elif key == 'sq':
		return T.sqr
    elif key in ('real' , 'None' , 'linear' , None):
        return lambda x: x
    else:
        print key
        raise NotImplementedError

def make_bad_outs(n_trace, x2):
        # since the img is ravelled, combinations of these numbers conveniently bring me to the 'centre' of the image and its NN
        s = int(np.sqrt(n_trace))
        sa, sb, sc = int(s/2-1), int(s/2), int(s/2+1) 
        NN_values = np.asarray([sb*s+sc, sa*s+sc, sa*s+sb, sa*s+sa, sb*s+sa, sc*s+sa, sc*s+sb, sc*s+sc], dtype = 'int32')

        # we want the columns of x2 at the indices specified by NN, inverted on [0,1]
        bad_outputs = T.transpose(T.transpose(x2)[NN_values])
        bad_outputs = T.join(1, bad_outputs, T.shape_padright(T.zeros_like(bad_outputs[:,0]))) # with stop option

        # we assume that of the nearest neighbours, exactly 1 will be part of the trace.  This is justified given the restrictions we place on the next point in the trace
        already_traced = T.argmax(bad_outputs, axis = 1) 

        # whitten indices within +-2 (%8) of the already traced pixel.  This restrics the output to 3 possible choices, all going forward
        ones = T.ones_like(already_traced)
        for i in xrange(-2,3):
                if not i == 0:
                    indices_to_whitten = (already_traced - i)%8
                    bad_outputs = T.set_subtensor(bad_outputs[T.arange(bad_outputs.shape[0]), indices_to_whitten], ones)

        # a vector of zeros and ones with zeros at the bad outputs
        bad_outputs = T.ones_like(bad_outputs) - bad_outputs  

        return bad_outputs

def cost_add_func(cost_add, layers):
        cost = 0
        dynamic_params = []
        for i in xrange(len(cost_add)):
          if not cost_add[i] == None:
                  if len(cost_add[i]) == 3:
                        err_label, coef, indices = cost_add[i]
                        dynamics = 1
                  elif len(cost_add[i]) == 4:
                        err_label, coef, indices, dynamics = cost_add[i]

                  #dynamic_param = T.scalar('%s_coef'%err_label)

                  err = err_add_ons(layers, err_label, indices)

                  if not err == None:
                    cost += coef*err #dynamic_param * err
                  #dynamic_params.append([dynamic_param, coef, dynamics])
        return cost #, dynamic_params


def err_add_ons(layers, err_label, layer_indices = None):
  if layer_indices == None:
        layer_indices = np.arange(len(layers))

  add_on = 0

  for l in layer_indices:
          if err_label == 'sparse1':
            add_on +=  T.sum(T.abs_(layers[l].output), axis = 1).mean()
          elif err_label == 'sparse2':
            add_on +=  T.sum(layers[l].output ** 2, axis = 1).mean()
          elif err_label == 'L2':
            add_on += T.mean(layers[l].params[0]**2)
          elif err_label == 'L1':
            add_on += T.mean(T.abs_(layers[l].params[0]))
          elif err_label == 'J':
                if layers[l].act_name == 'relu':
                        J = T.maximum(0, layers[l].lin_output/T.abs_(layers[l].lin_output)).dimshuffle(0, 'x', 1) * layers[l].params[0].dimshuffle('x', 0, 1)
                elif layers[l].act_name == 'sigmoid':
                        J = (layers[l].output*(1-layers[l].output)).dimshuffle(0, 'x', 1) * layers[l].params[0].dimshuffle('x', 0, 1)
                add_on += T.sum(J**2)/layers[l].lin_output.shape[0]
          elif err_label == 'H':
                p = layers[l].output
                add_on += T.mean(T.sum(p * T.log(p), axis = 1))

  return add_on


def make_params_to_train(params, params_i_to_train):
	params_to_train = []
        for i in xrange(len(params)):
                if params_i_to_train in ("all", None):
                        this_layer_params = [j for j in xrange(len(params[i]))]
                else:
                        if params_i_to_train[i] == "all":
                                this_layer_params = [j for j in xrange(len(params[i]))]
                        elif params_i_to_train[i] == None:
                                this_layer_params = []
                        else:
                                this_layer_params = params_i_to_train[i]
                params_to_train.append(this_layer_params)

	return params_to_train

# the glorious dropout droplet
# source: github.com/mdenil/dropout/
def dropout_from_layer(rng, layer, p):    
    # p is probability of dropping unit
    srng = T.shared_randomstreams.RandomStreams(rng.randint(99999))
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    output = layer * T.cast(mask, theano.config.floatX)
    return output

def convolve(input, kerns, in_shp, kern_shp, connectivity_table=None):

	if connectivity_table == None: # layer is fully connected:
		out = conv.conv2d(input=input, filters=kerns,
                filter_shape=kern_shp, image_shape=in_shp)
		return out
	else:
		# implement restricted connectivity, according to connectivity_table 
		# ie. for a given input map, which output feature map to add the conv output to
		raise NotImplementedError
