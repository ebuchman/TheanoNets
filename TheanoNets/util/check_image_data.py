""" Test script to visualize some of the random Curve data.
@author Graham Taylor
"""
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('code')
sys.path.append('code/util')

from util import get_data_path, dispims, tile_raster_images
from util.serialization import deserialize_object
import os.path

data_path = get_data_path()

# will not reload if data is already in workspace
try:
    datasets
except NameError:
    print 'loading data'
    datasets = deserialize_object(os.path.join(data_path, 'results/params_tracer_data_multi_full_33_17_17_nC1_11000_switch1.00_2718_noise_nh1a_300_nh1b_50_nh2_500_nout_9_preTrainFalse_LRi_0.01000_reg_10.00_dropoutFalse_2.647312.pkl'))

w0 = datasets[2].T
print w0.shape

n_cases, n_dims = w0.shape
im_w = int(np.sqrt(n_dims))  # assume square
case_w = int(np.sqrt(n_cases))+1

out = tile_raster_images(w0, (im_w, im_w), (case_w, case_w), tile_spacing=(3,3))
plt.imshow(out, cmap='gray')
plt.show()
quit()

#map_w = np.sqrt(n_dims_out)  # assume square
print im_w

n_train_batches = int(np.ceil(float(n_cases) / show_batchsize))


for b in xrange(n_train_batches):
    plt.figure(b)
    plt.subplot(1, 2, 1)
    batch_start = b * show_batchsize
    batch_end = min((b + 1) * show_batchsize, n_cases)

    this_view = w0[batch_start:batch_end]

    # must send matrix and exampes in second dimensiton (ie .T)
    dispims(this_view.T, im_w, im_w, border=2, bordercolor=this_view.max())
    '''
    plt.subplot(1, 2, 2)
    this_view = train_y[batch_start:batch_end]
    dispims(this_view.T, map_w, map_w, border=2, bordercolor=this_view.max())
    '''
