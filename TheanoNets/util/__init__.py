""" Miscellaneous utilities. """
import cPickle as pickle
from PIL import Image
import os
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import socket
import datetime
import logging


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`PIL.Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                        in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(out_shape,
                        dtype=dt) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                        ] = this_img * c
        return out_array


def dispims(M, height, width, border=0, bordercolor=0.0, layout=None, show=True, **kwargs):
    """ Display a whole stack (colunmwise) of vectorized matrices. Useful 
        eg. to display the weights of a neural network layer.
    """
    numimages = M.shape[1]
    if layout is None:
        n0 = int(np.ceil(np.sqrt(numimages)))
        n1 = int(np.ceil(np.sqrt(numimages)))
    else:
        n0, n1 = layout
    im = bordercolor * np.ones(((height+border)*n0+border,(width+border)*n1+border),dtype='<f8')
    for i in range(n0):
        for j in range(n1):
            if i*n1+j < M.shape[1]:
                im[i*(height+border)+border:(i+1)*(height+border)+border,
                   j*(width+border)+border :(j+1)*(width+border)+border] = np.vstack((
                            np.hstack((np.reshape(M[:,i*n1+j],(height, width)),
                                   bordercolor*np.ones((height,border),dtype=float))),
                            bordercolor*np.ones((border,width+border),dtype=float)
                            ))
    
    plots = plt.imshow(im, cmap=plt.cm.gray, interpolation='nearest', **kwargs)
    
    if show == True:
        plt.show()
    else:
        return plots
        
def setupLogging(dir_name=None):
    ROOT = get_data_path()
    # debug and up goes to file
    # info and up goes to screen

    if dir_name == None: 
        now = datetime.datetime.now()
        year, month, day, hour, minute = now.year, now.month, now.day, now.hour, now.minute
        current = '%d_%d_%d_%d_%d'%(year, month, day, hour, minute)
        dir_name = current
        
        
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    
    dir = os.path.join(ROOT, dir_name)
    if not os.path.exists(dir): os.makedirs(dir)
    hdlr = logging.FileHandler(os.path.join(dir, 'details.log'))
    hdlr.setLevel(logging.DEBUG)
    logger.addHandler(hdlr)

    return logger


def compute_AUC(output, target, plot = False):
    # output and target are matrices where each row is a serialized image patch.  output is probabilistic and target is near binary (includes gaussian dilation)
    from sklearn.metrics import auc
    threshold_steps = 100
    fuzzy_margin = 3                                                 
    num_patches = output.shape[0]
    
    precision = np.zeros((num_patches, threshold_steps))
    recall = np.zeros((num_patches, threshold_steps))
                                                

    for index in xrange(num_patches):
        serial_out = output[index]
        serial_target = target[index]
        serial_target[serial_target < 1] = 0

        #takes about 0.08 seconds per patch
        precision[index], recall[index] = precision_recall(serial_out, serial_target, threshold_steps, fuzzy_margin)

    mean_precision = np.mean(precision, 0)
    mean_recall = np.mean(recall, 0)

    auc = auc(mean_recall, mean_precision)
    
    if plot == True:
        plot_AUC(mean_recall, mean_precision, auc)
    
    return auc     


def precision_recall(output, target, threshold_steps, margin):
    # output is probabilistic 
    # target is binary
    patch_width = float(output.size)**0.5

    precision = np.zeros(threshold_steps)
    recall = np.zeros(threshold_steps)                                                 
                                          
    for i in xrange(threshold_steps):
        threshold = float(i)/threshold_steps
        threshed_output = output.copy()                                            
        threshed_output[output < threshold] = 0
        threshed_output[output >= threshold] = 1                                           

        threshed_output.resize(patch_width, patch_width)
        target.resize(patch_width, patch_width)

        precision[i] = compute_precision(threshed_output, target, margin)
        recall[i] = compute_recall(threshed_output, target, margin)                                             


    return precision, recall
                               

def compute_precision(output, target, margin):
    # both output and target are binary.  Computes the fractions of white pixels in output that exist in target,
    # where existence indicates a white pixel in target that is within margin pixels from a white pixel in output
    num_white = output[output == 1].size
    
    target_invert = 1-target
    distance_map = ndimage.distance_transform_edt(target_invert)
    
    if num_white == 0:
        precision = 1
    else:
        num_precise = ((output == 1) & (distance_map <= margin)).sum()
        precision = (num_precise/num_white)                                             

    return precision

                                                     
def compute_recall(output, target, margin):
    # both output and target are binary.  Computes the fraction of white pixels in target recalled by output,
    # where recalled indicates a white pixel in output that is within margin pixels from a white pixel in target
    num_real = target[target == 1].size
    
    output_invert = 1-output
    distance_map = ndimage.distance_transform_edt(output_invert)

    num_recalled = ((target == 1) & (distance_map <= margin)).sum()                                 

    if num_real == 0:
        recall = 1
    else:
        recall = float(num_recalled)/num_real

    return recall                                                 
                   
def plot_AUC(recall, precision, area):
    import pylab as pl
    
    pl.clf()
    pl.plot(recall, precision, label='Precision-Recall curve')
    pl.xlabel('Recall')
    pl.ylabel('Precision')
    pl.ylim([0.0, 1.05])
    pl.xlim([0.0, 1.0])
    pl.title('Precision-Recall example: AUC=%0.2f' % area)
    pl.legend(loc="lower left")
    pl.show()   

def rebuild_image(data, data_name=None, img_num = None):
    ## If the image is within a data set, it will come with an image number, and have missing patches.
    ## If image is stand alone, it will not come with img_num

    if not img_num == None:
        print '...loading missing_stats'
        data_path = get_data_path()

        ########################
        #my missing is probably all fucked from doing new serialize/section stuff...
        ########################

        
        f = open(os.path.join(data_path, '%s_missing_stats.pkl'%data_name), 'rb')
        missing=pickle.load(f)
        f.close()
        num_blocks = data.shape[0]+len(missing[img_num])
    else:
        num_blocks = data.shape[0]
    
    print '...rebuilding image'

    data*=255
    
    block_size = data.shape[1]**0.5 #assume square
    img_size = block_size*(num_blocks**0.5)

    img = np.zeros((img_size,img_size))

    current_pos = [0,0] #pixel position in full image
    non_trivial_counter = 0 #keeps track of position within the data (real output patches).. only relevant if patches are missing (otherwise it trivializes)

#    print data.shape, num_blocks, img_size, block_size


    #rebuild image: 
    for i in xrange(num_blocks):
        #print i, non_trivial_counter, current_pos[0], current_pos[1]
            
        if not img_num == None and not (i in missing[img_num]):
            img[current_pos[0]:(current_pos[0]+block_size), current_pos[1]:(current_pos[1]+block_size)] = data[non_trivial_counter].reshape(block_size, block_size)
            non_trivial_counter += 1
        elif img_num == None:
            img[current_pos[0]:(current_pos[0]+block_size), current_pos[1]:(current_pos[1]+block_size)] = data[non_trivial_counter].reshape(block_size, block_size)
            non_trivial_counter += 1
            
        if current_pos[1] < (img_size - block_size):
            current_pos[1] += block_size
        else:
            current_pos[1] = 0
            current_pos[0] += block_size

    img_pil = Image.fromarray(np.uint8(img))
    return img_pil

def get_data_path(glob = False):
    name = socket.gethostname()
    """ Use hostname to determine where data is saved. """
    if 'sharcnet' in name or 'ang' in name or 'gup' in name or 'mon' in name or 'mako' in name: #name in  ('sharcnet' , 'ang' , 'gup' , 'mon', 'mako'):# 
        # Sharcnet
        data_path = '/work/ebuchman/data'
    elif 'MacBook' in name and glob == True:
      data_path = '/Users/BatBuddha/Programming/Datasets/'
    elif 'SOE' in name:
      if glob == True:
          data_path = '/export/mlrg/ebuchman/datasets'
      else:
          data_path = './data'
    else:
      data_path = './data'  # default
    

    return data_path
