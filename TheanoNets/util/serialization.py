""" Serialization utilities. """

import cPickle as pickle
import datetime
import os.path
import logging
import tempfile

logger = logging.getLogger(__name__)

def serialize_object(obj, fpath='.', fname=None, gz=False):
    """ Serialize object, giving it a default name if one is not specified. """
    fpathstart, fpathext = os.path.splitext(fpath)
    if fpathext == '.pkl' or fpathext == '.pklz':
        # User supplied an absolute path to a pickle file
        fpath, fname = os.path.split(fpath)
        if fpathext == '.pklz':
            gz = True
    elif fname is None:
        if gz:
            fpathext = '.pklz'
        else:
            fpathext = '.pkl'

        # Generate filename based on date
        date_obj = datetime.datetime.now()
        date_str = date_obj.strftime('%Y-%m-%d-%H:%M:%S')
        class_name = obj.__class__.__name__
        fname = '%s.%s%s' % (class_name, date_str, fpathext)

    fabspath = os.path.join(fpath, fname)

    logger.info("Saving to %s ..." % fabspath)
    if gz:
        import gzip
        file = gzip.open(fabspath, 'wb')
    else:
        file = open(fabspath, 'wb')
    pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()


def deserialize_object(fpath, gz=False):
    """ Deserialize object from path. """
    # Try to decide on whether or not pickle is gzipped by extension
    fpathstart, fpathext = os.path.splitext(fpath)
    if fpathext == '.pklz':
        gz = True
    if gz:
        import gzip
        file = gzip.open(fpath, 'rb')
    else:
        file = open(fpath, 'rb')
    return pickle.load(file)
