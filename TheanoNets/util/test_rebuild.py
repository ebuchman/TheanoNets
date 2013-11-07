
from util import rebuild_image
import cPickle as pickle


f = open('data/curvyData.pkl','rb')
data = pickle.load(f)

rebuild_image(data[0],0)
