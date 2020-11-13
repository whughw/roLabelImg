import numpy as np
from libs.lib import struct
def inference_detector(detector, im):
    return np.array([[[100,100,150,150,1]],
                      [[150,150,200,200,1]],
                      [[200,200,250,250,1]]])

def init_detector(config,weight):
    model = struct(CLASSES=['a','b','c'])
    return model