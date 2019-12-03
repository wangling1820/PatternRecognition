import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FuncFormatter
from keras.models import Model, load_model 
from keras.applications import vgg16, inception_v3
from keras.preprocessing import image
path = '/home/wangling/faceImages/project/weights/VGG.h5'
# in_path = '/home/wangling/faceImages/project/weights/GoogleNet.h5'
# inception = '/home/wangling/faceImages/project/weights/GoogLeNet_Max_01.h5'
# base_model = load_model(path)
# base_model.summary()
"""
fc1 (Dense)                  (None, 4096)              102764544 
_________________________________________________________________
fc2 (Dense)                  (None, 4096)              16781312  
"""
# # base_model = load_model(path)
# # model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)
# # model.summary()
# def num_2_percent(temp, pos):
#     return '%2.2f' % temp + '%'

# x = np.arange(8)
# x = x*0.1
# y = np.arange(10, dtype='float32')
# y = np.array([0.02, 0.33, 0.32, 0.1, 0.23, 0.4, 0.6, 0.3])
# plt.gca().yaxis.set_major_formatter(FuncFormatter(num_2_percent))
# plt.plot(x, y)
# plt.show()
name = 'GoogLeNetFC'
print(name[:-2])