import os
import time
import numpy as np
from keras.models import Model, load_model 
from keras.applications import vgg16, inception_v3
from keras.preprocessing import image
# path = '/home/wangling/faceImages/project/weights/vgg16.h5'
# in_path = '/home/wangling/faceImages/project/weights/GoogleNet.h5'
# inception = '/home/wangling/faceImages/project/weights/GoogLeNet_Max_01.h5'
# base_model = load_model(in_path)
# base_model.summary()
# # base_model = load_model(path)
# # model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)
# # model.summary()

def get_model(model_path, layer_name):
    base_model = load_model(path)
    model = Model(input=base_model.input, output=base_model.get_layer(layer_name).output)
    # model.summary()
    return model

def get_feas(image_path, model):
    img = image.load_img(path=image_path, target_size=(224, 224))
    img = image.img_to_array(img=img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    fea = model.predict(img)[0]            # [[2,2,2,2]] --> [2, 2, 2, 2]
    return fea

if __name__ == '__main__':
    print(time.strftime("%m-%d-%H-%M-%S", time.localtime()) )
