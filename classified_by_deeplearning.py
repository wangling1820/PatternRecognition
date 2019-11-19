import os
import numpy as np

from keras.models import Model, load_model 
from keras.applications import vgg16, inception_v3
from keras.preprocessing import image

from base import base

class ClassifiedByDeepLearning(base):
    def __init__(self, type):
        super().__init__(type)
        if self.type == 'GoogleNet':
            # self.model_path = './weights/GoogleNet.h5'
            self.model_path = './weights/GoogLeNet.h5'
            self.layer_name = 'Avg_Pooling'
        elif self.type == 'VGG':
            self.model_path = './weights/VGG.h5'
            self.layer_name = 'fc2'


    def get_model(self):
        base_model = load_model(self.model_path)
        self.model = Model(input=base_model.input, output=base_model.get_layer(self.layer_name).output)


    def googlenet_feature(self, image_path):
        img = image.load_img(path=image_path, target_size=(299, 299))
        img = image.img_to_array(img=img)
        img = np.expand_dims(img, axis=0)
        img = inception_v3.preprocess_input(img)
        feature = self.model.predict(img)[0]         # [[2,2,2,2]] --> [2, 2, 2, 2]
        return feature


    def vgg_feature(self, image_path):
        img = image.load_img(path=image_path, target_size=(224, 224))
        img = image.img_to_array(img=img)
        img = np.expand_dims(img, axis=0)
        img = vgg16.preprocess_input(img)
        fea = self.model.predict(img)[0]     # [[2,2,2,2]] --> [2, 2, 2, 2]
        return fea


    def get_image_fea(self, folder, feas_dir, labels_dir, feature_extractor):
        print('提取图像特征中.....')
        feas = []
        labels = []
        file_list = os.listdir(folder)
        for file in file_list:
            labels.append(int(file[1:4]))   #   保存每个特征对应的类别
            img_path = os.path.join(folder, file)   # 读取图像进行特征提取
            fea = feature_extractor(img_path)
            feas.append(fea)
        feas = np.array(feas)
        labels = np.array(labels)
        self.autoNorm(feas)
        np.save(feas_dir, feas)
        np.save(labels_dir, labels)
        print('图像特征提取完毕')
        return feas, labels


    def autoNorm(self, dataSet):
        # ndarray.min(axis=None, out=None, keepdims=False, initial=<no value>, where=True)
        dataMin = dataSet.min(0)    
        dataMax = dataSet.max(0)    
        dataRange = dataMax - dataMin   # 此时的维度与原始数据不同,需要进行惟独的扩展
        dataRange = np.tile(dataRange, (dataSet.shape[0], 1))
        dataMin = np.tile(dataMin, (dataSet.shape[0], 1))
        dataSetNorm = (dataSet - dataMin) / dataRange
        return dataSetNorm
        

    def image2features(self):
        if self.type == 'GoogleNet':
            self.known_feas, self.known_labels = self.get_image_fea(folder=self.known_folder, feas_dir=self.known_feas_dir, 
                    labels_dir=self.known_labels_dir, feature_extractor=self.googlenet_feature)

            self.unknown_feas, self.unknown_labels = self.get_image_fea(folder=self.unknown_folder, 
                    feas_dir=self.unknown_feas_dir, labels_dir=self.unknown_labels_dir,feature_extractor=self.googlenet_feature)
        elif self.type == 'VGG':
            self.known_feas, self.known_labels = self.get_image_fea(folder=self.known_folder, feas_dir=self.known_feas_dir, 
                    labels_dir=self.known_labels_dir, feature_extractor=self.vgg_feature)
                    
            self.unknown_feas, self.unknown_labels = self.get_image_fea(folder=self.unknown_folder, 
                    feas_dir=self.unknown_feas_dir, labels_dir=self.unknown_labels_dir,feature_extractor=self.vgg_feature)
        else:
            raise RuntimeError('输入的特征类别有误!')
    
     def get_all_distance_by_euclidean(self):
        super().get_all_distance()

def test_vgg(first=False):
    test = ClassifiedByDeepLearning('VGG')
    if first:
        test.get_model()
        test.image2features()
    else:
        test.load_feas()
    test.get_all_distance()
    test.classify()
    test.evaluting()


def test_googlenet(first=False):
    test = ClassifiedByDeepLearning('GoogleNet')
    if first:
        test.get_model()
        test.image2features()
    else:
        test.load_feas()
    test.get_all_distance()
    test.classify()
    test.evaluting()

if __name__ == '__main__':
    test_googlenet(False)
    test_vgg(False)

    """
    google and vgg
    K = 1 时的分类错误率为 0.0036 K = 1 时的分类正确率为 0.9964
    K = 5 时的分类错误率为 0.0000 K = 5 时的分类正确率为 1.0000
    K = 10 时的分类错误率为 0.0000 K = 10 时的分类正确率为 1.0000
    K = 15 时的分类错误率为 0.0109 K = 15 时的分类正确率为 0.9891
    K = 20 时的分类错误率为 0.0290 K = 20 时的分类正确率为 0.9710
    K = 1 时的分类错误率为 0.0181 K = 1 时的分类正确率为 0.9819
    K = 5 时的分类错误率为 0.0362 K = 5 时的分类正确率为 0.9638
    K = 10 时的分类错误率为 0.0507 K = 10 时的分类正确率为 0.9493
    K = 15 时的分类错误率为 0.0761 K = 15 时的分类正确率为 0.9239
    K = 20 时的分类错误率为 0.0833 K = 20 时的分类正确率为 0.9167
    """
    