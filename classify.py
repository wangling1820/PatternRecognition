import os
import operator
import cv2
import numpy as np

from sklearn import preprocessing
from scipy.spatial.distance import cdist
from base import base


class ClassifiedByPixels(object):
    def __init__(self, type):
        self.unknown_folder = './unknown/'
        self.known_folder = './known/'
        type_list = ['RGB', 'HSV', 'GoogleNet', 'VGG']
        if type not in type_list:
            raise RuntimeError('classified_by_tradition:输入的参数:type无效!')
        self.type = type
        self.root_dir = os.path.join(os.getcwd(), type)

        if not os.path.isdir(self.root_dir):
            os.mkdir(self.root_dir)

        self.unknown_feas_dir = os.path.join(self.root_dir, 'unknown_feas.npy')
        self.unknown_labels_dir = os.path.join(self.root_dir, 'unknown_labels.npy')

        self.known_feas_dir = os.path.join(self.root_dir, 'known_feas.npy')
        self.known_labels_dir = os.path.join(self.root_dir, 'known_labels.npy')

    def rgb_feature(self, image_path):
        """
        计算图像的RGB颜色直方图,并经过归一化和平滑处理生成一个特征向量.
        Args:
            image_path: 图像的路径.
        Return:
            numpy包中的array类型的特征向量.
        Raise:
            当输入的图像路径无效时,抛出异常.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError('hist_feature:path of image is invaild!')
        hist0 = cv2.calcHist([img], [0], None, [256], [0., 255.])
        hist1 = cv2.calcHist([img], [1], None, [256], [0., 255.])
        hist2 = cv2.calcHist([img], [2], None, [256], [0., 255.])
        cv2.normalize(hist0, hist0)
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)
        hist = []
        hist.extend(hist0.flatten())
        hist.extend(hist1.flatten())
        hist.extend(hist2.flatten())
        return np.array(hist)


    def image2fea(self, folder, feas_dir, labels_dir, feature_extractor):
        '''
        将folder文件下的图片按照way方法转换成特征向量.
        :param folder: 文件夹路径
        :param feas_dir: 存储特征向量的路径
        :param labels_dir: 存储图像标签的路径
        :param way: 图像特征编码方式,此处只使用rgb和gray两种方式
        '''
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
        np.save(feas_dir, feas)
        np.save(labels_dir, labels)
        print('图像特征提取完毕')
        return feas, labels
        
            
    def hsv_feature(self, image_path):
        """
        计算图像的HSV颜色直方图,并经过归一化和平滑处理后生成一个特征向量.
        Args:
            image_path: 图像的路径.
        Return:
            numpy包中的array类型的特征向量.
        Raise:
            当输入的图像路径无效时,抛出异常.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError('hsv_feature:path of image is invaild!')
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv0 = cv2.calcHist([img_hsv], [0], None, [256], [0., 180.])
        hsv1 = cv2.calcHist([img_hsv], [1], None, [256], [0., 255.])
        hsv2 = cv2.calcHist([img_hsv], [2], None, [256], [0., 255.])
        cv2.normalize(hsv0, hsv0)
        cv2.normalize(hsv1, hsv1)
        cv2.normalize(hsv2, hsv2)
        hsv = []
        hsv.extend(hsv0.flatten())
        hsv.extend(hsv1.flatten())
        hsv.extend(hsv2.flatten())
        return np.array(hsv)
        
    def image2features(self):
        if self.type == 'RGB':
            self.known_feas, self.known_labels = self.image2fea(self.known_folder, self.known_feas_dir, self.known_labels_dir, self.rgb_feature)
            self.unknown_feas, self.unknown_labels = self.image2fea(folder=self.unknown_folder, 
                    feas_dir=self.unknown_feas_dir, labels_dir=self.unknown_labels_dir,feature_extractor=self.rgb_feature)
        elif self.type == 'HSV':
            self.known_feas, self.known_labels = self.image2fea(folder=self.known_folder, feas_dir=self.known_feas_dir, 
                    labels_dir=self.known_labels_dir, feature_extractor=self.hsv_feature)
                    
            self.unknown_feas, self.unknown_labels = self.image2fea(folder=self.unknown_folder, 
                    feas_dir=self.unknown_feas_dir, labels_dir=self.unknown_labels_dir,feature_extractor=self.hsv_feature)
        else:
            raise RuntimeError('输入的特征类别有误!')

    def load_feas(self):
        self.known_feas = np.load(self.known_feas_dir)
        self.known_labels = np.load(self.known_labels_dir)
        self.unknown_feas = np.load(self.unknown_feas_dir)
        self.unknown_labels = np.load(self.unknown_labels_dir)


    def get_all_distance(self):
        distance = []
        for unknown_fea in self.unknown_feas:
            dis = cdist(np.expand_dims(unknown_fea, axis=0), self.known_feas, metric='euclidean')[0]
            distance.append(dis)
        self.distances = np.stack(distance)


    def classify_by_knn(self, K=5): 
        error_cnt = 0.0
        for index, dis in zip(range(len(self.unknown_labels)), self.distances):
            sorted_dis_indices = dis.argsort()
            class_cnt = {}
            for i in range(K):
                classLabel = self.known_labels[sorted_dis_indices[i]]
                class_cnt[classLabel] = class_cnt.get(classLabel, 0) + 1
            sorted_class_cnt = sorted(class_cnt.items(), key=operator.itemgetter(1), reverse=True)
            predicted_label = sorted_class_cnt[0][0]
            if predicted_label != self.unknown_labels[index]:
                error_cnt += 1.0
        error_ratio = error_cnt / len(self.unknown_labels)
        return error_ratio

    def classify(self, K=[1, 5, 10, 15, 20]):
        self.error_ratio = {}
        for k in K:
            self.error_ratio[k] = self.classify_by_knn(K=k)

    def evaluting(self):
        for key in self.error_ratio.keys():
            print('K = %d 时的分类错误率为 %0.4f'%(key, self.error_ratio[key]), end=' ')
            print('K = %d 时的分类正确率为 %0.4f'%(key, 1 - self.error_ratio[key]))


if __name__ == "__main__":
    test = ClassifiedByPixels('HSV')
    test.image2features()
    test.get_all_distance()
    test.classify()
    test.evaluting()
    # test = classified_by_tradition('RGB')
    # test.load_info()
    # test.get_all_distance()
    # test.classify()
    # test.evaluting()
    # test1 = classified_by_tradition('HSV')
    # test1.load_info()
    # test1.get_all_distance()
    # test1.classify()
    # test1.evaluting()
    # dd = os.path.join(os.getcwd(), 'hello')
    # if not os.path.isdir(dd):
    #     print('bucunzai ' + dd)
    #     os.mkdir(dd)
    # dc = os.path.join(dd, 'dd')
    # a = np.array([1,1,1])
    # np.save(dc, a)
    # a = np.load('/home/wangling/faceImages/project/RGB/known_feas.npy')
    # b = np.load('/home/wangling/faceImages/project/RGB/unknown_feas.npy')
    # al = np.load('/home/wangling/faceImages/project/RGB/known_labels.npy')
    # bl = np.load('/home/wangling/faceImages/project/RGB/unknown_labels.npy')
    # dis = get_all_distance(a, b)
    # error_ratio = classify_by_knn(dis, al, bl)
    # print(error_ratio)

    # a = np.load('/home/wangling/faceImages/project/HSV/known_feas.npy')
    # b = np.load('/home/wangling/faceImages/project/HSV/unknown_feas.npy')
    # al = np.load('/home/wangling/faceImages/project/HSV/known_labels.npy')
    # bl = np.load('/home/wangling/faceImages/project/HSV/unknown_labels.npy')
    # dis = get_all_distance(a, b)
    # error_ratio = classify_by_knn(dis, al, bl)
    # print(error_ratio)
    # data = preprocessing.minmax_scale(a, feature_range=(0, 1), axis=0, copy=True)
    # data = autoNorm(a)
    # print(data)
    # [wangling@Manjaro project]$ python classified_by_tradition.py 
    # 0.8369565217391305
    # 0.8152173913043478

    # image2fea()
    # labels = np.load('./RGB/unknown_labels.npy')
    # print(labels)
    # image = cv2.imread('/home/wangling/faceImages/project/unknown/i000qa-fn.jpg')
    # cv2.imshow('hello', image)
    # print(image.shape)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print(image.shape)
    # cv2.imshow('world', image)
    # cv2.waitKey(0)
