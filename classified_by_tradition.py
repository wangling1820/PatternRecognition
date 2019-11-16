import os
import operator
import cv2
import numpy as np

from sklearn import preprocessing
from scipy.spatial.distance import cdist

# class classified_by_tradition(object):
#     self.unknown_folder = './unknown/'
#     self.known_folder = './known/'
#     self.typle = 'rgb'


def image2fea(folder='./known/', feas_dir='./RGB/known_feas.npy', 
        labels_dir='./RGB/known_labels.npy', way='rgb'):
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
        image = cv2.imread(img_path) 
        if way=='gray':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img_shape = image.shape
            fea = image.reshape([img_shape[0]*img_shape[1]])
            feas.append(fea)
        else:
            img_shape = image.shape
            fea = image.reshape([img_shape[0]*img_shape[1]*img_shape[2]])
            feas.append(fea)
    feas = np.array(feas)
    labels = np.array(labels)
    # print(feas)
    # print(labels[0: 10])
    # print(feas.shape)
    # print(labels.shape)
    np.save(feas_dir, feas)
    np.save(labels_dir, labels)
    print('图像特征提取完毕')


def get_all_distance(known_feas, unknown_feas):
    distance = []
    for unknown_fea in unknown_feas:
        dis = cdist(np.expand_dims(unknown_fea, axis=0), known_feas, metric='euclidean')[0] # 取相反数便于排序
        distance.append(dis)
    return np.vstack(distance)



def classified_by_knn(known_feas_dir='./RGB/known_feas.npy', known_labels_dir='./RGB/known_labels.npy',
            unknown_feas_dir='./RGB/unknown_feas.npy', unknown_labels_dir='./RGB/unknown_labels.npy'):
    known_feas = np.load(known_feas_dir)
    known_labels = np.load(known_labels_dir)

    unknown_feas = np.load(unknown_feas_dir)
    unknown_labels = np.load(unknown_labels_dir)

    for unknown in unknown_feas:
        dist = -cdist(np.expand_dims(unknown, axis=0), known_feas,metric='euclidean')

        

def autoNorm(dataSet):
    # ndarray.min(axis=None, out=None, keepdims=False, initial=<no value>, where=True)
    dataMin = dataSet.min(0)    
    dataMax = dataSet.max(0)    
    dataRange = dataMax - dataMin   # 此时的维度与原始数据不同,需要进行惟独的扩展
    dataRange = np.tile(dataRange, (dataSet.shape[0], 1))
    dataMin = np.tile(dataMin, (dataSet.shape[0], 1))
    dataSetNorm = (dataSet - dataMin) / dataRange
    print(dataSetNorm)
    return dataSetNorm


def classify_by_knn(distances, known_labels, unknown_labels, K=5):
    error_cnt = 0.0

    for index, dis in zip(range(len(unknown_labels)), distances):
        sorted_dis_indices = dis.argsort()
        class_cnt = {}
        for i in range(K):
            classLabel = known_labels[sorted_dis_indices[i]]
            class_cnt[classLabel] = class_cnt.get(classLabel, 0) + 1
        sorted_class_cnt = sorted(class_cnt.items(), key=operator.itemgetter(1), reverse=True)
        predicted_label = sorted_class_cnt[0][0]
        if predicted_label != unknown_labels[index]:
            error_cnt += 1.0
    error_ratio = error_cnt/len(unknown_labels)
    print(error_ratio)
    return error_ratio


if __name__ == "__main__":
    a = np.load('/home/wangling/faceImages/project/RGB/known_feas.npy')
    b = np.load('/home/wangling/faceImages/project/RGB/unknown_feas.npy')
    al = np.load('/home/wangling/faceImages/project/RGB/known_labels.npy')
    bl = np.load('/home/wangling/faceImages/project/RGB/unknown_labels.npy')
    dis = get_all_distance(a, b)
    error_ratio = classify_by_knn(dis, al, bl)
    print(error_ratio)

    # data = preprocessing.minmax_scale(a, feature_range=(0, 1), axis=0, copy=True)
    # data = autoNorm(a)
    # print(data)

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
