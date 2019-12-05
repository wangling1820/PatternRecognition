import os
import operator
import time

import cv2
import numpy as np
from scipy.spatial.distance import cdist

from base import base


class ClassifiedByPixels(base):
    def __init__(self, type):
        super().__init__(type)
        
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


    def get_hist_distance(self, unknown_fea, knwon_feas):
        """
        相关性比较 (method=cv.HISTCMP_CORREL) 值越大，相关度越高，最大值为1，最小值为0
        卡方比较(method=cv.HISTCMP_CHISQR 值越小，相关度越高，最大值无上界，最小值0
        巴氏距离比较(method=cv.HISTCMP_BHATTACHARYYA) 值越小，相关度越高，最大值为1，最小值为0
        """
        distance = []
        for known_fea in knwon_feas:
            # dist = -cv2.compareHist(unknown_fea, known_fea, cv2.HISTCMP_CORREL)  # 取相反数便于计算
            dist = cv2.compareHist(unknown_fea, known_fea, cv2.HISTCMP_BHATTACHARYYA) 
            distance.append(dist)
        return np.array(distance)

    def get_all_distance(self):
        distance = []
        for unknown_fea in self.unknown_feas:
            dis = self.get_hist_distance(unknown_fea, self.known_feas)
            distance.append(dis)
        self.distances = np.stack(distance)


def test_hsv(first=False):
    test = ClassifiedByPixels('HSV')
    if first:
        start = time.time()
        test.image2features()
        end = time.time()
        used_time = int(end - start)
        print('hsv提取特征的时间是 %d 分, %d 秒.'%(used_time/60, used_time%60))
    else:
        test.load_feas()
    test.get_all_distance()
    test.classify()
    test.evaluting()

def test_rgb(first=False):
    test = ClassifiedByPixels('RGB')
    if first:
        start = time.time()
        test.image2features()
        end = time.time()
        used_time = int(end - start)
        print('rgb提取特征的时间是 %d 分, %d 秒.'%(used_time/60, used_time%60))
    else:
        test.load_feas()
    test.get_all_distance()
    test.classify()
    test.evaluting()

    
if __name__ == "__main__":
    start = time.time()
    test_rgb(True)
    end = time.time()
    used_time = int(end - start)
    print('rgb使用的总时间是 %d 分, %d 秒.'%(used_time/60, used_time%60))

    start = time.time()
    test_hsv(True)
    end = time.time()
    used_time = int(end - start)
    print('rgb使用的总时间是 %d 分, %d 秒.'%(used_time/60, used_time%60))
    """
    rgb and cv2.HISTCMP_BHATTACHARYYA
    K = 1 时的分类错误率为 0.7101 K = 1 时的分类正确率为 0.2899
    K = 5 时的分类错误率为 0.7645 K = 5 时的分类正确率为 0.2355
    K = 10 时的分类错误率为 0.8152 K = 10 时的分类正确率为 0.1848
    K = 15 时的分类错误率为 0.8370 K = 15 时的分类正确率为 0.1630
    K = 20 时的分类错误率为 0.8225 K = 20 时的分类正确率为 0.1775

    hsv and cv2.HISTCMP_BHATTACHARYYA
    K = 1 时的分类错误率为 0.6558 K = 1 时的分类正确率为 0.3442
    
    K = 5 时的分类错误率为 0.6739 K = 5 时的分类正确率为 0.3261
    K = 10 时的分类错误率为 0.6522 K = 10 时的分类正确率为 0.3478
    K = 15 时的分类错误率为 0.6558 K = 15 时的分类正确率为 0.3442
    K = 20 时的分类错误率为 0.6775 K = 20 时的分类正确率为 0.3225

    rgb hsv and euclidean
    K = 1 时的分类错误率为 0.8478 K = 1 时的分类正确率为 0.1522
    K = 5 时的分类错误率为 0.8370 K = 5 时的分类正确率为 0.1630
    K = 10 时的分类错误率为 0.8841 K = 10 时的分类正确率为 0.1159
    K = 15 时的分类错误率为 0.8768 K = 15 时的分类正确率为 0.1232
    K = 20 时的分类错误率为 0.8768 K = 20 时的分类正确率为 0.1232

    K = 1 时的分类错误率为 0.8152 K = 1 时的分类正确率为 0.1848
    K = 5 时的分类错误率为 0.8152 K = 5 时的分类正确率为 0.1848
    K = 10 时的分类错误率为 0.8297 K = 10 时的分类正确率为 0.1703
    K = 15 时的分类错误率为 0.8333 K = 15 时的分类正确率为 0.1667
    K = 20 时的分类错误率为 0.8370 K = 20 时的分类正确率为 0.1630

    rgb hsv and HISTCMP_CHISQR
    K = 1 时的分类错误率为 0.7029 K = 1 时的分类正确率为 0.2971
    K = 5 时的分类错误率为 0.7826 K = 5 时的分类正确率为 0.2174
    K = 10 时的分类错误率为 0.8188 K = 10 时的分类正确率为 0.1812
    K = 15 时的分类错误率为 0.8043 K = 15 时的分类正确率为 0.1957
    K = 20 时的分类错误率为 0.8116 K = 20 时的分类正确率为 0.1884

    K = 1 时的分类错误率为 0.6812 K = 1 时的分类正确率为 0.3188
    K = 5 时的分类错误率为 0.7210 K = 5 时的分类正确率为 0.2790
    K = 10 时的分类错误率为 0.7609 K = 10 时的分类正确率为 0.2391
    K = 15 时的分类错误率为 0.7790 K = 15 时的分类正确率为 0.2210
    K = 20 时的分类错误率为 0.7790 K = 20 时的分类正确率为 0.2210

    rgb and hsv HISTCMP_CORREL
    K = 1 时的分类错误率为 0.8478 K = 1 时的分类正确率为 0.1522
    K = 5 时的分类错误率为 0.8442 K = 5 时的分类正确率为 0.1558
    K = 10 时的分类错误率为 0.8804 K = 10 时的分类正确率为 0.1196
    K = 15 时的分类错误率为 0.8696 K = 15 时的分类正确率为 0.1304
    K = 20 时的分类错误率为 0.8804 K = 20 时的分类正确率为 0.1196

    K = 1 时的分类错误率为 0.8152 K = 1 时的分类正确率为 0.1848
    K = 5 时的分类错误率为 0.8225 K = 5 时的分类正确率为 0.1775
    K = 10 时的分类错误率为 0.8261 K = 10 时的分类正确率为 0.1739
    K = 15 时的分类错误率为 0.8333 K = 15 时的分类正确率为 0.1667
    K = 20 时的分类错误率为 0.8406 K = 20 时的分类正确率为 0.1594
    """
    '''
    图像特征提取完毕
    图像特征提取完毕
    rgb提取特征的时间是 0 分, 25 秒.
    K = 1 时的分类错误率为 0.7101 K = 1 时的分类正确率为 0.2899
    K = 5 时的分类错误率为 0.7645 K = 5 时的分类正确率为 0.2355
    K = 10 时的分类错误率为 0.8152 K = 10 时的分类正确率为 0.1848
    K = 15 时的分类错误率为 0.8370 K = 15 时的分类正确率为 0.1630
    K = 20 时的分类错误率为 0.8225 K = 20 时的分类正确率为 0.1775
    rgb使用的总时间是 0 分, 28 秒.
    图像特征提取完毕
    图像特征提取完毕
    hsv提取特征的时间是 0 分, 25 秒.
    K = 1 时的分类错误率为 0.6558 K = 1 时的分类正确率为 0.3442
    K = 5 时的分类错误率为 0.6739 K = 5 时的分类正确率为 0.3261
    K = 10 时的分类错误率为 0.6522 K = 10 时的分类正确率为 0.3478
    K = 15 时的分类错误率为 0.6558 K = 15 时的分类正确率为 0.3442
    K = 20 时的分类错误率为 0.6775 K = 20 时的分类正确率为 0.3225
    rgb使用的总时间是 0 分, 28 秒.
    '''
 