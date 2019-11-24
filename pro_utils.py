import os
import numpy as np
import cv2


class classified_by_pixels(object):
    pass


def gen_info(folder='./unknown', info='./info/unknown.txt'):
    
    '''
    功能为:根据folder文件夹中的图像名称生成图像的类列,并存储在info文件中.
    :param folder 文件夹的路径
    :param info 生成文件的路径
    '''
    file_list = os.listdir(folder)
    with open(info, 'w') as output:
        for file in file_list:
            print(file.__class__)
            output.write(file + '\t' + str(int(file[1:4])) + '\n')


def rgb_feature(image_path):
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


def image2fea(folder='./known/', feas_dir='./RGB/known_feas.npy', 
        labels_dir='./RGB/known_labels.npy', feature_extractor=rgb_feature):
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


def hsv_feature(image_path):
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


def get_labels(known_labels='./FullyCon/known_labels.npy', unknown_labels='./FullyCon/unknown_labels.npy'):
    known_labels = np.load(known_labels)
    unknown_labels = np.load(unknown_labels)

    index = 0
    hash_map = {}
    known_list = []
    unknown_list = []

    for label in known_labels:
        if label not in hash_map.keys():
            hash_map[label] = index
            index += 1
        known_list.append(hash_map[label])

    for label in unknown_labels:
        if label not in hash_map.keys():
            raise RuntimeError('%d is not found', label)
        unknown_list.append(hash_map[label])
    
    known_labels = np.array(known_list)
    unknown_labels = np.array(unknown_list)
    return known_labels, unknown_labels


if __name__ == "__main__":
    known_labels = np.load('./FullyCon/known_labels.npy')
    unknown_labels = np.load('./FullyCon/unknown_labels.npy')

    index = 1
    hash_map = {}
    known_list = []
    unknown_list = []

    for label in known_labels:
        if label not in hash_map.keys():
            hash_map[label] = index
            index += 1
        known_list.append(hash_map[label])

    for label in unknown_labels:
        if label not in hash_map.keys():
            raise RuntimeError('%d is not found', label)
        unknown_list.append(hash_map[label])
    
    known_labels = np.array(known_list)
    unknown_labels = np.array(unknown_list)
 
    
    