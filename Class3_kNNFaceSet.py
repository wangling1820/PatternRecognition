"""
baseline
"""
import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# torch.utils.data.Dataset是一个PyTorch用来表示数据集的抽象类。我们用这个类来处理自己的数据集的时候必须继承Dataset,然后重写下面的函数：
# __len__: 使得len(dataset)返回数据集的大小；
# __getitem__：使得支持dataset[i]能够返回第i个数据样本这样的下标操作。
class kNNFaceSet(Dataset):

    def __init__(self, root_path):

        self.mDataX = []
        self.mDataY = []
        file_list  = os.listdir(root_path)

        for path in file_list:
            img_path = os.path.join(root_path, path)
            img = Image.open(img_path)
            img = img.convert('L')  # 转换成灰度图
            # new_size = np.array(img.size) / 4
            # new_size = new_size.astype(int)
            # img = img.resize(new_size, Image.BILINEAR)  # 从(宽，高)(640, 480)缩小为(160, 120)
            img_data = np.array(img, dtype=float)
            img_data = img_data.reshape(-1)     # 将数据转化成一行
            self.mDataX.append(img_data)
            img_id = int(path[1: 4])
            self.mDataY.append(img_id)

    def __getitem__(self, data_index):
        input_tensor = torch.tensor(self.mDataX[data_index])
        output_tensor = torch.tensor(self.mDataY[data_index])
        return input_tensor, output_tensor

    def __len__(self):
        return len(self.mDataX)


if __name__ == "__main__":

    img = Image.open('/home/wangling/faceImages/project/unknown/i000qa-fn.jpg')
    img = img.convert('L')  # 转换成灰度图
    # new_size = np.array(img.size) / 4
    # new_size = new_size.astype(int)
    # img = img.resize(new_size, Image.BILINEAR)  # 从(宽，高)(640, 480)缩小为(160, 120)
    img_data = np.array(img, dtype=float)
    img_data = img_data.reshape(-1)
    print(img_data.__class__)
    print(img_data.shape)
