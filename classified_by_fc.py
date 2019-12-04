import os
import time

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.models import load_model 
from keras import optimizers
from keras.layers import Dropout
from matplotlib.ticker import FuncFormatter

from base import base
from classified_by_models import ClassifiedByModels

class ClassifiedByFC(base):
    def __init__(self, type):
        super().__init__(type)
        time_string = time.strftime("%m-%d-%H-%M-%S", time.localtime())
        model_name = self.type + time_string
        fig_name_acc = time_string + '_acc.png'
        fig_name_loss = time_string + '_loss.png'
        self.tainning_acc_fig = os.path.join(self.root_dir, fig_name_acc)
        self.tainning_loss_fig = os.path.join(self.root_dir, fig_name_loss)
        self.model_path = os.path.join(self.root_dir, model_name)

    def continuous_labels(self, known_labels, unknown_labels):
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


    def load_feas(self):
        if os.path.exists(self.known_feas_dir) and os.path.exists(self.known_labels_dir) and os.path.exists(self.unknown_feas_dir) and os.path.exists(self.unknown_labels_dir):
            super().load_feas()
            self.known_labels, self.unknown_labels = self.continuous_labels(self.known_labels, self.unknown_labels)
            self.known_labels = to_categorical(self.known_labels)
            self.unknown_labels = to_categorical(self.unknown_labels)
        else:
            raise RuntimeError('无法获得特征文件, 使用classified_by_models提取特征.')


    def get_model(self):
        self.model = Sequential()
        if self.type is 'VGGFC':
            self.model.add(Dense(1024, input_dim=4096, activation='relu'))
            self.model.add(Dropout(0.4))
            self.model.add(Dense(276, activation='softmax'))
            sgd = optimizers.SGD(lr=0.001, decay=1e-4, momentum=0.9, nesterov=True)
            self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            self.model.summary()
        elif self.type is 'GoogLeNetFC':
            self.model.add(Dense(512, input_dim=2048, activation='relu'))
            self.model.add(Dense(276, activation='softmax'))
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            self.model.summary()


    def train_model(self):
        if self.type is 'GoogLeNetFC':
            self.history = self.model.fit(x=self.known_feas, y=self.known_labels, 
                batch_size=256, epochs=15, validation_split=0.1, shuffle=True)
        elif self.type is 'VGGFC':
            self.history = self.model.fit(x=self.known_feas, y=self.known_labels, 
                epochs=20, batch_size=8, validation_split=0.1, shuffle=True)


    def classifiy(self):
        self.scores = self.model.evaluate(x=self.unknown_feas, y=self.unknown_labels, batch_size=32, verbose=1)


    def evaluting(self):
        print("图像分类的%s: %.2f%%" % (self.model.metrics_names[1], self.scores[1]*100))
        if self.scores[1] >= 0.99:
            self.model.save(self.model_path)


    def load_model(self, model_weight):
        self.model = load_model(model_weight)


    def num_2_percent(self, temp, pos):
        return '%1.00f' % (temp*100) + '%'

    def train_acc_info(self):
        x = np.arange(1, len(self.history.history['accuracy'])+1)
        y = np.arange(11)
        y = y * 0.1
        plt.plot(x, self.history.history['accuracy'], color='r')
        plt.plot(x, self.history.history['val_accuracy'], color='b',linestyle='--')
        plt.legend(['train', 'validation'])
        plt.grid(True)
        plt.grid(color='r', alpha=0.4, linestyle='--')
        plt.title('Model Accuracy', fontsize=15)
        plt.xlabel('Epoch', fontsize=15)
        plt.ylabel('Accuracy', fontsize=15)
        plt.yticks(y)
        plt.xticks(x)
        plt.gca().yaxis.set_major_formatter(FuncFormatter(self.num_2_percent))
        
        plt.show()
        plt.savefig(self.tainning_acc_fig)
        plt.close()


    def train_loss_info(self):
        x = np.arange(1, len(self.history.history['loss'])+1)
        plt.plot(x, self.history.history['loss'], color='r')
        plt.plot(x, self.history.history['val_loss'], color='b',linestyle='--')
        plt.legend(['train', 'validation'])
        plt.grid(True)
        plt.grid(color='r', alpha=0.4, linestyle='--')
        plt.title('Model Loss', fontsize=15)
        plt.xlabel('Epoch', fontsize=15)
        plt.ylabel('Loss', fontsize=15)
        plt.xticks(x)
        plt.show() 
        plt.savefig(self.tainning_loss_fig)
        plt.close()


    def train_info(self):
        self.train_acc_info()
        self.train_loss_info()


def train_and_test_GoogLeNetFC():
    fc = ClassifiedByFC('GoogLeNetFC')
    fc.get_model()
    fc.load_feas()
    fc.train_model()
    fc.train_info()
    fc.classifiy()
    fc.evaluting()


def test_GoogLeNetFC():
    fc = ClassifiedByFC('GoogLeNetFC')
    fc.load_model()
    fc.load_feas()
    fc.classifiy()
    fc.evaluting()


def train_and_test_VGGFC():
    fc = ClassifiedByFC('VGGFC')
    fc.get_model()
    fc.load_feas()
    fc.train_model()
    fc.train_info()
    fc.classifiy()
    fc.evaluting()


def test_VGGFC():
    fc = ClassifiedByFC('VGGFC')
    fc.load_model()
    fc.load_feas()
    fc.classifiy()
    fc.evaluting()


if __name__ == '__main__':
    '''
    start = time.time()
    train_and_test_GoogLeNetFC()
    end = time.time()
    used_time = int(end - start)
    print('GoogLeNetFC使用时间是 %d 分, %d 秒.'%(used_time/60, used_time%60))
'''
    start = time.time()
    train_and_test_VGGFC()
    end = time.time()
    used_time = int(end - start)
    print('VGGFC使用时间是 %d 分, %d 秒.'%(used_time/60, used_time%60))


'''
图像分类的accuracy: 100.00%
GoogLeNetFC使用时间是 0 分, 26 秒.

图像分类的accuracy: 100.00%
VGGFC使用时间是 0 分, 57秒.
'''