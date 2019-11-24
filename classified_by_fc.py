import os
import time

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.models import load_model 

from base import base
from pro_utils import get_labels

class ClassifiedByFC(base):
    def __init__(self, type):
        super().__init__(type)
        fig_name_acc = time.strftime("%m-%d-%H-%M-%S", time.localtime()) + '_acc.png'
        fig_name_loss = time.strftime("%m-%d-%H-%M-%S", time.localtime()) + '_loss.png'
        self.tainning_acc_fig = os.path.join(self.root_dir, fig_name_acc)
        self.tainning_loss_fig = os.path.join(self.root_dir, fig_name_loss)
        self.model_path = os.path.join(self.root_dir, 'model_saved')

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
        super().load_feas()
        self.known_labels, self.unknown_labels = self.continuous_labels(self.known_labels, self.unknown_labels)
        self.known_labels = to_categorical(self.known_labels)
        self.unknown_labels = to_categorical(self.unknown_labels)


    def get_model(self):
        self.model = Sequential()
        self.model.add(Dense(512, input_dim=2048, activation='relu'))
        self.model.add(Dense(276, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()


    def train_model(self, epochs=30, batch_size=32, validation_split=0.1):
        self.history = self.model.fit(x=self.known_feas, y=self.known_labels, 
            batch_size=batch_size,epochs=epochs, validation_split=validation_split, shuffle=True)
    

    def classfiy(self):
        self.scores = self.model.evaluate(x=self.unknown_feas, y=self.unknown_labels, batch_size=32, verbose=1)


    def evaluting(self):
        print("图像分类的%s: %.2f%%" % (self.model.metrics_names[1], self.scores[1]*100))
        if self.scores[1] == 1:
            self.model.save(self.model_path)


    def load_model(self):
        self.model = load_model(self.model_path)


    def train_acc_info(self):
        x = np.arange(1, len(self.history.history['accuracy'])+1)
        plt.plot(x, self.history.history['accuracy'])
        plt.plot(x, self.history.history['val_accuracy'])
        plt.legend(['acc', 'val_acc'])
        plt.grid(True)
        plt.grid(color='r', alpha=0.4, linestyle='--')
        plt.title('accuracy')
        plt.show()
        plt.savefig(self.tainning_acc_fig)


    def train_loss_info(self):
        x = np.arange(1, len(self.history.history['loss'])+1)
        plt.plot(x, self.history.history['loss'])
        plt.plot(x, self.history.history['val_loss'])
        plt.legend(['loss', 'val_loss'])
        plt.grid(True)
        plt.grid(color='r', alpha=0.4, linestyle='--')
        plt.title('loss')
        plt.show() 
        plt.savefig(self.tainning_loss_fig)


    def train_info(self):
        self.train_acc_info()
        self.train_loss_info()


def train_and_test_fc():
    fc = ClassifiedByFC('FC')
    fc.get_model()
    fc.load_feas()
    fc.train_model()
    fc.train_info()
    fc.classfiy()
    fc.evaluting()


def test_fc():
    fc = ClassifiedByFC('FC')
    fc.load_model()
    fc.load_feas()
    fc.classfiy()
    fc.evaluting()


if __name__ == '__main__':
    test_fc()

