import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from pro_utils import get_labels
# kk = np.load('/home/wangling/faceImages/project/FullyCon/known_labels.npy')
# k = to_categorical(kk)
# print(k.shape)
known_labels, unknown_labels = get_labels()
known_labels = to_categorical(known_labels)
unknown_labels = to_categorical(unknown_labels)


def model():
    model = Sequential()
    model.add(Dense(512, input_dim=2048, activation='relu'))
    model.add(Dense(276, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print("hello world")
    model = model()
    known_labels, unknown_labels = get_labels()
    known_labels = to_categorical(known_labels)
    unknown_labels = to_categorical(unknown_labels)

    x = np.load('/home/wangling/faceImages/project/GoogleNet/known_feas.npy')
    x1 = np.load('/home/wangling/faceImages/project/GoogleNet/unknown_feas.npy')
    # x = list(x)
    # print(x)
    print(x[1].shape)
    model.summary()
    history = model.fit(x=x, y=known_labels, epochs=2, validation_split=0.1, shuffle=True)
    print(history.history.keys())
    print(history.history['accuracy'])
    plt.plot(history.history['accuracy'], color='r')
    plt.plot(history.history['val_accuracy'], color='b')
    plt.legend(['acc', 'val_acc'])
    plt.grid(True)
    plt.grid(color='r', alpha=0.4, linestyle='--')
    plt.title('acc')
    plt.show()
    
    scores = model.evaluate(x=x1, y=unknown_labels, batch_size=32, verbose=1)

    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


