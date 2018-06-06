#https://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/
from keras.models import Sequential, model_from_json
from keras.layers import Dense, MaxPooling2D, Flatten, Dropout, Conv2D
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import toimage
from keras import backend as K
K.set_image_dim_ordering('th')


#Loading
np.random.seed(7)
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train, X_test = X_train/255.0, X_test/255.0
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


#Pre Processing
def print_shapes():
    print("X_train",X_train.shape)
    print("X_test", X_test.shape)
    print("y_train", y_train.shape)
    print("y_test", y_test.shape)
def plotimgs():
    for i in range(9,9+9):
        plt.subplot(330+i-9+1)
        plt.imshow(toimage(X_train[i]))
    plt.show()
    plt.imshow(toimage(X_train[9]))
    plt.show()


#Learning
def cnn_ver1():
    model = Sequential()
    # conv2d->dropout->conv2d->maxpool->flatten->dense->dropout->output(dense)

    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', input_shape=(3,32,32),
                     activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same',
                     activation='relu', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())

    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    # Compile model
    epochs = 250
    lrate = 0.01
    decay = lrate / epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())
    return model


#post-processing & conclusions
def save_model(model):
    model_json = model.to_json()
    with open("model-objectrec.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model-objectrec.h5")
    print("Saved model to disk")


def load_json_module():
    # load json and create model
    json_file = open('model-objectrec.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model-objectrec.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return loaded_model

#Main functions
def main():
    #print_shapes()
    #plotimgs()

    model = cnn_ver1()
    model.fit(X_train, y_train, batch_size=64, epochs=10)
    save_model(model)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Baseline Error: %.2f%%" % (100-scores[1]*100))


if __name__ == '__main__':
    main()