from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import  numpy as np
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
from sklearn.metrics import confusion_matrix
import itertools


np.random.seed(7)
# load mnist data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# prepare the data
# new_shape = X_train.shape[1]*X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_samples(X_train):
    #plot 4 samples
    plt.subplot('221')
    plt.imshow(X_train[24], cmap=plt.get_cmap('gray'))
    plt.subplot('222')
    plt.imshow(X_train[5], cmap=plt.get_cmap('gray'))
    plt.subplot('223')
    plt.imshow(X_train[9], cmap=plt.get_cmap('gray'))
    plt.subplot('224')
    plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
    plt.show()


#create the model
def create_baseline(input_dim, num_classes):
    model = Sequential()
    model.add(Dense(units=500, activation='relu', input_dim=input_dim))
    #model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))

    #compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',])
    return model

def create_cnn(input_dim, num_classes):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5,5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.2))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',])
    return model

def create_cnn_ver2(num_classes):
    model = Sequential()
    model.add(Conv2D(filters=30, kernel_size=(5,5), activation='relu', input_shape=(1,28,28)))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=15, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(rate=0.2))
    model.add(Flatten())

    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=50, activation='relu'))

    model.add(Dense(units=num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', ])
    return model

def save_model(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("Saved model to disk")


def load_json_module():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return loaded_model


def main():
    num_classes = y_test.shape[1]

    print(X_test.shape)
    input_dim = (X_train.shape[0], X_train.shape[1], X_train.shape[2], 32)
    model = load_json_module() # create_cnn_ver2(num_classes)
    #fit the model
    #model.fit(x=X_train, y=y_train, epochs=10, batch_size=200, verbose=2)

    #save_model(model)

    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Baseline Error: %.2f%%" % (100-scores[1]*100))

    y_pred = model.predict(x=X_test)
    y_pred_vals = np.argmax(y_pred, axis=1)
    y_test_vals = np.argmax(y_test, axis=1)
    cnf_mat = confusion_matrix(y_test, y_pred)
    plt.figure()
    plot_confusion_matrix(cnf_mat, classes=list(range(10)))
    plt.show()

    np.abs(np.subtract(y_pred_vals, y_test_vals))

def main1():
    model = load_json_module()
    y_pred = model.predict(x=X_test)
    y_pred_vals = np.argmax(y_pred, axis=1)
    y_test_vals = np.argmax(y_test, axis=1)
    xt = X_test[y_pred_vals != y_test_vals]
    yt = y_test_vals[y_pred_vals != y_test_vals]
    yp = y_pred_vals[y_pred_vals != y_test_vals]
    xt = xt.reshape(xt.shape[0], 28,28)
    for idx, x in enumerate(xt):
        plt.imshow(x,cmap=plt.get_cmap('gray'))
        print("pred={0}\tactual={1}".format(yp[idx], yt[idx]))
        plt.show()

if __name__ == '__main__':
    #main()
    main1()