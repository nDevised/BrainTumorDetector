import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

def procImg(img, INPUT_SIZE):
    # convert image to RGB format from BGR
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # resize the image to 64x64
    return cv2.resize(imgRGB, (INPUT_SIZE, INPUT_SIZE))


def readAndProcImg(filesList,directory, INPUT_SIZE=64):
    imgs = []
    for i, fname in enumerate(filesList):
        if fname.split('.')[1] == "jpg":
            # this is a jpg image so read it
            img = cv2.imread(directory + fname)
            img = procImg(img, INPUT_SIZE)
            imgs.append(img)
    return imgs

def addAndLabelImgs(imgs,label,dataset,labels):
    for img in imgs:
        dataset.append(np.array(img))
        labels.append(label)

def model(INPUT_SIZE,x_train,y_train,x_test,y_test):
    # Building a sequential model
    model = Sequential()

    # Add the Conv2D layer (1st layer)

    # Conv2D will learn 32 different 3x3 filters from input images, used for edge detection, corners,
    # and simple patterns
    model.add(Conv2D(32, (3,3), input_shape=(INPUT_SIZE,INPUT_SIZE,3)))

    # introduce non-linearity so it detects more features
    model.add(Activation('relu'))

    # Use max pooling size to take the max of every 2x2 area, reducing the inputsize to 32 x 32
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Add the Conv2D layer (2nd Layer)
    model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Add the Conv2D layer
    model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Flatten (Convert to 1d vector)
    model.add(Flatten())

    # Layer with 64 neurons to learn fully flattened features
    model.add(Dense(64))

    # Non-linearity
    model.add(Activation('relu'))

    # Randomly deactivates neurons during training to prevent overfitting
    model.add(Dropout(0.5))

    # Output layer (since output is binary, we only need neuron).
    # If we were using Categorical Cross Entropy, we would need to use the number of classes which is 2 and
    # softmax activation func.

    # But here we are using Binary Cross Entropy
    model.add(Dense(1))

    # Since the model is used for binary classification, use sigmoid activation function
    model.add(Activation('sigmoid'))

    # compile the model and train it
    model.compile(loss = "binary_crossentropy", optimizer='adam',metrics=['accuracy'])

    model.fit(x_train,y_train,batch_size=16,verbose=True,epochs=10,validation_data=(x_test,y_test),shuffle=False)

    model.save('BT10Ep.h5')

def main():
    INPUT_SIZE = 64
    dataset=[]
    labels=[]

    imgPath = 'data/'

    noTumorImg = os.listdir(imgPath + 'no/')
    yesTumorImg = os.listdir(imgPath + 'yes/')
    readAndProcImg(noTumorImg,imgPath+'no/')
    # CHECK if the files are jpg and process them
    noTumorImgs = readAndProcImg(noTumorImg,imgPath + 'no/')
    yesTumorImgs = readAndProcImg(yesTumorImg,imgPath + 'yes/')

    # Label the non and yes tumor imgs
    addAndLabelImgs(noTumorImgs, 0, dataset, labels)
    addAndLabelImgs(yesTumorImgs, 1, dataset, labels)

    # convert the dataset to a np array
    dataset = np.array(dataset)
    labels = np.array(labels)

    # divide the data into training data and testing
    # the shape is (number of images, width, height, color channels)
    # x is the data, and y is the labels for the data (i.e. no tumors or tumors)
    x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=0)

    # normalize the data
    x_train = normalize(x_train, axis=1)
    x_test = normalize(x_test, axis=1)

    # run the model
    model(INPUT_SIZE, x_train, y_train, x_test, y_test)


main()