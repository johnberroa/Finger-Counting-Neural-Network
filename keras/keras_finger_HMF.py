#
#    'How many fingers' CNN
#

import os, time
import numpy as np
import cv2

from sklearn.model_selection import train_test_split as split_data
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Flatten, MaxPool2D, Dense, Dropout, BatchNormalization
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


# Boolean to tell the program to resize the images to a smaller resolution
RESIZE = False
print("RESIZE to 200", RESIZE)
print("MORE LAYERS")
# Hyperparameters at top so that they are easily changed in vim
EPOCHS = 200
BATCHSIZE = 128
print("BATCHSIZE:", BATCHSIZE)

if RESIZE:
    size = (200, 200)
else:
    size = (300, 300)


def feature_extract(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (7,7), 3)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, new = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return new


train_datagen = ImageDataGenerator( rescale=1./255,
                                    rotation_range=10.,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    zca_whitening=True,
                                    preprocessing_function=feature_extract
                                  )

valid_datagen  = ImageDataGenerator(rescale=1./255, zca_whitening=True)  # maybe no whitening on testing sets?

train_gen = train_datagen.flow_from_directory(
        'data/train/',
        target_size=size,
        color_mode='rgb',
        batch_size=BATCHSIZE,
        classes=['1','2','3','4','5'],
        class_mode='categorical'
    )

valid_gen = valid_datagen.flow_from_directory(
        'data/val/',
        target_size=size,
        color_mode='rgb',
        batch_size=BATCHSIZE,
        classes=['1','2','3','4','5'],
        class_mode='categorical'
    )



# Create proper input dimensions
if RESIZE:
    input_dims = (200, 200, 3)
else:
    input_dims = (300, 300, 3)


# Creation of the model
model = Sequential()
# model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_dims))
# model.add(MaxPool2D((2,2)))
# model.add(Conv2D(64, (3,3), activation='relu'))
# model.add(MaxPool2D((2,2)))
# model.add(Conv2D(128, (3,3), activation='relu'))
# model.add(MaxPool2D((2,2)))
# model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPool2D((10,10)))
model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['acc'])

callbacks_list = [ModelCheckpoint(filepath='weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', save_best_only=True)]

hist = model.fit_generator(
        train_gen,
        steps_per_epoch=200//BATCHSIZE,  # len(train) / batchsize
        epochs=EPOCHS,
        validation_data=valid_gen,
        validation_steps=100//BATCHSIZE, # same as above
        callbacks=callbacks_list,
        verbose=2
    )


# Save the training data into npy save files so that they can be plotted separately
np.save('HMF-tb{}-acc.npy'.format(BATCHSIZE), hist.history['acc'])
np.save('HMF-tb{}-loss.npy'.format(BATCHSIZE), hist.history['loss'])
np.save('HMF-vb{}-acc.npy'.format(BATCHSIZE), hist.history['val_acc'])
np.save('HMF-vb{}-loss.npy'.format(BATCHSIZE), hist.history['val_loss'])
print("FINAL ACCURACY:", hist.history['acc'][-1])
