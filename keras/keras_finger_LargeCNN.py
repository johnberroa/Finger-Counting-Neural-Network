#
#    LARGE CNN (not really large, but that's how I named it)
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
BATCHSIZE = 100
print("BATCHSIZE:", BATCHSIZE)

if RESIZE:
	size = (200, 200)
else:
	size = (300, 300)


train_datagen = ImageDataGenerator( rescale=1./255,
                                    rotation_range=10.,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    zca_whitening=True
                                  )

valid_datagen  = ImageDataGenerator(rescale=1./255, zca_whitening=True)  # maybe no whitening on testing sets?

train_gen = train_datagen.flow_from_directory(
        'data/train/',
        target_size=size,
        color_mode='color',
        batch_size=BATCHSIZE,
        classes=['1','2','3','4','5'],
        class_mode='categorical',
        save_to_dir='augmented',  # SAVING THE IMAGES TO VISUALIZE WHAT THE AUGMENTATION IS DOING
        save_prefix='AUG',
		save_format="jpeg"
    )

valid_gen = test_datagen.flow_from_directory(
        'data/val/',
        target_size=size,
        color_mode='color',
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
# Only 3 Conv sections because max pool would make the images too small for any useful analysis
model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=input_dims, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2)))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2)))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2)))
model.add(Conv2D(16, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(512, activation='relu', activity_regularizer=regularizers.l1(0.01)))
model.add(Dropout(.5))
model.add(Dense(128, activation='relu', activity_regularizer=regularizers.l1(0.01)))
model.add(Dropout(.5))
model.add(Dense(5, activation='softmax', activity_regularizer=regularizers.l1(0.01)))

model.compile(optimizer='adam',
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])

#one_hots = to_categorical(labels-1, num_classes=5)  # due to out of bounds error, must be labels - 1 to make it start at 0
#one_hots_v = to_categorical(v_labels-1, num_classes=5)

#hist = model.fit(train, one_hots, epochs=EPOCHS, batch_size=BATCHSIZE, validation_data=(validation, one_hots_v), verbose=2)
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
np.save('tLb{}-acc.npy'.format(BATCHSIZE), hist.history['acc'])
np.save('tLb{}-loss.npy'.format(BATCHSIZE), hist.history['loss'])
np.save('vLb{}-acc.npy'.format(BATCHSIZE), hist.history['val_acc'])
np.save('vLb{}-loss.npy'.format(BATCHSIZE), hist.history['val_loss'])
print("FINAL ACCURACY:".format(BATCHSIZE), hist.history['acc'][-1])
