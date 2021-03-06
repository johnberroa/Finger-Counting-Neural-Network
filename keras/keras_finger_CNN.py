#
#    SIMPLE CNN
#

import os, time
import numpy as np
import cv2

from sklearn.model_selection import train_test_split as split_data
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Flatten, MaxPool2D, Dense, Dropout
from tensorflow.python.keras.utils import to_categorical


# Boolean to tell the program to resize the images to a smaller resolution
RESIZE = False
print("RESIZE:", RESIZE)
# Hyperparameters at top so that they are easily changed in vim
EPOCHS = 500
BATCHSIZE = 32


def load_data(resize):
	"""
	Data loading from the tmp folder
	"""
	start = time.time()
	if resize:
		images = np.zeros((2110, 50, 50, 3))
	else:
		images = np.zeros((2110, 300, 300, 3))
	labels = np.zeros((2110, 1))

	print("Loading data...")
	os.chdir('./tmp')
	contents = os.listdir('./')
	for i, entry in enumerate(contents):
		img = cv2.imread(entry)
		if resize:
			images[i] = cv2.resize(img, (50,50))
		else:
			images[i] = img
		labels[i] = int(entry[0])
	os.chdir('..')
	end = time.time()
	print("Data loaded. ({} mins)".format((end - start) / 60))
	return images, labels


def split(images, labels):
	"""
	Splits images and labels into training, validation, and test sets.
	"""
	train, valtest, labels, v_labels = split_data(images, labels,
															shuffle=True, train_size=.6,
															test_size=.4)
	validation, test, v_labels, t_labels = split_data(valtest, v_labels,
															shuffle=True,
															train_size=.5,
															test_size=.5)
	return train, validation, test, labels, v_labels, t_labels


# Load the data and split into sets
images, labels = load_data(RESIZE)
train, validation, test, labels, v_labels, t_labels = split(images, labels)

# Create proper input dimensions
if RESIZE:
	input_dims = (50, 50, 3)
else:
	input_dims = (300, 300, 3)


# Creation of the model
model = Sequential()
model.add(Conv2D(4, (3, 3), input_shape=input_dims, padding='same', activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(8, (3, 3), padding='same', activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='adam',
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])

one_hots = to_categorical(labels-1, num_classes=5)  # due to out of bounds error, must be labels - 1 to make it start at 0
one_hots_v = to_categorical(v_labels-1, num_classes=5)

hist = model.fit(train, one_hots, epochs=EPOCHS, batch_size=BATCHSIZE, validation_data=(validation, one_hots_v), verbose=0)

# Save the training data into npy save files so that they can be plotted separately
np.save('t-acc.npy', hist.history['acc'])
np.save('t-loss.npy', hist.history['loss'])
np.save('v-acc.npy', hist.history['val_acc'])
np.save('v-loss.npy', hist.history['val_loss'])
print("FINAL ACCURACY:", hist.history['acc'][-1])
