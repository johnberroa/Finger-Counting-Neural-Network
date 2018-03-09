#
#    'How many fingers' CNN
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
# Hyperparameters as per 'How Many Fingers'
# https://github.com/jgv7/CNN-HowManyFingers/blob/master/trainModel.ipynb
EPOCHS = 500
BATCHSIZE = 128


def load_data(resize):
	"""
	Data loading from the tmp folder
	"""
	start = time.time()
	if resize:
		images = np.zeros((2110 * 2, 50, 50, 3))
	else:
		images = np.zeros((2110 * 2, 300, 300, 3))
	labels = np.zeros((2110 * 2, 1))

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
	images, labels = preprocess_data(images, labels)
	end = time.time()
	print("Data loaded. ({} mins)".format((end - start) / 60))
	return images, labels


def preprocess_data(imgs, lbls):
	imgs = imgs / 255 # normalize between 0 and 1
	# Flip all images (they will be randomly shuffled when split)
	for i in range(2110):
		imgs[i + 2110] = np.fliplr(imgs[i])
		lbls[i + 2110] = lbls[i]
	return imgs, lbls


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
model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_dims))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['acc'])

one_hots = to_categorical(labels-1, num_classes=5)  # due to out of bounds error, must be labels - 1 to make it start at 0
one_hots_v = to_categorical(v_labels-1, num_classes=5)

hist = model.fit(train, one_hots, epochs=EPOCHS, batch_size=BATCHSIZE, validation_data=(validation, one_hots_v), verbose=0)

# Save the training data into npy save files so that they can be plotted separately
np.save('HMF-t-acc.npy', hist.history['acc'])
np.save('HMF-t-loss.npy', hist.history['loss'])
np.save('HMF-v-acc.npy', hist.history['val_acc'])
np.save('HMF-v-loss.npy', hist.history['val_loss'])
print("FINAL ACCURACY:", hist.history['acc'][-1])
