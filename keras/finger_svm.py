import matplotlib.pyplot as plt
import time, os, cv2
import numpy as np

from sklearn.model_selection import train_test_split as split_data
from sklearn import svm, metrics


# Boolean to tell the program to resize the images to a smaller resolution
RESIZE = False
print("FULL SIZE IMAGES!!")


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

# Flatten to (samples, feature) array:
n_train = len(train)
n_valid = len(validation)
n_test = len(test)
train = train.reshape((n_train, -1))
validation = validation.reshape((n_valid, -1))
test = test.reshape((n_test, -1))

labels = np.ravel(labels)
v_labels = np.ravel(v_labels)

# Create classifier
print(".001 gamma")
SVM = svm.SVC(gamma=.001)

# Train on the training set
SVM.fit(train, labels)

# Validation test
score = SVM.score(validation, v_labels)

print("VALIDATION ACCURACY:", score)
