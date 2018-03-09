import os, cv2, sys, shutil
import numpy as np
import matplotlib.pyplot as plt

from shutil import copyfile as copy
from sklearn.model_selection import train_test_split as split_data


class FingerData:
	def __init__(self, directory="./"):
		"""
		Initializes the data holding class and loads data from given directory.
		Args:
			directory: Directory of images.
		"""
		self._directory = directory

		counter = self._get_image_count()
		self._images = np.zeros((counter, 300, 300, 3))
		self._labels = np.zeros(counter)

		self._load_images()
		self._split()  # split into training, validation, and test

	def _get_image_count(self):
		"""
		Counts total number of images collected.
		Returns:
			counter: Number of images found.
		"""
		counter = 0
		for number in range(5):
			number += 1  # to get it in range 1-5 not 0-4
			try:  # try to find folder
				os.chdir('./' + str(number))
			except:
				continue
			contents = os.listdir(self._directory)
			if contents == []:  # if folder is empty
				os.chdir('..')
				continue
			counter += len(contents)
			os.chdir('..')
		return counter

	def _load_images(self):
		"""
		Loads images and labels by stepping through folder structure and loading in images to the image array.
		"""
		index = 0
		for number in range(5):
			number += 1  # to get it in range 1-5 not 0-4
			try:  # try to find folder
				os.chdir('./' + str(number))
			except:
				raise FileNotFoundError("No data exists to load")
			contents = os.listdir(self._directory)
			if contents == []:  # if folder is empty
				print("No data for", str(number), '\nTraining highly not recommended!')
				os.chdir('..')
				continue
			for entry in contents:
				img = cv2.imread(entry)
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				self._images[index] = img
				self._labels[index] = int(entry[0])
				index += 1
			os.chdir('..')
		print("Loaded {} images.".format(index))

	def _split(self):
		"""
		Splits images and labels into training, validation, and test sets.
		"""
		self._train_imgs, valtest, self._train_labels, valtest_labels = split_data(self._images, self._labels,
																				   shuffle=True, train_size=.6,
																				   test_size=.4)
		self._valid_imgs, self._test_imgs, self._valid_labels, self._test_labels = split_data(valtest, valtest_labels,
																							  shuffle=True,
																							  train_size=.5,
																							  test_size=.5)

	def get_training_batch(self, batchsize):
		"""
		Creates a training batch object to iterate through.
		Args:
			batchsize: How big the batch should be.

		Returns:
			Iterable training batch.
		"""
		return self._get_batch(self._train_imgs, self._train_labels, batchsize)

	def get_validation_batch(self, batchsize):
		"""
		Creates a validation batch object to iterate through.
		Args:
			batchsize: How big the batch should be.

		Returns:
			Iterable validation batch.
		"""
		return self._get_batch(self._valid_imgs, self._valid_labels, batchsize)

	def get_test_batch(self, batchsize):
		"""
		Creates a test batch object to iterate through.
		Args:
			batchsize: How big the batch should be.

		Returns:
			Iterable test batch.
		"""
		return self._get_batch(self._test_imgs, self._test_labels, batchsize)

	def _get_batch(self, images, labels, batchsize):
		"""
		Creates a yielded batch object of the data set inputted.
		Args:
			images: The images to have batched.
			labels: The labels to have batched.
			batchsize: How big the batch should be.

		Returns:
			Iterable batch.
		"""
		count = labels.shape[0]

		if batchsize <= 0:  # take all data (batch learning)
			batchsize = count

		random_indices = np.random.choice(count, count, replace=False)
		images = images[random_indices]
		labels = labels[random_indices]
		for i in range(count // batchsize):
			start = i * batchsize
			end = start + batchsize
			yield images[start:end], labels[start:end]

	def get_sizes(self):
		"""
		Returns the size of the different data sets.
		Returns:
			training_samples_n: Number of training samples.
			validation_samples_n: Number of validation samples.
			test_samples_n: Number of test samples.
		"""
		training_samples_n = self._train_labels.shape[0]
		validation_samples_n = self._valid_labels.shape[0]
		test_samples_n = self._test_labels.shape[0]
		return training_samples_n, validation_samples_n, test_samples_n


class FingerDataBatch:
	def __init__(self, batchsize, directory="./"):
		"""
		Initializes the data holding class and loads data batch by batch from given directory.
		Args:
			batchsize: The number of images to load per batch
			directory: Directory of images.
		"""
		self.batchsize = batchsize
		self._directory = directory

		self._images = np.zeros((self.batchsize, 300, 300, 3))
		self._labels = np.zeros(self.batchsize)

		self._collect_images()
		self.indicies = self._get_image_indicies()

		# There is no guarantee the images will remain in proper order so we generate labels AFTER generating the batch
		self._split()  # split into training, validation, and test

	def _get_image_count(self):
		"""
		Counts total number of images (entries) in a folder.
		Returns:
			Number of images found.
		"""
		return len(os.listdir(self._directory))

	def _collect_images(self):
		"""
		Copies all images into a temporary folder so that they are all together
		"""
		print("Collecting all data into one directory 'tmp'...")
		directories = ['1','2','3','4','5']
		try: # make temporary directory for all images to reside
			os.mkdir('./tmp')
		except:
			print("Could not make 'tmp' folder.  Fatal error, quitting...")
			sys.exit(0)
		# copy all images into new temp folder
		for directory in directories:
			try:  # try to find folder
				os.chdir(self._directory + directory)
			except:
				raise FileNotFoundError("No data exists to load")
			contents = os.listdir(self._directory) # this works because we changed directories
			if contents == []:  # if folder is empty
				print("No data for", str(number), '\nTraining highly not recommended!')
				os.chdir('..')
				continue
			for entry in contents:
				copy(entry, '../tmp/' + entry)
			os.chdir('..')
		print("Data collected...")

	def _get_image_indicies(self):
		"""
		Creates an array of indicies to later generate train/val/test splits without loading the data
		Returns:
			indicies: An array of increasing numbers with length of number of images
		"""
		try:  # try to find folder
			os.chdir('./tmp')
		except:
			raise FileNotFoundError("Data has not been collected.  Run _collect_images first!")
		count = self._get_image_count()
		indicies = np.arange(0, count)
		return indicies

	def _split(self):
		"""
		Splits images and labels into training, validation, and test sets.
		"""
		self._train_indicies, valtest = split_data(self.indicies, shuffle=True, train_size=.6, test_size=.4)
		self._valid_indicies, self._test_indicies = split_data(valtest, shuffle=True, train_size=.5, test_size=.5)

	def get_training_batch(self):
		"""
		Creates a training batch via _get_batch random indicies.
		Returns:
			Training batch.
		"""
		return self._get_batch(self._train_indicies, 'Train')

	def get_validation_batch(self):
		"""
		Creates a validation batch via _get_batch random indicies.
		Returns:
			Validation batch.
		"""
		return self._get_batch(self._valid_indicies, 'Validation')

	def get_test_batch(self):
		"""
		Creates a test batch via _get_batch random indicies.
		Returns:
			Test batch.
		"""
		return self._get_batch(self._test_indicies, 'Test')

	def _get_batch(self, dataset, name):
		"""
		Creates a a batch of the indicies inputted.  Not as robust as the old method!
		Args:
			dataset: The images to have batched.
			name: Training, valid, or test for the print statement

		Returns:
			Batch.
		"""
		random_indicies = np.random.choice(dataset, self.batchsize, replace=False)
		
		# try:
		# 	os.chdir(self._directory+'/tmp')
		# except:
		# 	print("'tmp' folder doesn't exist!")
		contents = os.listdir(self._directory) # works cause working directory changed
		if contents == []:  # if folder is empty
				print("No data in 'tmp'!")
				sys.exit(0)
		for i, index in enumerate(random_indicies):
			img = cv2.imread(contents[index])
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			self._images[i] = img
			self._labels[i] = int(contents[index][0])
		print("{} batch loaded ({} images)".format(name, self.batchsize))
		return self._images, self._labels

	def get_sizes(self):
		"""
		Returns the size of the different data sets.
		Returns:
			training_samples_n: Number of training samples.
			validation_samples_n: Number of validation samples.
			test_samples_n: Number of test samples.
		"""
		training_samples_n = self._train_indicies.shape[0]
		validation_samples_n = self._valid_indicies.shape[0]
		test_samples_n = self._test_indicies.shape[0]
		return training_samples_n, validation_samples_n, test_samples_n

	def delete_temp_files(self):
		"""
		Deletes the temporary folder
		"""
		shutil.rmtree(self._directory+'./tmp')
		print("'tmp' folder removed.")


if __name__ == '__main__':
	# For debugging purposes
	loader = FingerDataBatch(10)
	print(loader.get_sizes())
