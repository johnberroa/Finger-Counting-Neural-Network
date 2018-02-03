import os
import numpy as np
import matplotlib.pyplot as plt
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
            contents = os.listdir()
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
            contents = os.listdir()
            if contents == []:  # if folder is empty
                print("No data for", str(number), '\nTraining highly not recommended!')
                os.chdir('..')
                continue
            for entry in contents:
                img = plt.imread(entry)
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
                                                                                   shuffle=True, train_size=.6)
        self._valid_imgs, self._test_imgs, self._valid_labels, self._test_labels = split_data(valtest, valtest_labels,
                                                                                              shuffle=True,
                                                                                              train_size=.5)

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



if __name__ == '__main__':
    # For debugging purposes
    loader = FingerData()
    print(loader.get_sizes())
