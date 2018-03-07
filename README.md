# Finger Counting Neural Network
A ResNet architecture neural network that counts the number of fingers held up in front of a webcam.  

*This network is currently a work in progress.  Data collection is essentially compelte.  Training is in progress.*

## Usage
* ``data_collection.py``: The python script to create a dataset of finger images.
* ``data_loader.py``: Loads the data into memory and creates batches for training.
* ``finger_resnet.py``: The neural network model to learn the task.

## Script Specifics

### Data Collection
``data_collection.py`` will capture any number of images, label them, and create a file structure to store them.  You can take any number of images at any time, and change which label you want to put on the images, on the fly.  Pressing ``q`` will quit if necessary.  Press ``h`` for help.  After creating the images, the script will move each image into its respective folder (named after the images' label).

The file structure used is five folders, each named after their respective label, placed in the root directory.  This structure was chosen because it allows easy access to check the data quickly during acquisition (i.e., no need to go click more than necessary).

### Data Loader
``data_loader.py`` will step through the folder structure and load each image into memory.  It will then split it into train (60%), validation (20%), and test (20%) datasets.  It also contains a method to create a generator object to step through training/validation/test batches.

### Finger ResNet
``finger_resnet.py`` is the ResNet implementation that is trained on the dataset.  It has the ability to pick up training from where it left off.  Structure will follow once it is finalized (will be based on a tradeoff between performance and speed).


