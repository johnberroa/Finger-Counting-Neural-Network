import os, shutil

backup_directory = "./FINGER DATA BACKUP"
directories = ['1', '2', '3', '4', '5']

def create_backup():
	"""
	Creates backup folder if doesn't exist, and moves all data into it.
	"""
	if not os.path.exists(backup_directory):
		print("Creating backup folder as 'FINGER DATA BACKUP'...")
		os.makedirs(backup_directory)
		for directory in directories:
			print("Backing up data for label '{}'...".format(directory))
			shutil.copytree('./'+directory, backup_directory+'/'+directory)
		print("Backup creation complete!")
	else:
		print("Backup already exists.  If data is missing, it must be manually moved to the backup.")


def restore_backup():
	"""
	Deletes old data folders, and recreates them from the backup folder.
	"""
	for directory in directories:
		shutil.rmtree('./'+directory)
	for directory in directories:
		print("Restoring data for label '{}'...".format(directory))
		shutil.copytree(backup_directory+'/'+directory, './'+directory)
	print("Data restoration complete!")


def backup_data():
	"""
	Moves into the backup folder, deletes all data, and then recopies the current data into it.
	"""
	try:
		os.chdir(backup_directory)
	except:
		print("Backup folder does not exist!")
	for directory in directories:
		shutil.rmtree('./'+directory)
	os.chdir('..')
	for directory in directories:
		print("Backing up data for label '{}'...".format(directory))
		shutil.copytree('./'+directory, backup_directory+'/'+directory)
	print("Backup complete!")


def get_checkpoint():
	"""
	Returns the average	number of files within the folders, so that a reference for future backups can be maintained.

	Returns:
	checkpoint: average number of datapoints in folders
	"""
	import numpy as np

	checkpoint = []
	for directory in directories:
		try: # try to find folder
			os.chdir('./'+directory)
		except:
			continue
		contents = os.listdir('./')
		if contents == []: # if folder is empty
			print("No data for", directory)
			os.chdir('..')
			continue
		counter = []
		for entry in contents:
			entry = entry.split('.')
			num = entry[0][2:]
			try:  # excludes files that aren't of type x-y.jpg
				num = int(num)
				counter.append(num)
			except:
				continue
		checkpoint.append(max(counter))
		os.chdir('..')
	checkpoint = np.mean(checkpoint)
	return checkpoint


if __name__ == "__main__":
	selection = int(input("Enter desired function...\n[0]: Quit\n[1]: Backup\n[2]: Restore backup\n[3]: Create backup\n..."))
	if selection == 0:
		print("Quitting...")
	elif selection == 1:
		backup_data()
	elif selection == 2:
		restore_backup()
	elif selection == 3:
		create_backup()
	else:
		print("Invalid option!")