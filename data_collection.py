import numpy as np
import cv2
import os


def crop_box(img, off_x, off_y):
    """
    Cuts out a (300,300) box (where the hand should be) for further processing.
    Args:
        img: Image to be cropped.
        off_x: Offset of the x dimension to start the crop.
        off_y: Offset of the y dimension to start the crop.

    Returns:
        (300,300) cropped image.
    """
    cropped = img[off_x:off_x + 300, off_y:off_y + 300]
    return cropped


def get_counts():
    """
    Gets the highest number for each datapoint and returns it into a new count dictionary.
    Returns:
        counts: Number of datapoints per number.
    """
    counts = {"1":0, "2":0, "3":0, "4":0, "5":0}
    for number in range(5):
        number += 1 # to get it in range 1-5 not 0-4
        try: # try to find folder
            os.chdir('./'+str(number))
        except:
            continue
        contents = os.listdir()
        if contents == []: # if folder is empty
            print("No data for", str(number))
            os.chdir('..')
            continue
        counter = []
        for entry in contents:
            entry = entry.split('.')
            num = entry[0][2:]
            try:  # excludes files that aren't of type x-y.jpg
                num = int(num)
                counter.append(int(num))
            except:
                continue
        counts[str(number)] = max(counter)
        os.chdir('..')
    print("Current database:", counts)
    return counts


def order_file_structure():
    """
    Moves saved images into respective folders for a cleaner file structure.
    """
    directories = ['1','2','3','4','5']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
        contents = os.listdir()
        for image in contents:
            if image[:2] == directory+'-' and image[-3:] == 'jpg': # make sure it's an image and not a folder with proper label
                os.rename("./"+image, "./"+directory+"./"+image) # move image to its respective folder


def capture_image_data(runs, counts=get_counts()):
    """
    Continually captures new images and labels to create the dataset.  Loops "runs" amount of times.
    Args:
        runs: Number of loops/images to create.
        counts: Current number of data per number collected, so data collection can pick up again at a future point.
    """

    def create_cam(on=False, cam=None):
        """Creates a camera object if none exists, and/or just returns the frame"""
        if not on:
            cam = cv2.VideoCapture(0)
            ret, frame = cam.read()
            return cam, ret, frame
        else:
            ret, frame = cam.read()
            return cam, ret, frame

    quit = False
    print("Press 'space' to capture image...")
    x0 = y0 = 100  # initial x and y rectangle crop offset
    label = input("Enter initial label to begin collecting data...") or "UNLABELED"
    assert 0 <= int(label) <= 5, "INVALID LABEL (1-5, input{})".format(label)
    for run in range(runs):
        if quit:  # ends loop
            break
        total = 1
        ret = cam = False
        capture = True
        try:
            while capture:
                cam, ret, frame = create_cam(ret, cam)
                frame = cv2.flip(frame, 1)  # mirror so that left is left
                key = cv2.waitKey(1) & 0xff
                if key == ord(' '):
                    # Take picture
                    assert frame is not None, "Camera did not work!"  # error handling
                    held_frame = frame  # just in case frames continually are recorded while the user is typing
                    counts[label] += 1
                    cv2.imwrite(label + '-' + str(counts[label]) + '.jpg', crop_box(held_frame, x0, y0))
                    capture = False
                elif key == ord("1"):
                    label = '1'
                    print("Collecting data for label: {}".format(label))
                elif key == ord("2"):
                    label = '2'
                    print("Collecting data for label: {}".format(label))
                elif key == ord("3"):
                    label = '3'
                    print("Collecting data for label: {}".format(label))
                elif key == ord("4"):
                    label = '4'
                    print("Collecting data for label: {}".format(label))
                elif key == ord("5"):
                    label = '5'
                    print("Collecting data for label: {}".format(label))
                elif key == ord("q"):
                    print("Quitting data capture...")
                    cam.release()
                    cv2.destroyAllWindows()
                    capture = False
                    quit = True
                    break
                else:
                    # Draw rectangle so user knows where to put hand
                    cv2.rectangle(frame, (x0, y0), (x0 + 300, y0 + 300), [0, 0, 255], 12)
                    cv2.putText(frame, "Collecting label: {}".format(label), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (0, 0, 255))
                    if ret == True:
                        cv2.imshow("Window", frame)
                    continue
        except Exception as e:
            cam.release()
            cv2.destroyAllWindows()
            raise e
        total += 1
    cam.release()
    cv2.destroyAllWindows()
    print("Data collection completed ({} images of {} recorded)".format(total, runs))
    print("Database:", counts)



if __name__ == "__main__":
    n = input("How many images would you like to collect?")
    capture_image_data(int(n))
    order_file_structure()
