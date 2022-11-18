"""
This script will be used to remove noisy background from cars images to
improve the quality of our data and get a better model.
The main idea is to use a vehicle detector to extract the car
from the picture, getting rid of all the background, which may cause
confusion to our CNN model.
We must create a new folder to store this new dataset, following exactly the
same directory structure with its subfolders but with new images.
"""

import argparse
import os
#from tensorflow.keras.utils import load_img
from utils.utils import walkdir
from utils.detection import get_vehicle_coordinates
import cv2
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "data_folder",
        type=str,
        help=(
            "Full path to the directory having all the cars images. Already "
            "splitted in train/test sets. E.g. "
            "`/home/app/src/data/car_ims_v1/`."
        ),
    )
    parser.add_argument(
        "output_data_folder",
        type=str,
        help=(
            "Full path to the directory in which we will store the resulting "
            "cropped pictures. E.g. `/home/app/src/data/car_ims_v2/`."
        ),
    )

    args = parser.parse_args()

    return args


def main(data_folder, output_data_folder):
    """
    Parameters
    ----------
    data_folder : str
        Full path to train/test images folder.

    output_data_folder : str
        Full path to the directory in which we will store the resulting
        cropped images.
    """
    # For this function, you must:
    #   1. Iterate over each image in `data_folder`, you can
    #      use Python `os.walk()` or `utils.waldir()``
    #   2. Load the image
    #   3. Run the detector and get the vehicle coordinates, use
    #      utils.detection.get_vehicle_coordinates() for this task
    #   4. Extract the car from the image and store it in
    #      `output_data_folder` with the same image name. You may also need
    #      to create additional subfolders following the original
    #      `data_folder` structure.
    # TODO
    #1.Iterate over each image in `data_folder`
    print(data_folder)
    for (dirpath, file_name) in walkdir(data_folder): # Iterate over each image in `data_folder`
        full_path = os.path.join(dirpath, file_name)
        print(full_path)
        #2 Use keras.utils.load_img() to load image
        image = cv2.imread(full_path)
        #3 Run the detector and get the vehicle coordinates
        box_coorinates = get_vehicle_coordinates(image)
        #4 Crop the image
        cropped_image= image[box_coorinates[1]:box_coorinates[3], box_coorinates[0]:box_coorinates[2],:]
        #5 Get label
        pre_label = os.path.dirname(full_path)
        label = os.path.basename(pre_label)
        #6 Create train and test folder directories
        train_folder = os.path.join(output_data_folder, "train")
        test_folder = os.path.join(output_data_folder, "test")
        #7 Store images in the created folders
        if 'train' in full_path:
            cropped_train_path = os.path.join(train_folder, label)
            if not os.path.isdir(cropped_train_path):
                os.makedirs(cropped_train_path)
            cropped_train_img_path = os.path.join(cropped_train_path, file_name)
            cv2.imwrite(cropped_train_img_path, cropped_image)
        if 'test' in full_path:
            cropped_test_path = os.path.join(test_folder, label)
            if not os.path.isdir(cropped_test_path):
                os.makedirs(cropped_test_path)
            cropped_test_img_path = os.path.join(cropped_test_path, file_name)
            cv2.imwrite(cropped_test_img_path, cropped_image)
        time.sleep(0.1) # Time gap necesary to be able to acces image file in folder

        
    

if __name__ == "__main__":
    args = parse_args()
    main(args.data_folder, args.output_data_folder)
