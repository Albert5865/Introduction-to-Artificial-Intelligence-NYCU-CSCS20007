import os
import cv2
import glob
import numpy as np


def load_data_small():
    """
        This function loads images form the path: 'data/data_small' and return the training
        and testing dataset. The dataset is a list of tuples where the first element is the 
        numpy array of shape (m, n) representing the image the second element is its 
        classification (1 or 0).

        Parameters:
            None

        Returns:
            dataset: The first and second element represents the training and testing dataset respectively
    """

    # Begin your code (Part 1-1)
    """
        This code segment loads image data for training and testing. It reads 
        images from specified directories for faces and non-faces, assigning 
        labels accordingly (1 for faces, 0 for non-faces).The images are 
        stored as tuples with their corresponding labels.
    """
    train_data = []  # List to store training data tuples (image, label)
    test_data = []   # List to store testing data tuples (image, label)

    # Path to training data
    train_path = "data/data_small/train/"

    # Load face images for training
    face_train_path = os.path.join(train_path, "face/")
    for image in os.listdir(face_train_path):
        img = cv2.imread(os.path.join(face_train_path, image), cv2.IMREAD_GRAYSCALE)
        train_data.append((img, 1))  # Add face image with label 1 to training data

    # Load non-face images for training
    non_face_train_path = os.path.join(train_path, "non-face/")
    for image in os.listdir(non_face_train_path):
        img = cv2.imread(os.path.join(non_face_train_path, image), cv2.IMREAD_GRAYSCALE)
        train_data.append((img, 0))  # Add non-face image with label 0 to training data

    # Path to testing data
    test_path = "data/data_small/test/"

    # Load face images for testing
    face_test_path = os.path.join(test_path, "face/")
    for image in os.listdir(face_test_path):
        img = cv2.imread(os.path.join(face_test_path, image), cv2.IMREAD_GRAYSCALE)
        test_data.append((img, 1))  # Add face image with label 1 to testing data

    # Load non-face images for testing
    non_face_test_path = os.path.join(test_path, "non-face/")
    for image in os.listdir(non_face_test_path):
        img = cv2.imread(os.path.join(non_face_test_path, image), cv2.IMREAD_GRAYSCALE)
        test_data.append((img, 0))  # Add non-face image with label 0 to testing data
    # End your code (Part 1-1)
    
    return train_data, test_data


def load_data_FDDB(data_idx="01"):
    """
        This function generates the training and testing dataset  form the path: 'data/data_small'.
        The dataset is a list of tuples where the first element is the numpy array of shape (m, n)
        representing the image the second element is its classification (1 or 0).
        
        In the following, there are 4 main steps:
        1. Read the .txt file
        2. Crop the faces using the ground truth label in the .txt file
        3. Random crop the non-faces region
        4. Split the dataset into training dataset and testing dataset
        
        Parameters:
            data_idx: the data index string of the .txt file

        Returns:
            train_dataset: the training dataset
            test_dataset: the testing dataset
    """

    with open("data/data_FDDB/FDDB-folds/FDDB-fold-{}-ellipseList.txt".format(data_idx)) as file:
        line_list = [line.rstrip() for line in file]

    # Set random seed for reproducing same image croping results
    np.random.seed(0)

    face_dataset, nonface_dataset = [], []
    line_idx = 0

    # Iterate through the .txt file
    # The detail .txt file structure can be seen in the README at https://vis-www.cs.umass.edu/fddb/
    while line_idx < len(line_list):
        img_gray = cv2.imread(os.path.join("data/data_FDDB", line_list[line_idx] + ".jpg"), cv2.IMREAD_GRAYSCALE)
        num_faces = int(line_list[line_idx + 1])

        # Crop face region using the ground truth label
        face_box_list = []
        for i in range(num_faces):
            # Here, each face is denoted by:
            # <major_axis_radius minor_axis_radius angle center_x center_y 1>.
            coord = [int(float(j)) for j in line_list[line_idx + 2 + i].split()]
            x, y = coord[3] - coord[1], coord[4] - coord[0]            
            w, h = 2 * coord[1], 2 * coord[0]

            left_top = (max(x, 0), max(y, 0))
            right_bottom = (min(x + w, img_gray.shape[1]), min(y + h, img_gray.shape[0]))
            face_box_list.append([left_top, right_bottom])
            # cv2.rectangle(img_gray, left_top, right_bottom, (0, 255, 0), 2)

            img_crop = img_gray[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]].copy()
            face_dataset.append((cv2.resize(img_crop, (19, 19)), 1))

        line_idx += num_faces + 2

        # Random crop N non-face region
        # Here we set N equal to the number of faces to generate a balanced dataset
        # Note that we have alreadly save the bounding box of faces into `face_box_list`, you can utilize it for non-face region cropping
        for i in range(num_faces):
            # Begin your code (Part 1-2)
            """
                This segment generates non-face image samples by randomly selecting regions from the 
                original image that do not overlap with the detected face regions. It ensures that the 
                selected regions have a maximum allowed intersection with the face region, defined by 
                the variable max_intersection. The selected non-face regions are resized to 19x19 pixels 
                and appended to the nonface_dataset list along with the label 0 (indicating non-face).
            """
            img_height, img_width = img_gray.shape
            face_left_top, face_right_bottom = face_box_list[i]

            # Define the maximum allowed intersection with the face region
            max_intersection = 0.2

            # Randomly choose non-face regions until a suitable one is found
            while True:
                x1 = np.random.randint(0, img_width - 19)
                y1 = np.random.randint(0, img_height - 19)
                x2 = x1 + 19
                y2 = y1 + 19

                # Calculate intersection area with the face region
                intersection_area = max(0, min(face_right_bottom[0], x2) - max(face_left_top[0], x1)) * \
                                    max(0, min(face_right_bottom[1], y2) - max(face_left_top[1], y1))

                # Check if the intersection area is small enough
                if intersection_area / (19 * 19) <= max_intersection:
                    nonface_dataset.append((cv2.resize(img_gray[y1:y2, x1:x2], (19, 19)), 0))
                    break
            # End your code (Part 1-2)

            nonface_dataset.append((cv2.resize(img_crop, (19, 19)), 0))

        # cv2.imshow("windows", img_gray)
        # cv2.waitKey(0)

    # train test split
    num_face_data, num_nonface_data = len(face_dataset), len(nonface_dataset)
    SPLIT_RATIO = 0.7

    train_dataset = face_dataset[:int(SPLIT_RATIO * num_face_data)] + nonface_dataset[:int(SPLIT_RATIO * num_nonface_data)]
    test_dataset = face_dataset[int(SPLIT_RATIO * num_face_data):] + nonface_dataset[int(SPLIT_RATIO * num_nonface_data):]

    return train_dataset, test_dataset


def create_dataset(data_type):
    if data_type == "small":
        return load_data_small()
    else:
        return load_data_FDDB()