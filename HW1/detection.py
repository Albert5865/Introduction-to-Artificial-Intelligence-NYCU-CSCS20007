import os
import cv2

def detect(dataPath, clf):
    """
    This function reads data from a file (detectData.txt) to get information about images and their corresponding faces,
    loads the images, detects faces using a classifier (clf), and draws rectangles around the detected faces based on the
    classifier's output. Finally, it saves the resulting images in a result/test/ directory.

    Parameters:
        dataPath: the path of detectData.txt
        clf: the classifier used for face detection

    Returns:
        No returns.
    """
    # Begin your code (Part 4)
    """
      This code reads data from a file containing image names and coordinates of detected faces. 
      It then loads each image, detects faces, and draws rectangles around them based on the detection 
      results. The processed images are saved with rectangles drawn around the detected faces.
    """
    with open(dataPath, "r") as file:
        # Iterate through each line in the file
        for line in file:
            # Split the line into image name and number of faces
            image_name, people_num = line.split()
            people = []
            # Iterate through each face in the image
            for _ in range(int(people_num)):
                # Extract face data (x, y, width, height)
                face_data = next(file).split()
                face = tuple(map(int, face_data))
                people.append(face)
            # Load the image
            image = cv2.imread(os.path.join("data/detect/", image_name))
            # Load the grayscale version of the image
            image_cmp = cv2.imread(os.path.join("data/detect/", image_name), cv2.IMREAD_GRAYSCALE)
            # Iterate through each face in the image
            for face in people:
                x, y, w, h = face
                # Crop and resize the face region
                face_image = cv2.resize(image_cmp[y:y+h, x:x+w], (19, 19), interpolation=cv2.INTER_LINEAR)
                # Classify the face image
                classification = clf.classify(face_image)
                # Determine the color of the rectangle based on classification result
                color = (0, 255, 0) if classification == 1 else (0, 0, 255)
                # Draw rectangle around the detected face
                cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness=2)
            # Save the resulting image with rectangles drawn around the detected faces
            cv2.imwrite("result_" + image_name, image)
    # End your code (Part 4)
