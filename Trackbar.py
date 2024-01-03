import cv2
import os
import numpy as np


# Callback function for the trackbar (does nothing in this case)
def nothing(x):
    pass


# Create a window
cv2.namedWindow('Canny')

# Create trackbars for threshold changes
cv2.createTrackbar('Threshold1', 'Canny', 50, 255, nothing)
cv2.createTrackbar('Threshold2', 'Canny', 150, 255, nothing)

# Define your folder path
input_folder = 'cropped_images'  # Replace with your images folder path

# Get the list of image file names
image_files = [f for f in os.listdir(input_folder) if f.endswith((".jpg", ".png", ".jpeg"))]

# Loop over images
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    while True:
        # Get current positions of the trackbars
        threshold1 = cv2.getTrackbarPos('Threshold1', 'Canny')
        threshold2 = cv2.getTrackbarPos('Threshold2', 'Canny')

        # Apply Canny edge detection
        edges = cv2.Canny(img, threshold1, threshold2)

        # Display the resulting frame
        cv2.imshow('Canny', edges)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Break the outer loop if the window is closed
    if cv2.getWindowProperty('Canny', 0) < 0:
        break

# When everything is done, release the window
cv2.destroyAllWindows()
