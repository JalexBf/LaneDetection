import cv2
import os
import numpy as np
import math

from numpy import average
from sklearn.linear_model import RANSACRegressor


def process_input_folder(input_folder, output_folder, file_extension_filter=None):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List to hold valid file paths
    valid_file_paths = []

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if file_extension_filter and not filename.lower().endswith(file_extension_filter):
            continue

        input_file_path = os.path.join(input_folder, filename)
        image = cv2.imread(input_file_path)
        if image is not None:
            valid_file_paths.append(input_file_path)

    return valid_file_paths


def crop_images(input_folder, output_folder):
    # Get valid file paths from input folder
    file_paths = process_input_folder(input_folder, output_folder, file_extension_filter=(".jpg", ".png", ".jpeg"))

    for input_file_path in file_paths:
        image = cv2.imread(input_file_path)
        if image is None:
            continue

        # Height to start the crop from
        start_crop = image.shape[0] // 4
        # Total height of the image
        total_height = image.shape[0]
        # Crop bottom 2/3 of the image
        cropped_image = image[start_crop:total_height, :]

        # Create filename for the cropped image
        filename = os.path.basename(input_file_path)
        output_file_path = os.path.join(output_folder, f'cropped_{filename}')

        cv2.imwrite(output_file_path, cropped_image)


def edge_detection(input_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            input_file_path = os.path.join(input_folder, filename)
            image = cv2.imread(input_file_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)

            output_file_path = os.path.join(output_folder, f'edges_{filename}')
            cv2.imwrite(output_file_path, edges)

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x1 == x2:  # Ignore vertical lines to avoid infinite slope
            continue
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_line = right_line = None

    if left_fit:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_line_points(image, left_fit_average)

    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_line_points(image, right_fit_average)

    return np.array([left_line, right_line], dtype=object)

def make_line_points(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    try:
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
    except Exception as e:
        print(f"Error calculating line points: {e}")
        return None

    # Adjust points to be within the image boundaries
    x1 = max(0, min(image.shape[1] - 1, x1))
    x2 = max(0, min(image.shape[1] - 1, x2))

    return np.array([x1, y1, x2, y2])


def detect_lanes(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.startswith("edges_"):
            edge_image_path = os.path.join(input_folder, filename)
            edge_image = cv2.imread(edge_image_path, cv2.IMREAD_GRAYSCALE)
            if edge_image is None:
                print(f"Could not read image {filename}")
                continue

            lines = cv2.HoughLinesP(edge_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
            if lines is not None:
                averaged_lines = average_slope_intercept(edge_image, lines)
                line_image = np.zeros((*edge_image.shape, 3), dtype=np.uint8)
                if averaged_lines is not None:
                    for line in averaged_lines:
                        if line is not None:
                            x1, y1, x2, y2 = line
                            # Check for valid coordinates
                            if all(isinstance(coord, (int, float)) and not math.isnan(coord) for coord in
                                   [x1, y1, x2, y2]):
                                cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 10)
                            else:
                                print(f"Invalid line coordinates: {line}")

                # Using cropped images as original images
                original_image_filename = filename.replace('edges_', '')
                original_image_path = os.path.join(cropped_folder, original_image_filename)
                original_image = cv2.imread(original_image_path)
                if original_image is not None:
                    combo_image = cv2.addWeighted(original_image, 0.8, line_image, 1, 1)
                    output_image_path = os.path.join(output_folder, original_image_filename)
                    cv2.imwrite(output_image_path, combo_image)
                else:
                    print(f"Could not read original image for {original_image_filename}")


input_folder = 'single_lane'
cropped_folder = 'cropped_images'
edges_folder = 'edges_images'
lanes_folder = 'lanes_images'

crop_images(input_folder, cropped_folder)
edge_detection(cropped_folder, edges_folder)
detect_lanes(edges_folder, lanes_folder)

