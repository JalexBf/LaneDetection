import cv2
import os
import numpy as np


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
    # Get valid file paths from input folder
    file_paths = process_input_folder(input_folder, output_folder, file_extension_filter=(".jpg", ".png", ".jpeg"))

    for input_file_path in file_paths:
        image = cv2.imread(input_file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        filename = os.path.basename(input_file_path)
        output_file_path = os.path.join(output_folder, f'edges_{filename}')
        cv2.imwrite(output_file_path, edges)



def detect_lanes(input_folder, output_folder):
    # Get valid file paths from input folder
    file_paths = process_input_folder(input_folder, output_folder, file_extension_filter=(".jpg", ".png", ".jpeg"))

    for input_file_path in file_paths:
        edge_image = cv2.imread(input_file_path, cv2.IMREAD_GRAYSCALE)
        if edge_image is None:
            continue

            # Hough Transform to detect lines in the edge image
            lines = cv2.HoughLinesP(edge_image, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)
            if lines is not None:
                # Create an empty image to draw lines on
                line_image = np.zeros_like(edge_image)

                # Draw lines on the empty image
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

                output_file_path = os.path.join(output_folder, f'lanes_{filename}')
                cv2.imwrite(output_file_path, line_image)


def refine_lane_lines(lanes_folder, cropped_folder, refined_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(refined_folder):
        os.makedirs(refined_folder)

    # Parameters for line filtering
    min_slope = 0.5  # Minimum slope to consider a line to be a lane
    max_slope = 2  # Maximum slope to consider a line to be a lane

    for filename in os.listdir(lanes_folder):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            detected_file_path = os.path.join(lanes_folder, filename)
            original_filename = filename.replace('lanes_', '')
            original_filename = original_filename.replace('edges_', '')
            original_file_path = os.path.join(cropped_folder, original_filename)

            # Read images
            detected_image = cv2.imread(detected_file_path)
            original_image = cv2.imread(original_file_path)

            if detected_image is None or original_image is None:
                continue

            # Convert detected image to grayscale
            gray = cv2.cvtColor(detected_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)

            # Filter lines
            lane_lines = []
            for line in lines:
                for x1, y1, x2, y2 in line:
                    slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
                    if min_slope < abs(slope) < max_slope:
                        lane_lines.append((x1, y1, x2, y2))

            # Draw lane lines on the original image
            lane_image = original_image.copy()
            for x1, y1, x2, y2 in lane_lines:
                cv2.line(lane_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

            output_file_path = os.path.join(refined_folder, f'refined_{filename}')
            cv2.imwrite(output_file_path, lane_image)


def calculate_lane_center(image, lines):
    left_lines = [line for line in lines if
                  (line[0][2] - line[0][0]) != 0 and ((line[0][3] - line[0][1]) / (line[0][2] - line[0][0])) < 0]
    right_lines = [line for line in lines if
                   (line[0][2] - line[0][0]) != 0 and ((line[0][3] - line[0][1]) / (line[0][2] - line[0][0])) > 0]

    if not left_lines or not right_lines:  # No lines detected
        return None

    # Average out the lines to get a single line for each side
    left_line = np.mean(left_lines, axis=0)
    right_line = np.mean(right_lines, axis=0)

    # Choose the position to calculate center
    height = image.shape[0]

    offset = 150
    position_y = height - offset

    left_bottom_x = int(
        (position_y - left_line[0][1]) * (left_line[0][2] - left_line[0][0]) / (left_line[0][3] - left_line[0][1]) +
        left_line[0][0])
    right_bottom_x = int(
        (position_y - right_line[0][1]) * (right_line[0][2] - right_line[0][0]) / (
                    right_line[0][3] - right_line[0][1]) +
        right_line[0][0])

    # Calculate midpoint
    center_x = (left_bottom_x + right_bottom_x) // 2
    return center_x, position_y


def draw_lane_center(refined_folder, center_lines_folder):
    # Get valid file paths from input folder
    refined_file_paths = process_input_folder(refined_folder, center_lines_folder, file_extension_filter=(".jpg", ".png", ".jpeg"))

    for refined_file_path in refined_file_paths:
        image = cv2.imread(refined_file_path)
        if image is None:
            continue

            green_mask = cv2.inRange(image, (0, 255, 0), (0, 255, 0))
            lines = cv2.HoughLinesP(green_mask, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)

            if lines is not None:
                center = calculate_lane_center(image, lines)
                if center:
                    cv2.circle(image, center, radius=5, color=(255, 0, 0), thickness=-1)
                else:
                    print(f"Could not find lane center for {filename}")
            else:
                print(f"No lines found in {filename}")

            output_file_path = os.path.join(center_lines_folder, f'center_{filename}')
            cv2.imwrite(output_file_path, image)


def calculate_deviation(center_coordinates, image_width):
    lane_center_x, _ = center_coordinates
    image_center_x = image_width // 2
    deviation = lane_center_x - image_center_x
    return deviation


def draw_deviation(refined_folder, deviation_folder, center_lines_folder):
    # Get valid file paths from input folder
    refined_file_paths = process_input_folder(refined_folder, deviation_folder, file_extension_filter=(".jpg", ".png", ".jpeg"))

    for refined_file_path in refined_file_paths:
        image = cv2.imread(refined_file_path)
        if image is None:
            continue

            green_mask = cv2.inRange(image, (0, 255, 0), (0, 255, 0))
            lines = cv2.HoughLinesP(green_mask, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)

            if lines is not None:
                center = calculate_lane_center(image, lines)
                if center:
                    # Draw the original center dot
                    cv2.circle(image, center, radius=5, color=(255, 0, 0), thickness=-1)

                    # Calculate deviation
                    deviation = calculate_deviation(center, image.shape[1])
                    print(f"Deviation for {filename}: {deviation}px")

                    # Draw deviation line and text
                    cv2.arrowedLine(image, (image.shape[1] // 2, center[1]), center, (0, 0, 255), 2)
                    deviation_text = f"Δd={deviation}px"
                    cv2.putText(image, deviation_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:
                    print(f"Could not find lane center for {filename}")
            else:
                print(f"No lines found in {filename}")

            # Write the image with deviation data to the new folder
            deviation_file_path = os.path.join(deviation_folder, f'deviation_{filename}')
            cv2.imwrite(deviation_file_path, image)


input_folder = 'single_lane'
cropped_folder = 'cropped_images'
edges_folder = 'edges_images'
lanes_folder = 'lanes_images'
refined_folder = 'refined_images'
center_lines_folder = 'center_lines'
deviation_folder = 'deviation_images'

crop_images(input_folder, cropped_folder)
edge_detection(cropped_folder, edges_folder)
detect_lanes(edges_folder, lanes_folder)
refine_lane_lines(lanes_folder, cropped_folder, refined_folder)
draw_lane_center(refined_folder, center_lines_folder)
draw_deviation(refined_folder, deviation_folder, center_lines_folder)
