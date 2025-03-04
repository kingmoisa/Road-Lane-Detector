# 1. Import necessary libraries
import cv2
import numpy as np

# Initialize video capture with the specified video file
cam = cv2.VideoCapture('Lane Detection Test Video 01.mp4')

# Initialize global variables for lane line parameters
l_bs = [0, 0]
l_as = [0, 0]
r_bs = [0, 0]
r_as = [0, 0]
final_l_bs = [0, 0]
final_l_as = [0, 0]
final_r_bs = [0, 0]
final_r_as = [0, 0]
left_top = [0, 0]
right_top = [0, 0]
left_bottom = [0, 0]
right_bottom = [0, 0]
final_left_top = [0, 0]
final_right_top = [0, 0]
final_left_bottom = [0, 0]
final_right_bottom = [0, 0]
final_left_top_x = 0
final_left_bottom_x = 0
final_right_top_x = 0
final_right_bottom_x = 0

# Infinite loop to process each frame from the video
while True:
    # Read each frame from the video
    ret, frame = cam.read()
    if not ret:
        break  # Exit the loop if there are no frames to read
    # cv2.imshow('Original', frame)

    # 2. Resize the frame for processing
    scale_percent = 30  # Percentage of the original size to keep
    scale_factor = scale_percent / 100  # Scaling factor

    # Calculate dimensions of the resized frame
    frame_height = int(frame.shape[0] * scale_factor)
    frame_width = int(frame.shape[1] * scale_factor)
    frame_dimensions = (frame_width, frame_height)

    # Resize the frame
    smaller = cv2.resize(frame, frame_dimensions)
    cv2.imshow('Smaller', smaller)

    # 3. Convert the resized frame to grayscale
    gray_frame = cv2.cvtColor(smaller, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscale', gray_frame)

    # 4. Create a mask to isolate the road area
    new_frame = np.zeros([frame_height, frame_width], dtype=np.uint8)

    # Define coordinates for a trapezoid to isolate the road
    trapez_upper_left = (int(frame_width * 0.45), int(frame_height * 0.80))
    trapez_upper_right = (int(frame_width * 0.55), int(frame_height * 0.80))
    trapez_lower_left = (int(frame_width * 0), int(frame_height))
    trapez_lower_right = (int(frame_width * 1.0), int(frame_height))

    # Create a trapezoidal mask for the road
    trapez_bounds = np.array([trapez_upper_left, trapez_upper_right, trapez_lower_right, trapez_lower_left], dtype=np.int32)
    trapez_frame = cv2.fillConvexPoly(new_frame, trapez_bounds, 255)
    cv2.imshow('Trapezoid', trapez_frame)

    # Isolate the road using the trapezoidal mask
    road = trapez_frame * gray_frame
    cv2.imshow('Road', road * 255)

    # 5. Obtain a top-down view of the road
    frame_upper_left = (0, 0)
    frame_upper_right = (frame_width, 0)
    frame_lower_left = (0, frame_height)
    frame_lower_right = (frame_width, frame_height)

    frame_bounds = np.array([frame_upper_left, frame_upper_right,
                             frame_lower_right, frame_lower_left], dtype=np.float32)
    trapez_bounds = np.float32(trapez_bounds)

    # Calculate the perspective transformation matrix
    magic_matrix = cv2.getPerspectiveTransform(trapez_bounds, frame_bounds)
    top_down_frame = cv2.warpPerspective(road, magic_matrix, frame_dimensions)
    cv2.imshow('Top-Down', top_down_frame * 255)

    # 6. Apply Gaussian blur to reduce noise
    ksize = (5, 5)  # Kernel size for blurring
    blurred_frame = cv2.blur(top_down_frame, ksize)
    cv2.imshow('Blur', blurred_frame * 255)

    # 7. Perform edge detection using Sobel filters
    sobel_vertical = np.float32([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # Vertical Sobel filter
    sobel_horizontal = np.transpose(sobel_vertical)  # Horizontal Sobel filter

    blurred_frame = np.float32(blurred_frame)  # Convert to float for filtering

    sobel_vertical_frame = cv2.filter2D(blurred_frame, -1, sobel_vertical)
    sobel_horizontal_frame = cv2.filter2D(blurred_frame, -1, sobel_horizontal)

    # Combine the results from both Sobel filters
    sobel_final = np.sqrt((sobel_vertical_frame ** 2) + (sobel_horizontal_frame ** 2))
    sobel_vertical_frame = cv2.convertScaleAbs(sobel_vertical_frame)
    sobel_horizontal_frame = cv2.convertScaleAbs(sobel_horizontal_frame)
    sobel_final_frame = cv2.convertScaleAbs(sobel_final)

    # Display the Sobel filter results
    cv2.imshow('Vertical Sobel', sobel_vertical_frame)
    cv2.imshow('Horizontal Sobel', sobel_horizontal_frame)
    cv2.imshow('Final Sobel', sobel_final_frame)

    # 8. Binarize the frame to isolate edges
    threshold = 127
    ret, binarized_frame = cv2.threshold(sobel_final, threshold, 255, cv2.THRESH_BINARY)
    cv2.imshow('Binarized Frame', binarized_frame)

    # 9. Identify lane lines
    # a. Remove noise from the edges
    copy_frame = binarized_frame.copy()
    copy_frame[:, 0:int(0.05 * frame_width)] = 0  # Set left edge to zero
    copy_frame[:, int(0.95 * frame_width):] = 0  # Set right edge to zero
    cv2.imshow('Processed Frame', copy_frame)

    # b. Identify white pixels in the left and right halves
    left_white_pixels = np.argwhere(copy_frame[:, :int(0.5 * frame_width)] > 255 // 2)
    right_white_pixels = np.argwhere(copy_frame[:, int(0.5 * frame_width):] > 255 // 2)

    left_xs = left_white_pixels[:, 1]
    left_ys = left_white_pixels[:, 0]
    right_xs = right_white_pixels[:, 1] + frame_width // 2
    right_ys = right_white_pixels[:, 0]

    # 10. Fit lines to the detected lane edges
    if len(left_xs) > 0:
        l_bs, l_as = np.polynomial.polynomial.polyfit(left_xs, left_ys, deg=1)  # Fit line for left lane

    if len(right_xs) > 0:
        r_bs, r_as = np.polynomial.polynomial.polyfit(right_xs, right_ys, deg=1)  # Fit line for right lane

    # Calculate the coordinates of the lane line endpoints
    left_top_y = 0
    left_top_x = (left_top_y - l_bs) / l_as

    left_bottom_y = frame_height
    left_bottom_x = (left_bottom_y - l_bs) / l_as

    right_top_y = 0
    right_top_x = (right_top_y - r_bs) / r_as

    right_bottom_y = frame_height
    right_bottom_x = (right_bottom_y - r_bs) / r_as

    # Store the calculated coordinates for drawing
    left_top[1] = left_top_y
    left_bottom[1] = left_bottom_y
    right_top[1] = right_top_y
    right_bottom[1] = right_bottom_y

    left_top[0] = int(left_top_x)
    left_bottom[0] = int(left_bottom_x)

    right_top[0] = int(right_top_x)
    right_bottom[0] = int(right_bottom_x)

    # Draw the detected lane lines on the binarized frame
    frame_lines = cv2.line(binarized_frame, left_top, left_bottom, (200, 0, 0), 5)
    frame_lines = cv2.line(frame_lines, right_top, right_bottom, (100, 0, 0), 5)
    cv2.line(frame_lines, (int(frame_width / 2), 0), (int(frame_width / 2), int(frame_height)), (255, 0, 0), 1)

    cv2.imshow('Lane Lines', frame_lines)

    # 11. Transform lane lines back to the original perspective
    new_left_frame = np.zeros((frame_dimensions[1], frame_dimensions[0]), dtype=np.uint8)
    new_left_frame = cv2.line(new_left_frame, left_top, left_bottom, (255, 0, 0), 5)
    #cv2.imshow('alt test left lane', new_left_frame)
    new_magic_matrix = cv2.getPerspectiveTransform(frame_bounds, trapez_bounds)
    back_to_normal_frame = cv2.warpPerspective(new_left_frame, new_magic_matrix, frame_dimensions)
    #cv2.imshow('Test1', back_to_normal_frame)

    new_right_frame = np.zeros((frame_dimensions[1], frame_dimensions[0]), dtype=np.uint8)

    new_right_frame = cv2.line(new_right_frame, right_top, right_bottom, (255, 0, 0), 5)
    #cv2.imshow('alt test', new_right_frame)
    # new_magic_matrix = cv2.getPerspectiveTransform(frame_bounds, trapez_bounds)
    back_to_normal_right_frame = cv2.warpPerspective(new_right_frame, new_magic_matrix, frame_dimensions)
    #cv2.imshow('Test2', back_to_normal_right_frame)

    # Find white pixels in the transformed frames
    final_left_white_pixels = np.argwhere(back_to_normal_frame > 0)
    final_right_white_pixels = np.argwhere(back_to_normal_right_frame > 0)

    # Fit lines to the white pixels
    final_left_xs = final_left_white_pixels[:, 1]
    final_left_ys = final_left_white_pixels[:, 0]

    final_right_xs = final_right_white_pixels[:, 1]  # + frame_width // 2
    final_right_ys = final_right_white_pixels[:, 0]

    if len(final_left_xs) != 0 or len(final_left_ys) != 0:
        # If points exist then calculate
        final_l_bs, final_l_as = np.polynomial.polynomial.polyfit(final_left_xs, final_left_ys, deg=1)

    if len(final_right_xs) != 0 or len(final_right_ys) != 0:
        # same thing as above
        final_r_bs, final_r_as = np.polynomial.polynomial.polyfit(final_right_xs, final_right_ys, deg=1)

    # Calculating coordinates of the top and bottom points of the final left and right lanes
    final_left_top_y = int(frame_height * 0.8)
    if not (abs(int((final_left_top_y - final_l_bs) / final_l_as)) > 10 ** 7):
        final_left_top_x = int((final_left_top_y - final_l_bs) / final_l_as)

    final_left_bottom_y = int(frame_height)
    if not (abs(int((final_left_bottom_y - final_l_bs) / final_l_as)) > 10 ** 7):
        final_left_bottom_x = int((final_left_bottom_y - final_l_bs) / final_l_as)

    final_right_top_y = int(frame_height * 0.8)
    if not (abs(int((final_right_top_y - final_r_bs) / final_r_as)) > 10 ** 7):
        final_right_top_x = int((final_right_top_y - final_r_bs) / final_r_as)

    final_right_bottom_y = int(frame_height)
    if not (abs(int((final_right_bottom_y - final_r_bs) / final_r_as)) > 10 ** 7):
        final_right_bottom_x = int((final_right_bottom_y - final_r_bs) / final_r_as)

    final_left_top[1] = int(final_left_top_y)
    final_left_bottom[1] = int(final_left_bottom_y)
    final_right_top[1] = int(final_right_top_y)
    final_right_bottom[1] = int(final_right_bottom_y)

    final_left_top[0] = int(final_left_top_x)
    final_left_bottom[0] = int(final_left_bottom_x)

    final_right_top[0] = int(final_right_top_x)
    final_right_bottom[0] = int(final_right_bottom_x)

    # Drawing the final lines on the original frame
    copy_final_frame = smaller.copy()
    copy_final_frame = cv2.line(copy_final_frame, final_left_top, final_left_bottom, (50, 50, 250), 2)
    copy_final_frame = cv2.line(copy_final_frame, final_right_top, final_right_bottom, (128, 0, 128), 2)
    cv2.imshow('Final', copy_final_frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
cam.release()
cv2.destroyAllWindows()
