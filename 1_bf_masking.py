import cv2
import numpy as np

image = cv2.imread('images/Picturetrain.png', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian Blur
sigma = 25
ksize = 2 * int(2 * sigma) + 1
blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)

# Convert to binary image
_, binary_image_outer_ring = cv2.threshold(blurred, 190, 255, cv2.THRESH_BINARY_INV)

# find contour
contours, _ = cv2.findContours(binary_image_outer_ring, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
num_of_contours = len(contours)
if num_of_contours >= 1:
    # color the contour into red
    bf_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    cv2.drawContours(bf_mask, [np.array(contours[num_of_contours - 2])], -1, (0, 0, 255), thickness=cv2.FILLED)  # Red color
    cv2.drawContours(bf_mask, [np.array(contours[num_of_contours - 1])], -1, (0, 0, 0), thickness=cv2.FILLED)  # black color

contours[0].shape
# overlay the color contour
alpha = 0.5
color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
masked_image = cv2.addWeighted(bf_mask, alpha, color_image, 1 - alpha, 0)
