import cv2
import numpy as np

#
# replace 'images/Picturetrain.png' with 'file_directory'
#
image_path = 'images/Picturetrain.png'

#
# parameter specification
#
sigma_1 = 25
ksize_1 = 2 * int(2 * sigma_1) + 1
alpha = 0.5      # transparency of overlay
#
#
#
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply Gaussian Blur
blurred = cv2.GaussianBlur(image, (ksize_1, ksize_1), sigma_1)

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
color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
masked_image = cv2.addWeighted(bf_mask, alpha, color_image, 1 - alpha, 0)
cv2.imshow("masked_image", masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()