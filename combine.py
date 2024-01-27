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
mask_value_3 = 255
sigma_4 = 10
ksize_4 = 2 * int(2 * sigma_4) + 1
sigma_3 = 25
alpha = 0.5      # transparency of overlay
#
#
#
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image_inner_ring = image.copy()
# Apply Gaussian Blur
blurred = cv2.GaussianBlur(image, (ksize_1, ksize_1), sigma_1)


#
# 1. bf masking
#
# Convert to binary image
_, binary_image_outer_ring = cv2.threshold(blurred, 190, 255, cv2.THRESH_BINARY_INV)

# find contour
contours, _ = cv2.findContours(binary_image_outer_ring, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
num_of_contours = len(contours)
if num_of_contours >= 1:
    # color the contour into red
    mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    cv2.drawContours(mask, [np.array(contours[num_of_contours - 2])], -1, (0, 0, 255), thickness=cv2.FILLED)  # Red color
    cv2.drawContours(mask, [np.array(contours[num_of_contours - 1])], -1, (0, 0, 0), thickness=cv2.FILLED)  # black color

#
# 3,4. fp drag/impression masking
#


#
# focus on inner ring
#
inner_zoom = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
cv2.fillPoly(inner_zoom, [contours[num_of_contours-1]], mask_value_3)
sel = inner_zoom != mask_value_3
image_inner_ring[sel] = mask_value_3

#
# 4.detect the firing pin drag
#

blurred = cv2.GaussianBlur(image_inner_ring, (ksize_4, ksize_4), sigma_4)
# Convert to binary image
_, binary_image_fp = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)
# find contour
contours, _ = cv2.findContours(binary_image_fp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
num_of_contours = len(contours)
if num_of_contours >= 1:
    # color the contour into blue
    cv2.drawContours(mask, [np.array(contours[1])], -1, (200, 200, 0), thickness=cv2.FILLED)  # black color

#(blue,green,red)

#
# 3. detect the firing pin impression
#
hull = cv2.convexHull(contours[1], returnPoints=True)
fpi = cv2.HoughCircles(binary_image_fp,
                   cv2.HOUGH_GRADIENT, 1, 100, param1 = 1,
               param2 = 10, minRadius = 60, maxRadius = 100)
if fpi is not None:
    # Convert the circle parameters a, b and r to integers.
    fpi = np.uint16(np.around(fpi))
    fpi = fpi[0,fpi[:,:,2].argsort()]
    pt = fpi[0, 0]
    a, b, r = pt[0], pt[1], pt[2]
    # Draw the circumference of the circle.
    cv2.circle(mask, (a, b), r, (200, 0, 200), -1)
    #
    binary_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(binary_mask, 1, 255, cv2.THRESH_BINARY)

    # Invert the binary mask to create an inverse mask (for unmasked areas)
    inverse_binary_mask = cv2.bitwise_not(binary_mask)

    # Blend the original image and the mask, applying the mask only where binary_mask is white
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    blended_masked_area = cv2.addWeighted(color_image, alpha, mask, 1 - alpha, 0)
    # Use the inverse binary mask to isolate unmasked areas from the original image
    unmasked_area = cv2.bitwise_and(color_image, color_image, mask=inverse_binary_mask)
    # Combine the blended masked regions with the unmasked regions
    masked_image = cv2.bitwise_or(blended_masked_area, unmasked_area)
    #
    # overlay the colored masking
    #
    cv2.imshow("overlay colored masking", masked_image)
    cv2.waitKey(0)

    cv2.circle(masked_image, (a, b), int(r/4), (255, 0, 0), 2)
    # Draw a small circle (of radius 1) to show the center.
    cv2.circle(masked_image, (a, b), 1, (255, 0, 0), 3)

    cv2.imshow("Detected Circle", masked_image)
    cv2.waitKey(0)

    max_length = 0
    pt1_longest = (0, 0)
    pt2_longest = (0, 0)

    for i in range(len(hull)):
        pt1 = tuple(hull[i][0])
        pt2 = (a, b)
        # Compute the distance between the points
        length = np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
        # Update max length and points if necessary
        if length > max_length:
            max_length = length
            pt1_longest, pt2_longest = pt1, pt2
    # drawing an arrow (the longest straight line in the inner ring) on it
    cv2.arrowedLine(masked_image, pt2_longest, pt1_longest, (255, 0, 0), 2)  # Draw in blue
    cv2.imshow("Detected arrow", masked_image)
    cv2.waitKey(0)
cv2.destroyAllWindows()


# Generate a timestamp
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"images/masked_image_{timestamp}.png"
cv2.imwrite(filename, masked_image)
print(f"Image saved as {filename}")


