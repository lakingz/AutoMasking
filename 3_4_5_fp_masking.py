import cv2
import numpy as np

image = cv2.imread('images/Picturetrain.png', cv2.IMREAD_GRAYSCALE)
image_inner_ring = cv2.imread('images/Picturetrain.png', cv2.IMREAD_GRAYSCALE)

#
# focus on inner ring
#
# Apply Gaussian Blur
sigma = 25
ksize = 2 * int(2 * sigma) + 1
blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)
# Convert to binary image
_, binary_image_outer_ring = cv2.threshold(blurred, 190, 255, cv2.THRESH_BINARY_INV)

# find contour
contours, _ = cv2.findContours(binary_image_outer_ring, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
num_of_contours = len(contours)

inner_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
fill_color = [0, 0, 0]
mask_value = 255
cv2.fillPoly(inner_mask, [contours[num_of_contours-1]], mask_value)
sel = inner_mask != mask_value
image_inner_ring[sel] = mask_value

#
# 4.detect the firing pin drag
#
sigma = 10
ksize = 2 * int(2 * sigma) + 1
blurred = cv2.GaussianBlur(image_inner_ring, (ksize, ksize), sigma)
cv2.imshow("blurred", blurred)
cv2.waitKey(0)
# Convert to binary image
_, binary_image_fp = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)
# find contour
contours, _ = cv2.findContours(binary_image_fp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
num_of_contours = len(contours)
if num_of_contours >= 1:
    # color the contour into blue
    bf_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    cv2.drawContours(bf_mask, [np.array(contours[1])], -1, (200,200,0), thickness=cv2.FILLED)  # black color
hull = cv2.convexHull(contours[1], returnPoints=True)
#(blue,green,red)

#
# 3. detect the firing pin impression
#
fpi = cv2.HoughCircles(binary_image_fp,
                   cv2.HOUGH_GRADIENT, 1, 100, param1 = 1,
               param2 = 10, minRadius = 60, maxRadius = 100)
fpi
if fpi is not None:
    # Convert the circle parameters a, b and r to integers.
    fpi = np.uint16(np.around(fpi))
    fpi = fpi[0,fpi[:,:,2].argsort()]
    pt = fpi[0, 0]
    a, b, r = pt[0], pt[1], pt[2]
    # Draw the circumference of the circle.

    cv2.circle(bf_mask, (a, b), r, (200, 0, 200), -1)
    # overlay the color contour
    alpha = 0.5
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    masked_image = cv2.addWeighted(bf_mask, alpha, color_image, 1 - alpha, 0)

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
    cv2.arrowedLine(masked_image, pt2_longest, pt1_longest, (255, 0, 0), 2)  # Draw in blue
    cv2.imshow("Detected Circle", masked_image)
    cv2.waitKey(0)



