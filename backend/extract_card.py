import cv2
import numpy as np

img = cv2.imread('/Users/sam/Desktop/computer_vision/final_project/backend/data/human_cards/turtwig0.png', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur to reduce noise in edge detection
img_blur = cv2.GaussianBlur(img, (9, 9), 0)

# Apply Canny edge detection
edges = cv2.Canny(img_blur, 100, 200)

# Display the edges
cv2.imshow('Edges', edges)
cv2.waitKey(0)

# Define a kernel for the morphological operations
kernel = np.ones((5,5),np.uint8)

# Perform a dilation and erosion to close gaps in between object edges
dilation = cv2.dilate(edges, kernel, iterations = 2)
erosion = cv2.erode(dilation, kernel, iterations = 1)

# Display the result after morphological operations
cv2.imshow('Morphological Operations', erosion)
cv2.waitKey(0)

# Perform a Hough Line Transform
lines = cv2.HoughLinesP(erosion, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

# Create a copy of the original image to draw lines on
img_lines = img.copy()

# Draw the lines on the image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the image with lines
cv2.imshow('Hough Lines', img_lines)
cv2.waitKey(0)

cv2.destroyAllWindows()