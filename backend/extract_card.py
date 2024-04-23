import cv2
import numpy as np

image = cv2.imread('data/human_cards/iron_leaves_ex0.jpeg', cv2.IMREAD_COLOR)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

edges = cv2.Canny(threshold, 100, 200)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

corners = cv2.approxPolyDP(contours[0], 30, True)

pts1 = np.float32([corners[0,0], corners[1,0], corners[2,0], corners[3,0]])
pts2 = np.float32([[0,0], [800,0], [800,600], [0,600]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(image, matrix, (800,600))

cv2.imshow('Card', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
