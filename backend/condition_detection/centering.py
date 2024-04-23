import cv2
import numpy as np


def get_centering(img_path):
  # Load the image
  image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

  # Detect edges using Canny edge detection
  edges = cv2.Canny(image, 100, 200)

  # Find contours
  contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Assume the largest contour is the card
  card_contour = max(contours, key=cv2.contourArea)

  # Get the bounding rectangle for the card contour
  x, y, w, h = cv2.boundingRect(card_contour)

  # Calculate centering
  centering_left_right = w / image.shape[1]
  centering_top_bottom = h / image.shape[0]

  print(f'Centering Left/Right: {centering_left_right * 100}%')
  print(f'Centering Top/Bottom: {centering_top_bottom * 100}%')

def show_image():
  # Load the image
  image = cv2.imread('/Users/sam/Desktop/computer_vision/final_project/backend/data/temporal_forces/tef.197.biancas_devotion.png')

  # Check if image is loaded
  if image is None:
      print("Could not open or find the image")
  else:
      # Convert to grayscale
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

      # Detect edges using Canny edge detection
      edges = cv2.Canny(gray, 100, 200)

      # Find contours
      contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

      # Assume the largest contour is the card
      card_contour = max(contours, key=cv2.contourArea)

      # Get the bounding rectangle for the card contour
      x, y, w, h = cv2.boundingRect(card_contour)

      # Draw the bounding rectangle on the original image
      cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

      # Display the image
      cv2.imshow('Image with Bounding Rectangle', image)
      cv2.waitKey(0)


get_centering('/Users/sam/Desktop/computer_vision/final_project/backend/data/temporal_forces/tef.197.biancas_devotion.png')
get_centering('/Users/sam/Desktop/computer_vision/final_project/backend/data/human_cards/iron_leaves_ex0.jpeg')
show_image()

# Gold standard
# Centering Left/Right: 59.34515688949522%
# Centering Top/Bottom: 31.34765625%