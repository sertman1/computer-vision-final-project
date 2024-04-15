import cv2

def pre_process_image(img_path):
  # load the image
  image = cv2.imread(img_path)
  # resize image to be standard resolution of 733 x 1024
  image = cv2.resize(image, (733, 1024))
  # convert to gray scale
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  return image

def keypoints_to_array(keypoints):
    return [kp.pt for kp in keypoints]

def array_to_keypoints(array):
    return [cv2.KeyPoint(x[0], x[1], 1) for x in array]

def get_keypoints_and_descriptors(img):
  orb = cv2.ORB_create(100000)
  keypoints, descriptors = orb.detectAndCompute(img, None)
  keypoints = keypoints_to_array(keypoints)
  return keypoints, descriptors