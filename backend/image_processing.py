import numpy as np
import cv2
import sqlite3
import pickle

MIN_MATCH_COUNT = 10

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

def get_homography(img, keypoints, descriptors, stored_img_path, stored_keypoints, stored_descriptors, flann):
    # Match the descriptors of the user-submitted image and the stored image
    matches = flann.knnMatch(descriptors, stored_descriptors, k=2)

    # Apply ratio test to find the good matches
    good_matches = [m for m,n in matches if m.distance < 0.7*n.distance]

    # If enough good matches are found
    if len(good_matches) > MIN_MATCH_COUNT:
        # Get the matched keypoints
        src_pts = np.float32([ keypoints[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ stored_keypoints[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

        # Compute the homography matrix
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

        # Get the corners of the Pokémon card image
        h,w = img.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

        # Project corners into frame
        dst = cv2.perspectiveTransform(pts,M)

        # Connect corners with lines
        img2 = cv2.polylines(img,[np.int32(dst)],True,255,3, cv2.LINE_AA)

        return img2
    else:
        print("Not enough matches are found - %d/%d" % (len(good_matches), MIN_MATCH_COUNT))

def find_k_nearest_neighbors(img_path, k):
  img = pre_process_image(img_path)
  keypoints, descriptors = get_keypoints_and_descriptors(img)
  keypoints = array_to_keypoints(keypoints)

  # FLANN parameters
  FLANN_INDEX_LSH = 6
  index_params= dict(algorithm = FLANN_INDEX_LSH,
                     table_number = 6,
                     key_size = 12,
                     multi_probe_level = 1)
  search_params = dict(checks=50)

  flann = cv2.FlannBasedMatcher(index_params, search_params)

  match_results = []

  # Connect to the SQLite database
  conn = sqlite3.connect('db/image_features.db')
  c = conn.cursor()

  # Query the database to get all stored image features
  rows = c.execute('SELECT * FROM features').fetchall()

  for row in rows:
    stored_img_path, image_link, serialized_kps, serialized_des = row

    # Deserialize the keypoints and descriptors
    kps = pickle.loads(serialized_kps)
    des = pickle.loads(serialized_des)
    kps = array_to_keypoints(kps)

    matches = flann.knnMatch(descriptors, des, k=2)
    good_matches = [m for m,n in matches if m.distance < 0.7*n.distance]
    match_results.append((stored_img_path, image_link, len(good_matches)))

  # Sort the results by the number of matches (in descending order)
  match_results.sort(key=lambda x: x[2], reverse=True)

  # Close the connection to the database
  conn.close()

  if k == 1:
      stored_img_path, image_link, num_matches = match_results[0]
      stored_img = cv2.imread(stored_img_path, cv2.IMREAD_GRAYSCALE)
      stored_keypoints, stored_descriptors = get_keypoints_and_descriptors(stored_img)
      stored_keypoints = array_to_keypoints(stored_keypoints)
      outlined_img = get_homography(img.copy(), keypoints, descriptors, stored_img_path, stored_keypoints, stored_descriptors, flann)

      cv2.imshow('Outlined card', outlined_img)
      cv2.waitKey(0)
      cv2.destroyAllWindows()

  # Return the top k results
  return match_results[:k]



find_k_nearest_neighbors('/Users/sam/Desktop/computer_vision/final_project/backend/data/human_cards/iron_leaves_ex0.jpeg', 1)