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
        img2 = cv2.polylines(img.copy(),[np.int32(dst)],True,(0, 0, 255),30, cv2.LINE_AA)

        print("Perimeter of the card:")
        for i, point in enumerate(dst):
          print(f"Point {i+1}: {point[0]}")

        return img2
    else:
        print("Not enough matches are found - %d/%d" % (len(good_matches), MIN_MATCH_COUNT))

def find_nearest_neighbor(img_path):
  img = pre_process_image(img_path)
  keypoints, descriptors = get_keypoints_and_descriptors(img)
  keypoints = array_to_keypoints(keypoints)

  # Initialize the Brute-Force matcher
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

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

    # Match the descriptors using the Brute-Force matcher
    matches = bf.match(descriptors, des)

    # Sort the matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Only keep the top match
    if matches:
        top_match = matches[0]
        match_results.append((stored_img_path, image_link, top_match.distance))

  print(match_results)

  # Sort the results by match quality (lower distance is better)
  match_results = sorted(match_results, key=lambda x: x[2])

  # Return the top result
  if match_results:
      return match_results[0]
  else:
      return None
