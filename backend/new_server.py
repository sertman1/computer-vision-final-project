import os
import cv2
import pickle
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import matplotlib.pyplot as plt
import sqlite3

app = Flask(__name__)
CORS(app)

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
  conn = sqlite3.connect('image_features.db')
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

  # Return the top k results
  return match_results[:k]

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = os.path.join('./uploads', file.filename)
        file.save(filename)
        match_results = find_k_nearest_neighbors(filename, 3)
        results = []
        for img_path, image_link, matches in match_results:
            img = cv2.imread(img_path)
            # Save the image to a file
            output_filename = os.path.join('./output', os.path.basename(img_path))
            cv2.imwrite(output_filename, img)
            results.append({'image_path': output_filename, 'image_link': image_link, 'matches': matches})
        return jsonify(results)

@app.route('/output/<filename>')
def get_image(filename):
    return send_file(os.path.join('./output', filename), mimetype='image/jpeg')

if __name__ == "__main__":
    process_gold_standard()
    app.run(host='0.0.0.0', port=5000)
