import os
import sqlite3
import pickle
import image_processing

def process_gold_standard():
  directory = 'data/temporal_forces'
  link_file = 'data/links/temporal_forces_links.txt'
  with open(link_file, 'r') as f:
    image_links = f.read().split(',')

  # Connect to the SQLite database (it will be created if it doesn't exist)
  conn = sqlite3.connect('db/image_features.db')
  c = conn.cursor()

  # Create a table to store the image features
  c.execute('''
      CREATE TABLE IF NOT EXISTS features (
          image_path TEXT,
          image_link TEXT,
          keypoints BLOB,
          descriptors BLOB
      )
  ''')

  filenames = os.listdir(directory)
  filenames.sort()

  for i, filename in enumerate(filenames):
    if filename.endswith('.png') or filename.endswith('.jpg'):
      img_path = os.path.join(directory, filename)

      if i < len(image_links):
        image_link = image_links[i]
      else:
         print(f"No link found for {filename}")

      image = image_processing.pre_process_image(img_path)
      keypoints, descriptors = image_processing.get_keypoints_and_descriptors(image)

      # Serialize the keypoints and descriptors
      serialized_keypoints = pickle.dumps(keypoints)
      serialized_descriptors = pickle.dumps(descriptors)

      # Insert the image path and features into the database
      c.execute('''
          INSERT INTO features (image_path, image_link, keypoints, descriptors)
          VALUES (?, ?, ?, ?)
      ''', (img_path, image_link, serialized_keypoints, serialized_descriptors))

  # Commit the changes and close the connection
  conn.commit()
  conn.close()

if __name__ == "__main__":
  process_gold_standard()