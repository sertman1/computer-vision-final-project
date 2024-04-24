import os
import cv2
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import image_processing as image_processing

app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = os.path.join('uploads', file.filename)
        file.save(filename)
        # just returning best match for now; highly confident in system
        match_results = image_processing.find_nearest_neighbor(filename)
        results = []
        print("recieved request")
        for img_path, image_link, matches in match_results:
            img = cv2.imread(img_path)
            # Save the image to a file
            output_filename = os.path.join('./output', os.path.basename(img_path))
            cv2.imwrite(output_filename, img)
            results.append({'image_path': output_filename, 'image_link': image_link, 'matches': matches})
        return jsonify(results)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000)
