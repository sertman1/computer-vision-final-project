# Pokémon Card Condition Detector

## Computer Vision Final Project
By Samuel A. Ertman

### Running the software locally

* To run the front-end React webapp (localhost 3000):
`cd frontend && npm i && npm run start`
* To run the back-end Flask server (localhost 5000):
`cd backend && npm run dev`

Make sure you have run the script `process_gold_standard.py` at least once to create a local SQLite database and index the keypoints and descriptors of the cards the system is capable of recognizing. Currently (as of April 2024), it uses the 218 latest cards from the Temporal Forces expansion. High-resolution photos of each card (733x1024x3) can be found in `data/temporal_forces`, and each card's corresponding link can be found in `data/links/temporal_forces.txt`. Going forward, this project will constantly be adding cards to this dataset.