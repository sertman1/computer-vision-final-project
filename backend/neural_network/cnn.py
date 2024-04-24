import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.metrics import f1_score, recall_score, precision_score
import numpy as np

# Define a function to load the data
def load_data(folder):
    # Read the annotations file
    df = pd.read_csv(f'{folder}/_annotations.csv')

    # Create an ImageDataGenerator instance
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    # Generate batches of tensor image data with real-time data augmentation
    data = datagen.flow_from_dataframe(
        dataframe=df,
        directory=f'{folder}/images',
        x_col='filename',
        y_col='class',
        target_size=(296, 416),  # replace with your desired image size
        class_mode='categorical',  # for multi-class classification problems
        batch_size=32  # replace with your desired batch size
    )

    return data

# Load the train, test, and validation data
train_data = load_data('Card Grader.v5i.tensorflow/train')
test_data = load_data('Card Grader.v5i.tensorflow/test')
valid_data = load_data('Card Grader.v5i.tensorflow/valid')

def get_cnn():
  # Define your CNN architecture
  model = Sequential()
  model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(296, 416, 3)))
  model.add(tf.keras.layers.MaxPooling2D((2, 2)))
  model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(tf.keras.layers.MaxPooling2D((2, 2)))
  model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
  model.add(tf.keras.layers.MaxPooling2D((2, 2)))
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(512, activation='relu'))
  model.add(tf.keras.layers.Dropout(0.5))
  model.add(tf.keras.layers.Dense(4, activation='softmax'))  # 4 classes: edge wear, corner wear, scratch, card

  # Compile the model
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  # Train the model
  history = model.fit(train_data, epochs=10, validation_data=valid_data)

  # Save the trained model
  model.save('pokemon_card_detector.h5')

  return model

def main():
    model = tf.keras.models.load_model('pokemon_card_detector.h5')
    if model is None:
      model = get_cnn()
    test_loss, test_accuracy = model.evaluate(test_data)
    print(f'Test loss: {test_loss}')
    print(f'Test accuracy: {test_accuracy}')

    # Get the true labels of the test data
    test_labels = test_data.classes

    # Predict the labels of the test data
    predictions = model.predict(test_data)
    predicted_labels = np.argmax(predictions, axis=1)

    # Calculate precision, recall, and F-score
    precision = precision_score(test_labels, predicted_labels, average='macro')
    recall = recall_score(test_labels, predicted_labels, average='macro')
    f_score = f1_score(test_labels, predicted_labels, average='macro')

    # Print the metrics
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F-score: {f_score}')

if __name__ == "__main__":
   main()