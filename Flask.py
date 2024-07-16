!pip install pyngrok

#mount drive if you are working in colab notebook
from google.colab import drive
drive.mount('/content/drive')


#log in into ngrok and get authtoken
from pyngrok import ngrok

# Replace with your ngrok authtoken
authtoken = "<YOUR_AUTH_TOKEN>"

ngrok.set_auth_token(authtoken)

from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from pyngrok import ngrok

app = Flask(__name__)

# Load the trained model
model_path = '/content/drive/MyDrive/H5model/my_model2.h5'  # Adjust the path as needed
model = tf.keras.models.load_model(model_path)

img_size = 128  # Ensure this matches the input size expected by your model

@app.route('/')
def home():
    return "Welcome to the Plant Species Classifier API. Use the /predict endpoint to classify plant images."

import io

# Define a mapping from class index to plant species names
class_labels = {
    0: 'Sweet Potatoes',
    1: 'Water Apple',
    2: 'Spinach',
    3: 'Tobacco',
    4: 'Watermelon'
    # Add more mappings as necessary
}

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = image.load_img(io.BytesIO(file.read()), target_size=(img_size, img_size))  # Use io.BytesIO
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]

 # Map the predicted class index to the species name
    predicted_species = class_labels.get(predicted_class, 'Unknown')

    return jsonify({'class_index': int(predicted_class), 'species': predicted_species})




if __name__ == '__main__':
    # Create a tunnel to the localhost server
    public_url = ngrok.connect(5000)
    print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:5000\"".format(public_url))
    # app.run(debug=True)

    app.run()
