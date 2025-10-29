# Image Classifier Web App

A Flask-based web application for classifying images of animals and objects using a pre-trained MobileNetV2 model.

## Features

- Upload images via web interface
- Classify images into 1000+ categories (animals, objects, etc.)
- Display top 5 predictions with confidence scores
- Responsive web design

## Installation

1. Clone or download this repository
2. Navigate to the project directory
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Flask application:
   ```
   python app.py
   ```
2. Open your web browser and go to `http://127.0.0.1:5000/`
3. Upload an image and click "Classify Image"
4. View the top predictions

## Technologies Used

- Flask (web framework)
- TensorFlow/Keras (machine learning)
- MobileNetV2 (pre-trained model)
- HTML/CSS/JavaScript (frontend)

## Model

This app uses MobileNetV2, a lightweight convolutional neural network pre-trained on the ImageNet dataset. It can classify images into over 1000 categories including various animals (cats, dogs, birds, etc.) and objects (cars, furniture, food, etc.).

## License

This project is open source and available under the MIT License.
