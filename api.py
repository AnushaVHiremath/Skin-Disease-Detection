import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from flask import Flask, request, jsonify
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load disease detection model
disease_model = load_model("savedSkinModel.keras", compile=False)

# Load pre-trained model from TensorFlow Hub for skin type classification
# Using MobileNetV2 as a base model for transfer learning
feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"
feature_extractor_layer = hub.KerasLayer(feature_extractor_url, input_shape=(224, 224, 3), trainable=False)

# Define skin type classification model
skin_type_model = tf.keras.Sequential([
    feature_extractor_layer,
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4, activation='softmax')  # 4 classes: Oily, Dry, Normal, Combination
])

# Compile the model
skin_type_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Skin type class names
skin_type_class_names = ['Oily', 'Dry', 'Normal', 'Combination']

# Image size for models
disease_img_size = (450, 450)
skin_type_img_size = (224, 224)

def predict_disease(model, img):
    """
    Predict skin disease from the input image
    """
    # Disease detection classes
    disease_class_names = [
        'Actinic keratosis', 
        'Tinea Ringworm Candidiasis', 
        'Atopic Dermatitis', 
        'Dermatofibroma', 
        'Squamous cell carcinoma', 
        'Benign keratosis', 
        'Melanocytic nevus', 
        'Melanoma', 
        'Vascular lesion',
        'clean'
    ]
    
    # Preprocess image for disease model
    array = tf.keras.utils.img_to_array(img)
    array = array / 255.0
    img_array = np.expand_dims(array, axis=0)
    
    # Predict
    preds = model.predict(img_array)
    top_prob_index = np.argmax(preds[0])
    top_prob = round(float(preds[0][top_prob_index]) * 100, 2)
    print(top_prob_index)
    top_class = disease_class_names[top_prob_index]
    
    return top_class, top_prob, preds[0]

def predict_skin_type(model, img):
    """
    Predict skin type from the input image using pre-trained model
    """
    # Preprocess image for skin type model
    img_rgb = img.convert('RGB')
    img_resized = img_rgb.resize(skin_type_img_size)
    img_array = img_to_array(img_resized)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    preds = model.predict(img_array)
    top_prob_index = np.argmax(preds[0])
    top_prob = round(float(preds[0][top_prob_index]) * 100, 2)
    top_class = skin_type_class_names[top_prob_index]
    
    return top_class, top_prob, preds[0]

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint for skin disease and skin type
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file provided'}), 400
    
    try:
        # Open the image
        img = Image.open(io.BytesIO(file.read()))
        
        # Resize images for respective models
        disease_img = img.resize(disease_img_size)
        
        # Predict disease
        top_disease, disease_prob, disease_preds = predict_disease(disease_model, disease_img)
        
        # Predict skin type (Note: This is a placeholder prediction)
        top_skin_type, skin_type_prob, skin_type_preds = predict_skin_type(skin_type_model, img)
        
        # Prepare response
        response = {
            'disease': {
                'top_class': top_disease,
                'top_prob': disease_prob,
                'all_predictions': dict(zip(
                    ['Actinic keratosis', 'Tinea Ringworm Candidiasis', 'Atopic Dermatitis', 
                     'Dermatofibroma', 'Squamous cell carcinoma', 'Benign keratosis', 
                     'Melanocytic nevus', 'Melanoma', 'Vascular lesion'],
                    [round(float(pred), 2) for pred in disease_preds]
                ))
            },
            'skin_type': {
                'top_class': top_skin_type,
                'top_prob': skin_type_prob,
                'all_predictions': dict(zip(
                    skin_type_class_names,
                    [round(float(pred), 2) for pred in skin_type_preds]
                ))
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)