import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('flower_classification_model.h5')

# Define class indices directly within the script
class_indices = {
    "bougainvillea": 0,
    "daisies": 1,
    "garden_roses": 2,
    "gardenias": 3,
    "hibiscus": 4,
    "hydrangeas": 5,
    "lilies": 6,
    "orchids": 7,
    "peonies": 8,
    "tulip": 9
}

# Reverse the class indices to get class labels
class_labels = {v: k for k, v in class_indices.items()}

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Function to plot prediction comparison
def plot_prediction_comparison(predictions, class_labels, top_n):
    flower_types = np.array(list(class_labels.values()))
    predicted_probs = predictions[0]

    top_indices = predicted_probs.argsort()[-top_n:][::-1]
    top_flower_types = flower_types[top_indices]
    top_predicted_probs = predicted_probs[top_indices]

    plt.figure(figsize=(12, 6))
    plt.bar(top_flower_types, top_predicted_probs, color='skyblue')
    plt.xlabel('Flower Type')
    plt.ylabel('Prediction Probability')
    plt.title('Top Prediction Probabilities')
    plt.ylim(0, 1)
    plt.tight_layout()

    st.pyplot(plt)

# Streamlit app
st.title("Flower Classification Application")

# Sidebar
st.sidebar.title("Options")
max_flower_types = len(class_labels)
top_n = st.sidebar.number_input("Select number of top predictions to display:", min_value=1, max_value=max_flower_types, value=max_flower_types, step=1)

# Main content
st.header("Flower Image Classification")
uploaded_file = st.file_uploader("Upload an image file for classification", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Classify the image
    with st.spinner("Classifying..."):
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions, axis=1)
        predicted_label = class_labels[predicted_class[0]]
        confidence = np.max(predictions) * 100

    # Highlight the prediction result
    st.markdown(f"<h2 style='color: #4CAF50; text-align: center;'>Prediction: <span style='color: #FF5722;'>{predicted_label}</span> ({confidence:.2f}%)</h2>", unsafe_allow_html=True)
    
    # Plot prediction comparison
    plot_prediction_comparison(predictions, class_labels, top_n)