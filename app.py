import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.cluster import KMeans
from PIL import Image
import tensorflow as tf

# Custom color mapping dictionary
COLOR_MAP = {
    'red': '#FF0000',
    'green': '#008000',
    'blue': '#0000FF',
    'yellow': '#FFFF00',
    'purple': '#800080',
    'orange': '#FFA500',
    'pink': '#FFC0CB',
    'brown': '#A52A2A',
    'black': '#000000',
    'white': '#FFFFFF',
    'gray': '#808080',
    'cyan': '#00FFFF',
    'magenta': '#FF00FF',
    'lime': '#00FF00',
    'navy': '#000080',
    'teal': '#008080',
    'olive': '#808000',
    'maroon': '#800000',
    'violet': '#EE82EE',
    'turquoise': '#40E0D0',
    'coral': '#FF7F50',
    'indigo': '#4B0082',
    'salmon': '#FA8072',
    'crimson': '#DC143C',
    'gold': '#FFD700',
    'silver': '#C0C0C0',
    'beige': '#F5F5DC',
    'lavender': '#E6E6FA',
    'khaki': '#F0E68C',
    'slateblue': '#6A5ACD',  # Close to #4f5685
    'darkslateblue': '#483D8B'
}

def hex_to_rgb(hex_color):
    """
    Convert hex color to RGB tuple.
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def get_color_name(hex_color):
    """
    Find the closest color name from the custom COLOR_MAP.
    """
    try:
        rgb_color = hex_to_rgb(hex_color)
        min_distance = float('inf')
        closest_name = 'Unknown'
        
        for name, hex_val in COLOR_MAP.items():
            css_rgb = hex_to_rgb(hex_val)
            distance = sum((c1 - c2) ** 2 for c1, c2 in zip(rgb_color, css_rgb)) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_name = name
        return closest_name
    except Exception as e:
        st.warning(f"Color name error: {e}")
        return 'Unknown'

# Define focal loss function (multi-class, as per your models)
def focal_loss_fn(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Multi-class focal loss function for imbalanced classification.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
    cross_entropy = -y_true * tf.math.log(y_pred)
    weight = alpha * y_true * tf.pow(1 - y_pred, gamma)
    focal_loss = weight * cross_entropy
    return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))

# Load custom models with custom_objects
nationality_model_path = "nationality_model (1).h5"
emotion_model_path = "emotion_model (1).keras"
age_model_path = "age_class_model (1).h5"

try:
    custom_objects = {'focal_loss_fn': focal_loss_fn}
    nationality_model = load_model(nationality_model_path, custom_objects=custom_objects)
    emotion_model = load_model(emotion_model_path, custom_objects=custom_objects)
    age_model = load_model(age_model_path, custom_objects=custom_objects)
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Define class labels (update these based on your model's output classes)
NATIONALITY_CLASSES = ['Indian', 'African', 'United States', 'Other']  # Adjust based on your model
EMOTION_CLASSES = ['happy', 'sad', 'angry', 'neutral', 'surprise', 'fear', 'disgust']  # Adjust based on your model
AGE_CLASSES = ['0-18', '19-30', '31-50', '51+']  # Adjust based on your model

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess image for model input with specified target size.
    """
    img = np.array(image)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_nationality(image):
    """
    Predict nationality using the custom model (assuming 224x224 input).
    """
    try:
        img = preprocess_image(image, target_size=(224, 224))  # Adjust if needed
        pred = nationality_model.predict(img, verbose=0)
        nationality_idx = np.argmax(pred, axis=1)[0]
        return NATIONALITY_CLASSES[nationality_idx]
    except Exception as e:
        st.warning(f"Nationality prediction error: {e}")
        return 'Other'  # Fallback

def predict_emotion(image):
    """
    Predict emotion using the custom model (requires 128x128 input).
    """
    try:
        img = preprocess_image(image, target_size=(128, 128))  # Emotion model expects 128x128
        pred = emotion_model.predict(img, verbose=0)
        emotion_idx = np.argmax(pred, axis=1)[0]
        return EMOTION_CLASSES[emotion_idx]
    except Exception as e:
        st.warning(f"Emotion prediction error: {e}")
        return 'Unknown'  # Fallback

def predict_age(image):
    """
    Predict age using the custom model (assuming 224x224 input).
    """
    try:
        img = preprocess_image(image, target_size=(224, 224))  # Adjust if needed
        pred = age_model.predict(img, verbose=0)
        age_idx = np.argmax(pred, axis=1)[0]
        return AGE_CLASSES[age_idx]
    except Exception as e:
        st.warning(f"Age prediction error: {e}")
        return 'Unknown'  # Fallback

def predict_dress_color(image):
    """
    Predict dress color by analyzing the dominant color in the lower half of the image.
    """
    try:
        img = np.array(image)
        h, w = img.shape[:2]
        lower_half = img[h//2:, :]
        pixels = lower_half.reshape(-1, 3)
        kmeans = KMeans(n_clusters=3, n_init=10)
        kmeans.fit(pixels)
        dominant_color = kmeans.cluster_centers_[kmeans.labels_].mean(axis=0)
        dominant_color = dominant_color[::-1]  # BGR to RGB
        hex_color = '#%02x%02x%02x' % tuple(map(int, dominant_color))
        color_name = get_color_name(hex_color)
        return hex_color, color_name
    except Exception as e:
        st.warning(f"Dress color prediction error: {e}")
        return 'Unknown', 'Unknown'  # Fallback

# Streamlit App
st.title("Nationality Detection Model")
st.write("Upload an image to predict nationality, emotion, and other attributes based on the detected nationality.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Predictions
    nationality = predict_nationality(image)
    emotion = predict_emotion(image)

    # Display results
    st.subheader("Prediction Results")
    st.write(f"**Nationality**: {nationality}")
    st.write(f"**Emotion**: {emotion}")

    if nationality == 'Indian':
        age = predict_age(image)
        hex_color, color_name = predict_dress_color(image)
        st.write(f"**Age**: {age}")
        st.write(f"**Dress Color**: {color_name} ({hex_color})")
    elif nationality == 'United States':
        age = predict_age(image)
        st.write(f"**Age**: {age}")
    elif nationality == 'African':
        hex_color, color_name = predict_dress_color(image)
        st.write(f"**Dress Color**: {color_name} ({hex_color})")