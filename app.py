from sklearn.kernel_approximation import PAIRWISE_KERNEL_FUNCTIONS
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.keras.layers import PReLU, ELU , LeakyReLU
import cv2
# Load the ML model

st.set_page_config(
    page_title="Alzheimer's Disease Prediction",
    page_icon="🌐",  # Use a valid emoji
    layout="centered"
)

@st.cache_resource
def load_model():
    custom_objects = {"PReLU": PReLU, "ELU": ELU, "LeakyReLU": LeakyReLU}
    model = tf.keras.models.load_model("model-ver-1.h5", custom_objects=custom_objects)
    return model

model = load_model()

def predict_image(image):
    if isinstance(image, str):
        image = cv2.imread(image)
    # Preprocess the image to match model requirements
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize pixel values to [0, 1]
    gray = gray / 255.0

    # Resize the image to (240, 240)
    gray = cv2.resize(gray, (240, 240))

    # Expand dimensions to add batch size (1, 240, 240, 1) for model input
    gray = np.expand_dims(gray, axis=-1)  # Add channel dimension
    gray = np.expand_dims(gray, axis=0)  # Add batch dimension

    # Make predictions using the model
    predictions = model.predict(gray)

# Streamlit app


st.markdown(
    """
    This application uses a CNN model to predict the type of Alzheimer's disease 
    from brain CT scans. Upload a CT scan to get a prediction.
    """,
    unsafe_allow_html=True,
)

# File upload
uploaded_file = st.file_uploader("Upload a Brain CT Scan Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file)

        # Display uploaded image
        st.image(image, caption="Uploaded Brain CT Scan", use_column_width=True)

        st.markdown("### Prediction")
        with st.spinner("Analyzing the image..."):
            predictions = predict_image(image)
            class_names = ["AD", "CN", "EMCI", "LMCI","MCI"]  # Update as per your model
            prediction_idx = np.argmax(predictions)
            prediction_class = class_names[prediction_idx]
            confidence = predictions[0][prediction_idx] * 100

            st.write(f"**Prediction:** {prediction_class}")
            st.write(f"**Confidence:** {confidence:.2f}%")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Footer styling
st.markdown(
    """
    <style>
    footer {
        visibility: hidden;
    }
    footer:after {
        content: 'Alzheimer's Prediction App | Designed by YourName';
        visibility: visible;
        display: block;
        position: relative;
        padding: 5px;
        top: 2px;
        font-size: 12px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)