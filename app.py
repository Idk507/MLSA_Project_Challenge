import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from skimage import transform

# Define photo size
photo_size = 224

# Function to load and preprocess image
def load_image_from_path(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32') / 255
    np_image = transform.resize(np_image, (photo_size, photo_size, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

# Load the model
vgg_model = load_model('my_model.keras')

# Define the Streamlit app
def main():
    st.title("Image Classification App")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Classify the image
        with st.spinner('Classifying...'):
            img_array = load_image_from_path(uploaded_file)
            prediction = vgg_model.predict(img_array).argmax()

        # Display the prediction result
        if prediction == 1:
            st.write("Prediction: Autistic")
        else:
            st.write("Prediction: Non-Autistic")

# Run the app
if __name__ == "__main__":
    main()
