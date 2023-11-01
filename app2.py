import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np

models = st.sidebar.selectbox("Select a model",['VGG16','DenseNet201','EfficientNetV2M'])

if models == "VGG16":
    # Load the pre-trained model
    model = load_model(r"C:\Users\edecr\Desktop\Project\project\vgg16model.h5")
    
    # Define the labels for classification
    labels = ['Does not have Tumor', 'Has Tumor']
    
    # Function to preprocess the image
    def preprocess_image(image):
        image = image.resize((256, 256))
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        return image
    
    # Function to make predictions
    def predict(image):
        image = preprocess_image(image)
        prediction = model.predict(image)
        return labels[np.argmax(prediction)]
    
    # Streamlit app
    def main():
        st.title("Brain Tumor Image Classification (VGG16)")
        st.write("Upload an image and the app will classify whether it contains a brain tumor or not.")
    
        uploaded_file = st.file_uploader("Choose an image...", type=["tif"])
        if uploaded_file is not None:
            
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            prediction = predict(image)
        if st.button("Check for Tumor"):
        # Display the prediction
            if prediction == 'Has Tumor':
                st.error('Has Tumor')
            else:
                st.success('Does not have Tumor')
    
            #st.write("Prediction:", prediction)
    
    if __name__ == '__main__':
        main()
    
elif models == "DenseNet201":
        # Load the pre-trained model
    model = load_model(r"C:\Users\edecr\Desktop\Project\project\DenseNet201model.h5")
    
    # Define the labels for classification
    labels = ['Does not have Tumor', 'Has Tumor']
    
    # Function to preprocess the image
    def preprocess_image(image):
        image = image.resize((256, 256))
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        return image
    
    # Function to make predictions
    def predict(image):
        image = preprocess_image(image)
        prediction = model.predict(image)
        return labels[np.argmax(prediction)]
    
    # Streamlit app
    def main():
        st.title("Brain Tumor Image Classification (DenseNet201)")
        st.write("Upload an image and the app will classify whether it contains a brain tumor or not.")
    
        uploaded_file = st.file_uploader("Choose an image...", type=["tif"])
        if uploaded_file is not None:
            
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            prediction = predict(image)
        if st.button("Check for Tumor"):
        # Display the prediction
            if prediction == 'Has Tumor':
                st.error('Has Tumor')
            else:
                st.success('Does not have Tumor')
    
            #st.write("Prediction:", prediction)
    
    if __name__ == '__main__':
        main()

elif models == "EfficientNetV2M":
        # Load the pre-trained model
    model = load_model(r"C:\Users\edecr\Desktop\Project\project\efficientNetV2Mmodel.h5")
    
    # Define the labels for classification
    labels = ['Does not have Tumor', 'Has Tumor']
    
    # Function to preprocess the image
    def preprocess_image(image):
        image = image.resize((256, 256))
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        return image
    
    # Function to make predictions
    def predict(image):
        image = preprocess_image(image)
        prediction = model.predict(image)
        return labels[np.argmax(prediction)]
    
    # Streamlit app
    def main():
        st.title("Brain Tumor Image Classification (EfficientNetV2M)")
        st.write("Upload an image and the app will classify whether it contains a brain tumor or not.")
    
        uploaded_file = st.file_uploader("Choose an image...", type=["tif"])
        if uploaded_file is not None:
            
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            prediction = predict(image)
        if st.button("Check for Tumor"):
        # Display the prediction
            if prediction == 'Has Tumor':
                st.error('Has Tumor')
            else:
                st.success('Does not have Tumor')
    
            #st.write("Prediction:", prediction)
    
    if __name__ == '__main__':
        main()