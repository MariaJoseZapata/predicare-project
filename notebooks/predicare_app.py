import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, color, exposure, io
from PIL import Image
from scipy import ndimage
import tensorflow as tf 
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Function to apply Local Binary Pattern (LBP) algorithm
def apply_lbp(image, P=8, R=1, method='uniform'):
    gray = color.rgb2gray(image)
    lbp = feature.local_binary_pattern(gray, P=P, R=R, method=method)
    lbp = np.expand_dims(lbp, axis= -1)
    return lbp

# Function to apply Histogram of Gradients (HOG) algorithm
def apply_hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    gray = color.rgb2gray(image)
    # Calculate HOG features using skimage's hog
    hog_features, hog_image = feature.hog(gray, orientations=orientations,
                                  pixels_per_cell=pixels_per_cell,
                                  cells_per_block=cells_per_block,
                                  visualize=True)

    # Enhance the contrast of HOG features for better visualization
    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return hog_features, hog_image

def preprocess_unet(image, lbp, hog_image):
    height, width, channels = 128,128,3
    processed_image = cv2.resize(image, (width, height))
    processed_lbp = cv2.resize(lbp, (width, height))
    processed_lbp = np.expand_dims(processed_lbp, axis=-1)
    processed_hog = cv2.resize(hog_image, (128, 128))
    processed_hog = np.expand_dims(processed_hog, axis=-1)
    processed_image_batch = np.expand_dims(processed_image, axis=0)
    processed_lbp_batch = np.expand_dims(processed_lbp, axis=0)
    processed_hog_batch = np.expand_dims(processed_hog, axis=0)   

    return processed_image_batch, processed_lbp_batch, processed_hog_batch

def apply_unet(image, lbp, hog_image, unet_model):
    # Preprocess the image (you may need to adapt this based on your model requirements)
    processed_image, processed_lbp, processed_hog = preprocess_unet(image, lbp, hog_image)
    # Concatenate the processed LBP and HOG images along the channels axis
    input_data = [processed_image, processed_lbp, processed_hog]

    # Predict mask using the U-Net model
    predicted_mask = unet_model.predict(input_data)
    predicted_mask_t = (predicted_mask > 0.5).astype(np.uint8)
    return predicted_mask, predicted_mask_t

def preprocess_classification(processed_image_batch, processed_lbp_batch, processed_hog_batch, predicted_mask_t):
    new_size = (197,197)
    zoom_factors = (new_size[0] / processed_image_batch.shape[1], new_size[1] / processed_image_batch.shape[2])
    processed_image_resized = np.array([ndimage.zoom(img, zoom_factors + (1,), order=1) for img in processed_image_batch])
    processed_lbp_resized = np.array([ndimage.zoom(img, zoom_factors + (1,), order=1) for img in processed_lbp_batch])
    processed_hog_resized = np.array([ndimage.zoom(img, zoom_factors + (1,), order=1) for img in processed_hog_batch])
    predicted_mask_t_resized = np.array([ndimage.zoom(img, zoom_factors + (1,), order=1) for img in predicted_mask_t])
    processed_image_resized_class = preprocess_input(processed_image_resized)
    return processed_image_resized_class, processed_lbp_resized, processed_hog_resized, predicted_mask_t_resized


def apply_classification(processed_image_resized_class, processed_lbp_resized, processed_hog_resized, predicted_mask_t_resized, class_model):
    pred = class_model.predict([processed_image_resized_class, processed_lbp_resized, processed_hog_resized, predicted_mask_t_resized])
    return pred

# Streamlit app
st.title("PrediCare - AI Breast Cancer Diagnosis")

# Upload image through the Streamlit UI
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
show_overlay = False

# Check if an image is uploaded
if uploaded_file is not None:
    # Read the image
    #image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    image = io.imread(uploaded_file)
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title('Original Image')
    ax.axis('off')
    st.pyplot(fig)
    # Display the original image
    #st.image(image, caption="Original Image", use_column_width=True)

    # Main header for Image Preprocessing
    st.header("Results of Image Processing")

    st.sidebar.header('Image Processing')
    # Options for LBP algorithm
    st.sidebar.subheader("Local Binary Pattern (LBP)")
    lbp_radius = st.sidebar.slider("Radius", 1, 10, 1, key="lbp_radius")
    lbp_neighbors = st.sidebar.slider("Neighbors", 1, 10, 8, key="lbp_neighbors")

    # Options for HOG algorithm
    st.sidebar.subheader("Histogram of Gradient (HOG)")
    hog_orientations = st.sidebar.slider("Orientations", 1, 16, 9, key="hog_orientations")
    hog_pixels_per_cell = st.sidebar.slider("Pixels per Cell", 4, 16, 8, key="hog_pixels_per_cell")
    hog_cells_per_block = st.sidebar.slider("Cells per Block", 1, 5, 2, key="hog_cells_per_block")

    button_mask = st.sidebar.button('Identify Anomalies!', key='button_mask')
   
    st.sidebar.markdown("&nbsp;")  # Add one empty line
    st.sidebar.header('AI-based Diagnosis')
    button_diagnosis = st.sidebar.button('Perform Diagnosis!', key='diagnosis')

    # Apply LBP and HOG algorithms
    lbp_result = apply_lbp(image, R=lbp_radius, P=lbp_neighbors)
    _, hog_result = apply_hog(image, orientations=hog_orientations,
                           pixels_per_cell=(hog_pixels_per_cell, hog_pixels_per_cell),
                           cells_per_block=(hog_cells_per_block, hog_cells_per_block))


    # Display the results from LBP and HOG processing
    fig_lbp, lbp = plt.subplots()
    lbp.imshow(lbp_result, cmap='gray')
    lbp.set_title('LBP-processed Image')
    lbp.axis('off')
    st.pyplot(fig_lbp)   

    fig_hog, hog = plt.subplots()
    hog.imshow(hog_result, cmap='gray')
    hog.set_title('HOG-processed Image')
    hog.axis('off')
    st.pyplot(fig_hog)

    # Load the pre-trained U-Net model
    unet_model = load_model('/Users/thiyennguyen/Documents/Bildung/DataScience-Bootcamp/predicare-project/models/hog+lbp_original_unet.h5')
    # Preprocessed images for U-net
    _, predicted_mask_t = apply_unet(image, lbp_result, hog_result, unet_model)

    class_model = load_model('/Users/thiyennguyen/Documents/Bildung/DataScience-Bootcamp/predicare-project/models/classification_pretrainedResNet50_orig+lbp+hog+mask_2.h5')


    # Button to apply U-Net and visualize predicted mask
    if button_mask:
              
        # Display the predicted mask
        resized_image_overlay = cv2.resize(image, (128, 128))

        fig_overlay, overlay = plt.subplots()
        overlay.imshow(resized_image_overlay)
        overlay.imshow(predicted_mask_t[0, :, :, 0], cmap='jet', alpha=0.5)  # Adjust the alpha for transparency
        overlay.set_title('Original Image with Anomalies (red color)')
        overlay.axis('off')
        st.pyplot(fig_overlay)  
        show_overlay = True
    
    if button_diagnosis:

        st.header("Results of AI-based Diagnosis") 
        
        processed_image_batch, processed_lbp_batch, processed_hog_batch = preprocess_unet(image, lbp_result, hog_result)
        processed_image_resized_class, processed_lbp_resized, processed_hog_resized, predicted_mask_t_resized = preprocess_classification(processed_image_batch, processed_lbp_batch, processed_hog_batch, predicted_mask_t)
        
        prediction = apply_classification(processed_image_resized_class, processed_lbp_resized, processed_hog_resized, predicted_mask_t_resized, class_model)
        

        # Sort probabilities in descending order
        sorted_indices = np.argsort(prediction[0])[::-1]

        class_labels = ["normal", "benign", "malignant"]

        st.subheader("Probabilites for each class:")
        for i in sorted_indices:
            prob_percent = prediction[0][i] * 100
            st.write(f"{class_labels[i]}: {prob_percent:.2f}%")



      