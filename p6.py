import pandas as pd
import numpy as np
import requests
import opencv as cv
from PIL import Image
import matplotlib.pyplot as plt
import random 
import json
import os
import streamlit  as st
from tensorflow.keras.applications import EfficientNetB0 
import pickle



uploaded_file = st.file_uploader("Choose an image...", type="jpg")
model = EfficientNetB0(include_top=False, weights='imagenet', pooling='avg')

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    im_array = np.array(image)
    
    im_feat = model(im_array).numpy() 
    
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Strandardisation std
    std_model = pickle.load(open('std_model.pkl', 'rb'))
    std_X = std_model.transform(im_feat)
    
    # PCA
    pca_model = pickle.load(open('pca_model.pkl', 'rb'))
    pca_X =  pca_model.transform(std_X)
    
    # Predidction        
    clf_model = pickle.load(open('clf_model.pkl', 'rb'))
    label = clf_model.predict(pca_X)
    st.write(label[0])
    #st.write('%s (%.2f%%)' % (label[1], label[2]*100))
    
    

    
# =============================================================================
# 
# st.write(im_feat)
# 
# =============================================================================
