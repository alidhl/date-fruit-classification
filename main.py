import streamlit as st
from PIL import Image
import numpy as np
from model import Model
# Load the model
model = Model()

st.title('Image Classification App')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = model.predict(image)
    st.write('%s (%.2f%%)' % (label[1], label[2]*100))