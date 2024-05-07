import streamlit as st
from PIL import Image
from model import Model
from chatbot import FactGenerator

st.set_page_config(page_title='Date Fruit Classification App', layout='wide', initial_sidebar_state='expanded')

st.title('🌴 Date Fruit Classification App')

st.markdown("""
This app uses machine learning to classify different types of date fruits. Simply upload an image of a date fruit, and let the magic happen!
""")

@st.cache_resource
def load_resources():
    model = Model()
    llm = FactGenerator()
    chain = llm.get_chain()
    return model, chain

model, chain = load_resources()

# Check if the 'uploaded_files' list exists in the session state. If not, initialize it.
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

uploaded_file = st.file_uploader("Choose an image...", type="jpg", accept_multiple_files=False)

# Function to display images and classification
def display_image_and_classification(image_file):
    left, right = st.columns(2)
    with left:
        image = Image.open(image_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

    with right:
        with st.spinner('Classifying...'):
            label = model.predict(image)
            st.write('Predicted class: **%s**' % label)
            st.write_stream(chain.stream(label))

if uploaded_file is not None:
    st.session_state.uploaded_files.append(uploaded_file)
    display_image_and_classification(uploaded_file)
        
for uploaded_file_item in st.session_state.uploaded_files[:-1]:  # Skip the last one as it's already displayed
    display_image_and_classification(uploaded_file_item)