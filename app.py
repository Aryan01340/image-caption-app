import streamlit as st
from PIL import Image
from utils import generate_caption

st.title("🖼️ Image Caption Generator")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Generating caption...")
    caption = generate_caption(image)

    st.success("Caption:")
    st.write(caption)