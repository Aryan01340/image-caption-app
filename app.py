import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import requests
from io import BytesIO

st.title("🖼️ Image Caption AI")
st.write("Upload image → Get AI caption!")

# Load model (cached)
@st.cache_resource
def load_blip():
    return BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base"), \
           BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

uploaded = st.file_uploader("Choose image", type=['png','jpeg','jpg'])
if uploaded:
    image = Image.open(uploaded).convert('RGB')
    st.image(image, use_column_width=True)
    
    processor, model = load_blip()
    
    with st.spinner('Generating...'):
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs, max_length=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
    
    st.success(f"**{caption}** 🎉")
