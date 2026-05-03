import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

st.set_page_config(page_title="🖼️ Image Caption", layout="wide")

@st.cache_resource
def load_model():
    """Load BLIP model once"""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

st.title("🖼️ AI Image Caption Generator")
st.markdown("Upload any image → Get smart AI captions instantly!")

# Sidebar for upload
st.sidebar.header("📤 Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose image", type=["png", "jpg", "jpeg"])
url_input = st.sidebar.text_input("Or image URL:")

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    source = "upload"
elif url_input:
    try:
        response = requests.get(url_input)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        source = "URL"
    except:
        st.error("❌ Invalid image URL")
        st.stop()
else:
    # Demo image
    st.info("👆 Upload an image or paste URL")
    image = Image.new('RGB', (400, 300), color='lightblue')
    st.stop()

# Main layout
col1, col2 = st.columns([1, 2])

with col1:
    st.image(image, caption=f"Source: {source}", use_column_width=True)

with col2:
    st.subheader("🤖 Generated Caption")
    
    if 'processor' not in locals():
        processor, model = load_model()
    
    with st.spinner("AI is thinking..."):
        inputs = processor(image, return_tensors="pt")
        generated_ids = model.generate(**inputs, max_new_tokens=50)
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    st.markdown(f"**{caption}**")
    st.balloons()

st.markdown("---")
st.caption("Powered by BLIP + Hugging Face Transformers")
