import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import numpy as np
import pickle
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model

# Load model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM

caption_model = load_model(
    'model.h5',
    compile=False,
    custom_objects={'LSTM': LSTM}
)

# Load tokenizer
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

max_length = 20

# Load VGG16 model
base_model = VGG16(weights='imagenet')
feature_extractor = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
# Extract features
def extract_feature(image):
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = feature_extractor.predict(image, verbose=0)
    return feature

# Generate caption
def generate_caption(image):
    photo = extract_feature(image)
    in_text = 'startseq'

    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        yhat = caption_model.predict([photo, sequence])
        yhat = np.argmax(yhat)

        word = None
        for w, index in tokenizer.word_index.items():
            if index == yhat:
                word = w
                break

        if word is None:
            break

        in_text += ' ' + word

        if word == 'endseq':
            break

    final_caption = in_text.replace('startseq', '').replace('endseq', '')
    return final_caption.strip()