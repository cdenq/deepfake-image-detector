# -------------------
# IMPORTS
# -------------------
import streamlit as st
import numpy as np
import pickle
import os
from PIL import Image, ImageOps
from streamlit_image_select import image_select
from tensorflow.keras.models import model_from_json
import time

# -------------------
# MAIN
# -------------------
def main():
  # main page
st.set_page_config(layout="wide")
st.title("Deepfake Detector:")


# function to load and cache pretrained model
@st.cache_resource()
def load_model():
    path = "../dffnetv2B0"
    # Model reconstruction from JSON file
    with open(path + '.json', 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(path + '.h5')
    return model

 
# function to preprocess an image and get a prediction from the model
def get_prediction(model, image):
    
    open_image = Image.open(image)
    resized_image = open_image.resize((256, 256))
    np_image = np.array(resized_image)
    reshaped = np.expand_dims(np_image, axis=0)

    predicted_prob = model.predict(reshaped)[0][0]
    
    if predicted_prob >= 0.5:
        return f"Real, Confidence: {str(predicted_prob)[:4]}"
    else:
        return f"Fake, Confidence: {str(1 - predicted_prob)[:4]}"

# generate selection of sample images 
@st.cache_data()
def load_images():
  real_images = ["images/Real/" + x for x in os.listdir("images/Real/")]
  fake_images = ["images/Fake/" + x for x in os.listdir("images/Fake/")]
  image_library = real_images + fake_images
  image_selection = np.random.choice(image_library, 20, replace=False)

  return image_selection

# load model
classifier = load_model()
images = load_images()

def game_mode():
  st.header("Game Mode")
  st.subheader("Can you beat the model?")

  # from streamlit_image_select docs - https://github.com/jrieke/streamlit-image-select
  selected_image = image_select(
    "Click on an image below to guess if it is real of fake:", 
    images,
    return_value="index")
  prediction = get_prediction(classifier, images[selected_image])
  true_label = 'Fake' if 'fake' in images[selected_image].lower() else 'Real'

  #st.text(true_label)

  st.subheader("Is this image real or fake?")
  st.image(images[selected_image])
    
  if st.button("It's Real"):
    st.text("You guessed:")
    st.subheader("*Real*")
    st.text("The Deepfake Detector model guessed...")
    time.sleep(1)
    st.subheader(f"*{prediction}*")
    st.text("The truth is...")
    time.sleep(1)
    st.subheader(f"***It's {true_label}!***")

  if st.button("It's Fake"):
    st.text("You guessed:")
    st.subheader("*Fake*")
    st.text("The Deepfake Detector model guessed...")
    time.sleep(1)
    st.subheader(f"*{prediction}*")
    st.text("The truth is...")
    time.sleep(1)
    st.subheader(f"***It's {true_label}!***")


def detector_mode():

  st.header("Detector Mode")
  st.subheader("Upload an Image to Make a Prediction")

  # upload an image
  uploaded_image = st.file_uploader("Upload your own image to test the model:", type=['jpg', 'jpeg'])

  # when an image is uploaded, display image and run inference
  if uploaded_image is not None:
    st.image(uploaded_image)
    st.text(get_prediction(classifier, uploaded_image))


page = st.sidebar.selectbox('Select Mode',['Detector Mode','Game Mode']) 

if page == 'Game Mode':
  game_mode()
else:
  detector_mode()

# -------------------
# SCRIPT/MODULE CHECKER
# -------------------
if __name__ == "__main__":
    main()










