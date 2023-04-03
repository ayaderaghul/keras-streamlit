
import streamlit as st

from tensorflow.keras.models import load_model

import numpy as np

import matplitlib.pyplot as plt

st.header("Photo to Monet")

st.caption('Upload an image 256x256')

model = load_model('g_model_AtoB_002160.h5')

@st.cache
def load_image(image_file):
    img=plt.imread(image_file)
    return img

imgpath = st.file_uploader("Choose a file", type =['png', 'jpeg', 'jpg'])

if imgpath is not None:

    img = load_image(imgpath )

    st.image(img, width=250)


def convert(image):
    img=load_image(img,target_size=(256,256))
    img_array = np.reshape(img, (1, 256, 256, 3))
    result=model.predict(img_array)
    result=np.squeeze(img,axis=0)
    return result

if st.button('Convert'):
    result=convert(imagepath)
    st.image(result)
