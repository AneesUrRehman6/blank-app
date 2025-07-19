import streamlit as st
# import cv2

import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
from collections import deque

# Load model
model = tf.keras.models.load_model("model.keras", compile=False)
labels = [chr(i) for i in range(65, 91)]

#classes 
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
           'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
           'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)


st.title("Sign Language Translator")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

st.write("Hello World ")

