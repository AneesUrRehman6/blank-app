# import streamlit as st
# import cv2
# import mediapipe as mp
# import numpy as np
# import tensorflow as tf
# import time
# from collections import deque

# # Load model
# model = tf.keras.models.load_model("model.keras", compile=False)
# labels = [chr(i) for i in range(65, 91)]

# #classes 
# classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
#            'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
#            'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# # MediaPipe hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)


# st.title("Sign Language Translator")
# st.write(
#     "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
# )

# st.write("Hello World ")


import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Streamlit app
st.title("Real-Time Hand Detection ✋")
st.markdown("Using MediaPipe and Streamlit")

# Create layout
col1, col2 = st.columns(2)
with col1:
    run = st.checkbox('Start Webcam', value=True)
with col2:
    hand_count = st.slider("Max hands to detect", 1, 4, 2)

# Update hand detector settings
hands.max_num_hands = hand_count

FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

# Custom drawing styles
hand_connections_drawing_spec = mp_drawing.DrawingSpec(
    thickness=2, 
    color=(0,255,0),
    circle_radius=2
)

while run:
    ret, frame = camera.read()
    if not ret:
        st.error("Failed to access camera")
        break
    
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                hand_connections_drawing_spec
            )
    
    # Convert back to RGB for Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
    
else:
    st.warning("Webcam stopped")
    camera.release()
    hands.close()

st.caption("Made with Streamlit and MediaPipe")
