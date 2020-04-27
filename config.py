import time
import numpy as np
import streamlit as st
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')
FACE_THRESHOLD = config.getfloat('main', 'face_threshold')
METHOD = config.get('main', 'method')
CUDA = config.getboolean('main', 'cuda')
"### This service is made by BEKKOUCH Imad for the Information Retrieval course in innopolis university"
"# Current Hyper Parameters"
st.write("Face Detection Threshold:")
FACE_THRESHOLD
"Search Method:"
st.markdown(
    f"""<span style="color:#09ab3b;background:#fafafa;padding: .2em .4em;    border-radius: .25rem;">{METHOD.replace('_', ' ')}</span>""",
    unsafe_allow_html=True)
"Use Cuda:"
st.write(CUDA)

# progress_bar = st.sidebar.progress(0)
# status_text = st.sidebar.empty()
# last_rows = np.random.randn(1, 1)
# chart = st.line_chart(last_rows)


# progress_bar.empty()

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
# btn = st.button("Re-run")
"# Configure the server"
"### Face Threshold:"
face_threshold = st.slider('', 0., 1., FACE_THRESHOLD)

with open('config.ini', 'w') as f:
    FACE_THRESHOLD = float(str(face_threshold))
    config.set('main', 'face_threshold', str(face_threshold))
    config.write(f)
"### How would you like to search for faces?"
method = st.selectbox('', ('Exact Search', 'KdTree Search'))
# face_threshold = st.sidebar.slider('Face Threshold', 0., 1., 0.1)

with open('config.ini', 'w') as f:
    METHOD = str(method)
    config.set('main', 'method', str(method).replace(' ', '_'))
    config.write(f)


"### Do you want to use cuda?"
cuda = st.selectbox('', ('True', 'False'))
# face_threshold = st.sidebar.slider('Face Threshold', 0., 1., 0.1)

with open('config.ini', 'w') as f:
    CUDA = bool(cuda)
    config.set('main', 'cuda', str(CUDA))
    config.write(f)
