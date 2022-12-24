import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import requests
from streamlit_lottie import st_lottie

st.set_page_config(
    page_title="노래 가사 n행시",
    page_icon="💌",
)

@st.cache
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url = "https://assets7.lottiefiles.com/private_files/lf30_fjln45y5.json"

lottie_json = load_lottieurl(lottie_url)
st_lottie(lottie_json, speed=1, height=200, key="initial")


# Grid Setup
row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
    (0.05, 2, 0.05, 1, 0.05)
)

# Title
row0_1.title("한글 노래 가사 n행시")

with row0_2:
    row0_2.subheader(
    "Likelion AIS7 Final Project"
    )
    st.write('''해파리팀 : 이지혜, 최지영, 권소희, 문종현, 구자현, 김의준''')

st.write('---')