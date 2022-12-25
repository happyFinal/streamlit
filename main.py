import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import requests
import re
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


# Title
row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
    (0.01, 2, 0.05, 1, 0.01)
)

with row0_1:
    st.title("한글 노래 가사 n행시")
    st.subheader("멋쟁이사자처럼 AIS7 파이널 프로젝트")

with row0_2:
    st.write("")
    st.subheader(
        "해파리팀"
    )
    st.write("이지혜, 최지영, 권소희")
    st.write("문종현, 구자현, 김의준")

st.write('---')

# Explanation
row1_spacer1, row1_1, row1_spacer2 = st.columns((0.01, 3, 0.01))

with row1_1:
    st.markdown(
        "**'MZ세대'에게**"
    )
    st.markdown(
        "음악은 세대를 드러내는 지표이자 자신의 감정 및 공동체를 드러내는 수단이다."
    )

st.write('---')

# Model & Input
row2_spacer1, row2_1, row2_spacer2, row2_2, row2_spacer3 = st.columns((0.01, 1.5, 0.05, 1.5, 0.01))

# def load_model():
#     return tf.keras.models.load_model('')

# model = load_model()

# Genre Selector
if "genre" not in st.session_state:
    st.session_state.genre = "전체"

with row2_1:
    st.radio(
        "장르를 선택해주세요. 👉",
        key="genre",
        options=["전체", "발라드", "록/메탈", "힙합", "트로트"],
    )

# Word Input
if "generate" not in st.session_state:
    st.session_state.generate = False

with row2_2:
    word_input = st.text_input(
            "n행시에 사용할 단어를 적고 Enter를 눌러주세요. 👇",
            placeholder='한글 단어'
    )
    
    if word_input:
        st.write("n행시 단어 :  ", word_input)

    if st.button('n행시 제작하기'):
        st.write("제작중...")



