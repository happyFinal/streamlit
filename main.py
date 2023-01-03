import requests
import streamlit as st
from streamlit_lottie import st_lottie

st.set_page_config(
    page_title="노래 가사 n행시",
    page_icon="💌",
)

### Model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("wumusill/final_20man")

@st.cache
def load_model():
    model = AutoModelForCausalLM.from_pretrained("wumusill/final_20man")
    return model

model = load_model()

def mind(input_letter):
    # 결과물을 담을 list
    res_l = []

    # 한 글자씩 인덱스와 함께 가져옴
    for idx, val in enumerate(input_letter):
 
        # 만약 idx 가 0 이라면 == 첫 글자
        if idx == 0:
            # 첫 글자 인코딩
            input_ids = tokenizer.encode(
            val, add_special_tokens=False, return_tensors="pt")
            
            # 첫 글자 인코딩 값으로 문장 생성
            output_sequence = model.generate(
                input_ids,
                do_sample=True, max_length=42)
        
        # 첫 글자가 아니라면
        else:
            # 좀더 매끄러운 삼행시를 위해 이전 문장이랑 현재 음절 연결
            # 이후 generate 된 문장에서 이전 문장에 대한 데이터 제거
            link_with_pre_sentence = " ".join(res_l) + " " + val  
            # print(link_with_pre_sentence)

            # 연결된 문장을 인코딩
            input_ids = tokenizer.encode(
            link_with_pre_sentence, add_special_tokens=False, return_tensors="pt")

            # 인코딩 값으로 문장 생성
            output_sequence = model.generate(
                input_ids,
                do_sample=True, max_length=42)

        # 생성된 문장 리스트로 변환 (인코딩 되어있고, 생성된 문장 뒤로 padding 이 있는 상태)
        generated_sequence = output_sequence.tolist()[0]

        # padding index 앞까지 slicing 함으로써 padding 제거
        generated_sequence = generated_sequence[:generated_sequence.index(tokenizer.pad_token_id)]
        
        # 첫 글자가 아니라면, generate 된 음절만 결과물 list에 들어갈 수 있게 앞 문장에 대한 인코딩 값 제거
        # print(generated_sequence)
        if idx != 0:
            # 이전 문장의 마지막 시퀀스 이후로 슬라이싱해서 앞 문장 제거
            generated_sequence = generated_sequence[generated_sequence.index(last_sequence) + 1:]

            # 다음 음절을 위해 마지막 시퀀스 갱신
            last_sequence = generated_sequence[-1]        
        
        # 첫 글자라면
        else:
            # 마지막 시퀀스 저장
            last_sequence = generated_sequence[-1]        
        
        # print(last_sequence)

        # 결과물 디코딩
        decoded_sequence = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        # 결과물 리스트에 담기
        res_l.append(decoded_sequence)

        # print(res_l)

    dictionary = {}

    for letter, res in zip(input_letter, res_l):
        dictionary[letter] = res

    return dictionary

###


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
        result = mind(word_input)
        for r in result:
            st.write(r)



