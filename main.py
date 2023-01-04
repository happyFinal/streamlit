import requests
import streamlit as st
from streamlit_lottie import st_lottie
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Page Config
st.set_page_config(
    page_title="노래 가사 n행시",
    page_icon="💌",
    layout="wide"
)

### Model
tokenizer = AutoTokenizer.from_pretrained("wumusill/final_project_kogpt2")

@st.cache(show_spinner=False)
def load_model():
    model = AutoModelForCausalLM.from_pretrained("wumusill/final_project_kogpt2")
    return model

model = load_model()

# Class : Dict 중복 키 출력
class poem(object):
    def __init__(self,letter):
        self.letter = letter

    def __str__(self):
        return self.letter

    def __repr__(self):
        return "'"+self.letter+"'"


def n_line_poem(input_letter):

    # 두음 법칙 사전
    dooeum = {"라":"나", "락":"낙", "란":"난", "랄":"날", "람":"남", "랍":"납", "랑":"낭", 
          "래":"내", "랭":"냉", "냑":"약", "략":"약", "냥":"양", "량":"양", "녀":"여", 
          "려":"여", "녁":"역", "력":"역", "년":"연", "련":"연", "녈":"열", "렬":"열", 
          "념":"염", "렴":"염", "렵":"엽", "녕":"영", "령":"영", "녜":"예", "례":"예", 
          "로":"노", "록":"녹", "론":"논", "롱":"농", "뢰":"뇌", "뇨":"요", "료":"요", 
          "룡":"용", "루":"누", "뉴":"유", "류":"유", "뉵":"육", "륙":"육", "륜":"윤", 
          "률":"율", "륭":"융", "륵":"늑", "름":"늠", "릉":"능", "니":"이", "리":"이", 
          "린":'인', '림':'임', '립':'입'}
    # 결과물을 담을 list
    res_l = []

    # 한 글자씩 인덱스와 함께 가져옴
    for idx, val in enumerate(input_letter):
        # 두음 법칙 적용
        if val in dooeum.keys():
            val = dooeum[val]
        
        times = 0
        while times < 3:
            # 만약 idx 가 0 이라면 == 첫 글자
            if idx == 0:
                # 첫 글자 인코딩
                input_ids = tokenizer.encode(
                val, add_special_tokens=False, return_tensors="pt")

                # 첫 글자 인코딩 값으로 문장 생성
                output_sequence = model.generate(
                    input_ids=input_ids, 
                    do_sample=True, max_length=42, no_repeat_ngram_size=2,
                    min_length=5, temperature=0.9, repetition_penalty=1.5)

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
                    input_ids=input_ids, 
                    do_sample=True, max_length=42, no_repeat_ngram_size=2,
                    min_length=len_sequence, temperature=0.9, repetition_penalty=1.5)

            # 생성된 문장 리스트로 변환 (인코딩 되어있고, 생성된 문장 뒤로 padding 이 있는 상태)
            generated_sequence = output_sequence.tolist()[0]

            # padding index 앞까지 slicing 함으로써 padding 제거, padding이 없을 수도 있기 때문에 조건문 확인 후 제거
            if tokenizer.pad_token_id in generated_sequence:
                generated_sequence = generated_sequence[:generated_sequence.index(tokenizer.pad_token_id)]

            # 첫 글자가 아니라면, generate 된 음절만 결과물 list에 들어갈 수 있게 앞 문장에 대한 인코딩 값 제거
            # print(generated_sequence)
            if idx != 0:
                # 이전 문장의 길이 이후로 슬라이싱해서 앞 문장 제거
                generated_sequence = generated_sequence[len_sequence:]

                # 다음 음절을 위해 길이 갱신
                len_sequence += len(generated_sequence)        

            # 첫 글자라면
            else:
                # 시퀀스 길이 저장
                len_sequence = len(generated_sequence)

            # print(last_sequence)

            # 결과물 디코딩
            decoded_sequence = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
            
            if len(decoded_sequence) > 1:
                break
            else:
                times += 1
                continue
                
        # 결과물 리스트에 담기
        res_l.append(decoded_sequence)

    poem_dict = {}

    for letter, res in zip(input_letter, res_l):
        poem_dict[poem(letter)] = res

    return poem_dict

###

# Image(.gif)
@st.cache(show_spinner=False)
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
    st.title("한글 노래 가사 n행시✍")
    st.subheader("🦁멋쟁이사자처럼 AIS7 파이널 프로젝트")

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
    st.markdown("### 가이드라인")
    st.markdown("1. 하단에 있는 텍스트바에 단어를 넣어주세요")
    st.markdown("2. 'n행시 제작하기' 버튼을 클릭해주세요")

st.write('---')

# Model & Input
row2_spacer1, row2_1, row2_spacer2, row2_2, row2_spacer3, row2_3, row2_spacer4 = st.columns((0.01, 1.5, 0.05, 1.5, 0.05, 1.5, 0.05))

# Word Input
if "generate" not in st.session_state:
    st.session_state.generate = False

with row2_2:
    word_input = st.text_input(
            "n행시에 사용할 단어를 적고 Enter를 눌러주세요. 👇",
            placeholder='한글 단어',
            max_chars=10
    )
    
    if word_input:
        st.write("n행시 단어 :  ", word_input)

    if st.button('n행시 제작하기'):
        with st.spinner('잠시 기다려주세요...'):
            result = n_line_poem(word_input)
        st.success('완료됐습니다!')
        for r in result:
            st.write(f'{r} : {result[r]}')



