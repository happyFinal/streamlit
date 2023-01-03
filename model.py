import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("wumusill/final_20man")

def load_model():
    model = AutoModelForCausalLM.from_pretrained("wumusill/final_20man")
    return model

model = load_model()

def mind(input_letter):
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


result = mind('임종우')
for r in result:
    print(f'{r} : {result[r]}')