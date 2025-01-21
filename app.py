# 보통 입구를 app.py라고 함

import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI


## 1. 파일을 dataframe으로 부른다.

df = pd.read_excel("AdidasSalesdata.xlsx")


## 2. dataframe을 화면에 출력한다.
# "dataframe:", my_data_frame
st.title("Adidas 데이터 프레임 필터링")
st.dataframe(df)

# 필수 파라미터-----
# 모델명
# system: - 너는 상담봇이야
# assistant: -로는 목록을 표시하지마
# user: 사용자 메세지의 예시
# content: 뭘 시키고 싶은지

# 선택 파라미터 -----



# !pip install openai
# https://platform.openai.com
# https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models
# https://platform.openai.com/docs/models
# https://velog.io/@yule/OpenAI-API-%EB%B0%9C%EA%B8%89


import streamlit as st
import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv # python-dotenv


load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=api_key)

# 데이터프레임을 캐싱하는 함수
@st.cache_data
def load_data():
    return pd.read_excel('AdidasSalesdata.xlsx')

# 캐싱된 데이터프레임을 불러옴
df = load_data()

# GPT에게 우리가 다루는 데이터프레임이 어떤 구조인지 알려주는 함수 작성
# 아래 컬럼 기준으로 코드를 작성해줄 것 요청
def table_definition_prompt(df):
    prompt = '''Given the following pandas dataframe definition,
            write queries based on the request
            \n### pandas dataframe, with its properties:
            
            #
            # df의 컬럼명({})
            #
            '''.format(",".join(str(x) for x in df.columns))
    
    return prompt

nlp_text = st.text_input('질문을 입력하세요: ')
accept = st.button('요청') # 아래 코드가 실행되지 않도록 

if accept:
    full_prompt = str(table_definition_prompt(df)) + str(nlp_text)


    # API 호출
    #  R T F C 프레임워크
    #  Role : AI가 수행할 역할을 명확히 정의
    #  Task : 수행할 구체적인 작업 기술
    #  Format : 결과물의 형식과 길이를 지정 
    #  Constraints : 준수해야할 규칙과 제한을 명시
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an assistant that generates Pandas boolean indexing code based on the given df definition\
            and a natural language request. The answer should start with df and contains only code by one line, not any explanation or ``` for copy."},
            {"role": "user", "content": f"A query to answer: {full_prompt}"}
        ],
        max_tokens=200, # 비용 발생하므로 시도하며 적당한 값 찾아간다. 200이면 최대 200단어까지 생성. 
                        # 영어는 한 단어가 1토큰, 한글은 한 글자가 1토큰 정도
        temperature=1.0, # 창의성 발휘 여부. 0~2 사이. 0에 가까우면 strict하게, 2에 가까우면 자유롭게(창의성 필요)
        stop=None # 특정 문자열이 들어오면 멈춘다든지. None이면 없음. .이면 문장이 끝나면 멈춘다든지
        )


    answer = response.choices[0].message.content

    st.code(full_prompt) # 우리 눈으로 확인하기 위해서 

    st.code(answer) # 결과 코드가 어떻게 왔는지

    # eval 함수를 사용하여 문자열로 된  ' '를 벗겨내고 코드 자체로 실행
    st.write(eval(answer))
    

