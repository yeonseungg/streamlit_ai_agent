import streamlit as st
import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv  # python-dotenv

# 환경 변수 로드
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=api_key)

# 데이터프레임 업로드 및 캐싱 함수
@st.cache_data
def load_data(uploaded_file, file_extension):
    if file_extension == "xlsx":
        return pd.read_excel(uploaded_file)
    elif file_extension == "csv":
        return pd.read_csv(uploaded_file)
    else:
        raise ValueError("지원하지 않는 파일 형식입니다.")

# 데이터프레임 구조를 설명하는 프롬프트 생성 함수
def table_definition_prompt(df):
    prompt = '''Given the following pandas dataframe definition,
            write queries based on the request
            \n### pandas dataframe, with its properties:
            
            #
            # df의 컬럼명({})
            #
            '''.format(",".join(str(x) for x in df.columns))
    return prompt

st.title("AI 기반 데이터프레임 필터링 도구")

# 파일 업로드 섹션
uploaded_file = st.file_uploader("엑셀 또는 CSV 파일을 업로드하세요", type=["xlsx", "csv"])

if uploaded_file:
    # 파일 확장자 확인
    file_extension = uploaded_file.name.split(".")[-1].lower()

    if file_extension not in ["xlsx", "csv"]:
        st.error("확장자를 확인해주세요. 엑셀(.xlsx) 또는 CSV 파일만 업로드 가능합니다.")
    else:
        try:
            # 업로드된 파일을 데이터프레임으로 변환
            df = load_data(uploaded_file, file_extension)
            st.write("### 데이터 미리보기")
            st.dataframe(df)

            # 사용자 질문 입력
            nlp_text = st.text_input("질문을 입력하세요 (예: 매출이 100 이상인 데이터 보여줘):")
            accept = st.button("요청 실행")

            if accept and nlp_text:
                # GPT에 전달할 프롬프트 생성
                full_prompt = str(table_definition_prompt(df)) + str(nlp_text)

                # OpenAI API 호출
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an assistant that generates Pandas boolean indexing code based on the given df definition\
                            and a natural language request. The answer should start with df and contains only code by one line, not any explanation or ``` for copy."
                        },
                        {
                            "role": "user",
                            "content": f"A query to answer: {full_prompt}"
                        },
                    ],
                    max_tokens=200,
                    temperature=0.7,
                )

                # 결과 코드 실행 및 출력
                answer = response.choices[0].message.content.strip()
                st.write("### 생성된 코드")
                st.code(answer)

                # 실행 및 결과 표시
                try:
                    result_df = eval(answer)
                    st.write("### 결과 데이터")
                    st.dataframe(result_df)
                except Exception as e:
                    st.error(f"코드 실행 중 오류가 발생했습니다: {e}")
        except Exception as e:
            st.error(f"파일 처리 중 오류가 발생했습니다: {e}")
else:
    st.info("엑셀(.xlsx) 또는 CSV 파일을 업로드하면 데이터를 필터링할 수 있습니다.")
