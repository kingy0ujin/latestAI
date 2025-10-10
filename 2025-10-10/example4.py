from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
prompt1 = ChatPromptTemplate.from_template("translates {korean_word} to English.")
prompt2 = ChatPromptTemplate.from_template(
"explain {english_word} using oxford dictionary to me in Korean."
)
llm = ChatOpenAI(model="gpt-4o-mini")
chain1 = prompt1 | llm | StrOutputParser()
chain2 = (
{"english_word": chain1}
| prompt2
| llm
| StrOutputParser()
)

response = chain2.invoke({"korean_word":"미래"})

print(response)

'''
답변
The Korean word "미래" translates to "future" in English.
'''

'''
답변
"미래"라는 한국어 단어는 영어로 "future"로 번역됩니다. 옥스포드 사전을 참고하면 "미래(future)"라는 단어는 다음과 같은 의미를 가지고 있습니다:

1. **지금부터 앞으로의 시간**: 현재 시점 이후에 일어날 수 있는 모든 사건이나 상태를 나타냅니다.
2. **앞으로의 가능성**: 사람이 경험할 수 있는 기회나 변화에 대한 예측을 포함합니다.

이처럼 "미래"는 우리가 아직 경험하지 않은 시간적인 개념과 관련이 있으며, 종종 희망이나 목표와 연결되어 사용됩니다.
'''