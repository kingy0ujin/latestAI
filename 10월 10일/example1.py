# 1. 라이브러리 임포트
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# 2. API 키 로드
load_dotenv()

# 3. 모델 실행 (기존 코드)
llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke("지구의 자전 주기는?")

print(response.content)

'''
답변
지구의 자전 주기는 약 24시간입니다. 
정확하게는 23시간 56분 4초 정도로, 이를 '항성일'이라고 합니다. 
하지만 일반적으로 우리는 하루를 24시간으 로 나누어 사용하고 있습니다. 
이 때문에 태양이 같은 위치에 다시 나타나는 시간을 기준으로 한 '태양일'은 약 24시간입니다.
'''