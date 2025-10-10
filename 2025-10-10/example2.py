from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

prompt = ChatPromptTemplate.from_template("You are an expert in astronomy.Answer the question. <Question>: {input}")
llm = ChatOpenAI(model="gpt-4o-mini")
# chain 연결
chain = prompt | llm
# chain 호출
response =chain.invoke({"input": "지구의 자전 주기는?"})

print(response.content)
'''
답변
지구의 자전 주기는 약 24시간입니다. 
이를 좀 더 정확히 말하면, 평균적으로 23시간 56분 4초(즉, 약 86164초)입니다. 
이 시간을 "항성일"이라고 하 며, 우리가 흔히 사용하는 24시간은 "태양일"로, 
이는 지구가 태양을 기준으로 하루를 돌며 태양이 같은 위치에서 다시 관찰되는 시간을 기준으로 합 니다. 
그렇기 때문에 태양일은 약 4분 정도 길어집니다.
'''