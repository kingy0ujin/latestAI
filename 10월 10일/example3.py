from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# prompt + model + output parser
prompt = ChatPromptTemplate.from_template("You are an expert in astronomy.Answer the question. <Question>: {input}")
llm = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()
# LCEL chaining
chain = prompt | llm | output_parser
# chain 호출
response=chain.invoke({"input": "지구의 자전 주기는?"})

print(response)

'''
답변
지구의 자전 주기는 약 24시간입니다. 정확히 말하면, 
지구가 자기 축을 한 바퀴 도는 데 걸리는 시간은 약 23시간 56분 4초(즉, 86164초)로, 
이를 ' 항성일'이라고 합니다. 그러나 우리가 일상에서 사용하는 24시간은 
태양이 하늘에서 한 바퀴 도는 것을 기준으로 하여 측정된 '태양일'입니다. 
태양일은 약간 더 길어지는 이유는 지구가 태양 주위를 공전하면서 자전하기 때문입니다.
'''