# 9) 스트리밍 출력 받기 (토큰 나오자마자 표시)
import ollama
from sys import stdout
from typing import List
import numpy as np
import faiss
import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv

load_dotenv()

slack_token = os.getenv("SLACK_BOT_TOKEN")

client = WebClient(token=slack_token)

channel_name = "C09KBJMKL4F" 

if not slack_token:
    print("오류: .env 파일에 SLACK_BOT_TOKEN이 설정되지 않았습니다.")
    exit(1)
stream = ollama.chat(
    model='gemma3:4b',
    messages=[{"role": "user", "content": "생성형 AI의 장단점을 항목별로 정리해줘."}],
    stream=True,  # 스트리밍
    options={"temperature": 0.3}
)

buf = []
for chunk in stream:
    token = chunk['message']['content']
    buf.append(token)
    stdout.write(token)
stdout.flush()

print("\n\n전체 응답:", "".join(buf))

try:
# chat_postMessage API 호출
    response = client.chat_postMessage(
        channel=channel_name,
        text="\n\n전체 응답: " + "".join(buf)
    )
    print("메시지가 성공적으로 전송되었습니다.")
except SlackApiError as e:
# API 호출 실패 시 에러 코드를 확인합니다.
# 에러 원인: 토큰이 유효하지 않거나, 봇이 채널에 없거나, 권한이 부족한 경우 등
    print(f"메시지 전송에 실패했습니다: {e.response['error']}")

