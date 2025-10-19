# 11) 간단 "정답 체크" 루브릭(자체평가) — 채점 프롬프트 -> 패스
import ollama
import json
import pandas as pd
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


def ask(model, task, system="한국어로 간결하고 정확하게 답해줘.", **options):
    return ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": task}
        ],
        options=options or {}
    )['message']['content']


student_answer = ask('gemma3:4b', "RAG의 정의를 한 문장으로 설명해줘.", temperature=0)

rubric = """
채점 기준:
1) 정의의 정확성(0~4)
2) 간결성(0~3)
3) 핵심 용어 사용(0~3)
총점=10, JSON으로만 출력: {"score": <0-10>, "feedback": "<한줄 피드백>"}
학생 답변:
""" + student_answer

grade = ollama.chat(
    model='gemma3:4b',
    messages=[{"role": "user", "content": rubric}],
    format='json',
    options={"temperature": 0}
)

print("학생 답변:", student_answer)
print("\n채점 결과:")
grade_content = json.loads(grade['message']['content'])
print(json.dumps(json.loads(grade['message']['content']), indent=2, ensure_ascii=False))

try:
# chat_postMessage API 호출
    response = client.chat_postMessage(
        channel=channel_name,
        text=json.dumps(json.loads(grade['message']['content']), indent=2, ensure_ascii=False)
    )
    print("메시지가 성공적으로 전송되었습니다.")
except SlackApiError as e:
# API 호출 실패 시 에러 코드를 확인합니다.
# 에러 원인: 토큰이 유효하지 않거나, 봇이 채널에 없거나, 권한이 부족한 경우 등
    print(f"메시지 전송에 실패했습니다: {e.response['error']}")

