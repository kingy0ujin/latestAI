# 4) 구조화 출력(JSON)로 정보추출
import ollama
import json
import ollama
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

text = """
주문 3건:
1) 상품: 무선마우스, 수량 2, 가격 25,000원
2) 상품: 기계식 키보드, 수량 1, 가격 89,000원
3) 상품: USB-C 케이블, 수량 3, 가격 9,900원
총 배송지는 서울시 강남구 테헤란로 1
"""

prompt = f"""
아래 텍스트에서 주문 항목을 JSON으로 추출해.
스키마:
{{
  "orders":[{{"item":str,"qty":int,"price_krw":int}}],
  "shipping_address": str,
  "total_price_krw": int
}}
텍스트:
{text}
반드시 JSON만 출력.
"""

resp = ollama.chat(
    model='gemma3:4b',
    messages=[{"role": "user", "content": prompt}],
    format='json',  # JSON 모드
    options={"temperature": 0}
)

data = json.loads(resp['message']['content'])
#print(json.dumps(data, indent=2, ensure_ascii=False))


# ============================================================================
# 미션: 영화 리뷰 분석 - 복잡한 JSON 구조 추출 연습
# ============================================================================
"""
[미션]
영화 리뷰 텍스트에서 다음 정보를 JSON으로 추출하는 코드를 작성하세요:
- 영화 제목 (title), 감독 (director), 장르 (genre)
- 평점 (rating) - 5점 만점의 숫자
- 장점 리스트 (pros) - 문자열 배열
- 단점 리스트 (cons) - 문자열 배열
- 추천 여부 (recommended) - true/false

아래 예시를 참고하여 다른 영화 리뷰로도 테스트해보세요!
"""

print("\n" + "="*80)
print("🎬 [미션] 영화 리뷰 분석")
print("="*80)

# 예시 영화 리뷰 텍스트
review_text = """
크리스토퍼 놀란 감독의 '인터스텔라'는 SF 장르의 걸작이다. 
시각 효과가 정말 환상적이고, 한스 짐머의 음악이 영화와 완벽하게 어우러진다.
스토리도 감동적이며 과학적 고증도 뛰어나다.
다만 러닝타임이 169분으로 너무 길고, 일부 과학 설명이 일반 관객에게는 어렵게 느껴질 수 있다.
그럼에도 불구하고 꼭 봐야 할 영화다. 5점 만점에 4.5점을 주고 싶다.
"""

# JSON 추출 프롬프트
review_prompt = f"""
아래 영화 리뷰 텍스트에서 정보를 추출하여 JSON으로 출력해줘.

스키마:
{{
  "title": str,           // 영화 제목
  "director": str,        // 감독
  "genre": str,           // 장르
  "rating": float,        // 평점 (0.0 ~ 5.0)
  "pros": [str],          // 장점 리스트
  "cons": [str],          // 단점 리스트
  "recommended": bool     // 추천 여부
}}

리뷰 텍스트:
{review_text}

반드시 JSON만 출력하고, 텍스트에서 유추 가능한 모든 정보를 포함해줘.
"""

# Ollama 호출
review_resp = ollama.chat(
    model='gemma3:4b',
    messages=[{"role": "user", "content": review_prompt}],
    format='json',
    options={"temperature": 0}
)

# 결과 파싱 및 출력
review_data = json.loads(review_resp['message']['content'])
print("\n📊 추출된 영화 정보:")
print(json.dumps(review_data, indent=2, ensure_ascii=False))

# 결과를 보기 좋게 출력
print("\n" + "-"*80)
print(f"🎬 영화: {review_data.get('title', 'N/A')}")
print(f"🎥 감독: {review_data.get('director', 'N/A')}")
print(f"🎭 장르: {review_data.get('genre', 'N/A')}")
print(f"⭐ 평점: {review_data.get('rating', 'N/A')}/5.0")
print(f"👍 추천: {'예' if review_data.get('recommended', False) else '아니오'}")

print(f"\n✅ 장점:")
for i, pro in enumerate(review_data.get('pros', []), 1):
    print(f"  {i}. {pro}")

print(f"\n❌ 단점:")
for i, con in enumerate(review_data.get('cons', []), 1):
    print(f"  {i}. {con}")

print("="*80)


# ============================================================================
# 추가 연습: 다른 영화 리뷰로 테스트해보세요!
# ============================================================================
"""
💡 연습 과제:
1. 위 코드를 참고하여 좋아하는 영화의 가상 리뷰를 작성하고 JSON 추출을 테스트해보세요.
2. 여러 개의 리뷰를 처리하는 반복문을 만들어보세요.
3. 추출된 JSON 데이터를 파일로 저장해보세요 (json.dump() 사용).
4. 평점이 4.0 이상이고 추천하는 영화만 필터링해보세요. -> 패스

예시 리뷰 텍스트 (테스트용):
- "봉준호 감독의 '기생충'은 블랙 코미디 스릴러로, 계급 갈등을 예리하게 그려냈다..."
- "제임스 카메론의 '아바타'는 판타지 액션 영화로, 3D 기술이 혁신적이다..."
"""

# 예시 영화 리뷰 텍스트
review_text = """
- 라라랜드는 정말 최고의 로맨스 영화입니다. 
사랑과 꿈 사이에서 고민하는 연인의 모습을 감미로운 음악과 아름다운 배경을 통해
표현해 준 영화입니다. 감독은 데이미어너 셔젤이며 장르는 뮤지컬 영화입니다.
평점은 4.5입니다.
단점은 영화가 너무 짧게 느껴져서 아쉽다는것과 뮤지컬 영화를 좋아하지 않는 사람에게는 호불호가 갈릴 수 있다는 점입니다.
그럼에도 저는 이 영화를 추천합니다.
- "봉준호 감독의 '기생충'은 블랙 코미디 스릴러로, 계급 갈등을 예리하게 그려냈다..."
- "제임스 카메론의 '아바타'는 판타지 액션 영화로, 3D 기술이 혁신적이다..."
"""

# JSON 추출 프롬프트
review_prompt = f"""
아래 영화 리뷰 텍스트에서 정보를 추출하여 JSON으로 출력해줘.

스키마:
{{
  "title": str,           // 영화 제목
  "director": str,        // 감독
  "genre": str,           // 장르
  "rating": float,        // 평점 (0.0 ~ 5.0)
  "pros": [str],          // 장점 리스트
  "cons": [str],          // 단점 리스트
  "recommended": bool     // 추천 여부
}}

리뷰 텍스트:
{review_text}

반드시 JSON만 출력하고, 텍스트에서 유추 가능한 모든 정보를 포함해줘.
"""

# Ollama 호출
review_resp = ollama.chat(
    model='gemma3:4b',
    messages=[{"role": "user", "content": review_prompt}],
    format='json',
    options={"temperature": 0}
)
review_data = json.loads(review_resp['message']['content'])
try:
# chat_postMessage API 호출
    response = client.chat_postMessage(
        channel=channel_name,
        text=json.dumps(review_data, indent=2, ensure_ascii=False)
    )
    print("메시지가 성공적으로 전송되었습니다.")
except SlackApiError as e:
# API 호출 실패 시 에러 코드를 확인합니다.
# 에러 원인: 토큰이 유효하지 않거나, 봇이 채널에 없거나, 권한이 부족한 경우 등
    print(f"메시지 전송에 실패했습니다: {e.response['error']}")
