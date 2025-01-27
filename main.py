import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 실행 시간 측정 시작
start_time = time.time()

# 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# 직무를 설정한 입력 메시지
system_message = "You are an engineer"
user_message = "How do military drones work?"
input_text = f"{system_message}\nUser: {user_message}\nAI:"
inputs = tokenizer(input_text, return_tensors="pt")

# 텍스트 생성
outputs = model.generate(**inputs, max_length=200)
english_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 생성된 텍스트 출력 (영어)
print("AI 응답 (영어):")
print(english_response)

# 번역 모델 로드 (대체 모델 사용)
translator = pipeline("translation", model="facebook/nllb-200-distilled-600M")

# 번역 수행 (언어 코드 지정)
translated_response = translator(english_response, src_lang="eng_Latn", tgt_lang="kor_Hang")[0]["translation_text"]

# 번역된 텍스트 출력 (한글)
print("\nAI 응답 (한글 번역):")
print(translated_response)

# 종료 시간 기록
end_time = time.time()

# 실행 시간 출력
execution_time = end_time - start_time
print(f"\n전체 실행 시간: {execution_time:.6f}초")
