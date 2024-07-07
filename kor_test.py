import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torch
import os

# 감정 예측 함수
def classify_emotion(audio_file, model_name="jungjongho/wav2vec2-xlsr-korean-speech-emotion-recognition2_data_rebalance"):
    # 모델 및 프로세서 로드
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

    # 오디오 파일 로드 및 샘플링 속도 변환
    speech_array, sampling_rate = torchaudio.load(audio_file)
    if sampling_rate != 16000:
        transform = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        speech_array = transform(speech_array)
        sampling_rate = 16000

    # 오디오 데이터를 1차원 배열로 변환
    speech_array = speech_array.squeeze().numpy()

    # 입력 데이터 전처리
    inputs = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

    # 모델 추론
    with torch.no_grad():
        logits = model(**inputs).logits

    # 감정 예측
    predicted_ids = torch.argmax(logits, dim=-1).item()
    emotions = ["angry", "fear", "happy", "neutral", "sad", "surprise"]
    predicted_emotion = emotions[predicted_ids]

    return predicted_emotion

# 예제 사용법
# 로컬에 있는 한국어 오디오 파일 경로 설정
audio_file = "5ebd3dd7c38c123b9ec6deba.wav"

# 한국어 모델 사용
korean_model = "jungjongho/wav2vec2-xlsr-korean-speech-emotion-recognition2_data_rebalance"
korean_predictions = classify_emotion(audio_file, korean_model)
print("Korean Predictions:", korean_predictions)
