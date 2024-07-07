import torchaudio
from speechbrain.inference.interfaces import foreign_class

# 감정 예측 함수
def classify_emotion(audio_file, model_source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP"):
    # 모델 로드
    classifier = foreign_class(source=model_source, pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")

    # 감정 예측
    out_prob, score, index, text_lab = classifier.classify_file(audio_file)
    return text_lab

# 예제 사용법
# 로컬에 있는 영어 오디오 파일 경로 설정
audio_file = "03-01-05-02-01-01-01.wav"

# 영어 모델 사용
english_model = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
english_predictions = classify_emotion(audio_file, english_model)
print("English Predictions:", english_predictions)
