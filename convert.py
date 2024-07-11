import torch
from speechbrain.inference import EncoderClassifier
from transformers import Wav2Vec2Processor
import onnx
from openvino.runtime import Core
import numpy as np

# 모델 로드
classifier = EncoderClassifier.from_hparams(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP")

# 모델 접근 수정
model = classifier.mods['wav2vec2']  # 실제 키로 수정
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

# 모델을 평가 모드로 전환
model.eval()

# 더미 입력 생성 (ONNX 변환용)
# 랜덤한 더미 오디오 데이터를 생성하여 processor에 전달
dummy_audio = np.random.randn(16000).astype(np.float32)  # 1초 분량의 더미 오디오 데이터 생성
dummy_input = processor(dummy_audio, return_tensors="pt", sampling_rate=16000).input_values

# PyTorch 모델을 ONNX로 변환
onnx_model_path = "emotion_recognition.onnx"
torch.onnx.export(model, dummy_input, onnx_model_path, input_names=["input"], output_names=["output"], opset_version=14)

# OpenVINO로 변환
core = Core()
onnx_model = core.read_model(model=onnx_model_path)
compiled_model = core.compile_model(onnx_model, device_name="CPU")

print("Model successfully converted to OpenVINO IR format.")
