import librosa
import torch
from flask import Flask, request, jsonify, render_template
import time
from threading import Thread
import numpy as np
import soundfile as sf
from openvino.runtime import Core, Tensor
from transformers import Wav2Vec2Processor

app = Flask(__name__)

# Dictionary to store session data
sessions = {}

# Session timeout (in seconds)
SESSION_TIMEOUT = 300  # 5 minutes

# Load OpenVINO model
core = Core()
model_path = "emotion_recognition.onnx"
compiled_model = core.compile_model(model_path, device_name="CPU")

# Load Wav2Vec2Processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_session', methods=['POST'])
def start_session():
    session_id = request.json.get('session_id')
    if session_id in sessions:
        return jsonify({"message": "Session already exists"}), 400
    sessions[session_id] = {'texts': [], 'emotions': [], 'last_activity': time.time()}
    return jsonify({"message": "Session started", "session_id": session_id})

@app.route('/process_audio', methods=['POST'])
def process_audio():
    session_id = request.form.get('session_id')
    audio_data = request.files['audio']
    
    if session_id not in sessions:
        return jsonify({"message": "Session not found"}), 404
    
    # Update last activity time
    sessions[session_id]['last_activity'] = time.time()

    # Extract emotions from audio
    emotions = extract_emotions(audio_data)
    
    # Transcribe audio to text
    text = transcribe_audio(audio_data)
    
    # Save the text and emotions to the session
    sessions[session_id]['texts'].append(text)
    sessions[session_id]['emotions'].append(emotions)
    
    return jsonify({"message": "Audio processed", "emotions": emotions, "text": text})

@app.route('/end_session', methods=['POST'])
def end_session():
    session_id = request.json.get('session_id')
    
    if session_id not in sessions:
        return jsonify({"message": "Session not found"}), 404
    
    # Summarize the text
    full_text = " ".join(sessions[session_id]['texts'])
    summary = summarize_text(full_text)
    
    # Combine emotions
    combined_emotions = combine_emotions(sessions[session_id]['emotions'])
    
    # Clean up the session data
    del sessions[session_id]
    
    return jsonify({"summary": summary, "emotions": combined_emotions})

def extract_emotions(audio_data):
    # Read the audio file
    audio_input, sample_rate = sf.read(audio_data)
    
    # Resample the audio to 16000Hz if needed
    if sample_rate != 16000:
        audio_input = librosa.resample(audio_input, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    # Ensure the audio is 1 second long
    if len(audio_input) > 16000:
        audio_input = audio_input[:16000]
    elif len(audio_input) < 16000:
        audio_input = np.pad(audio_input, (0, 16000 - len(audio_input)), 'constant')

    # Preprocess the audio data
    input_tensor = processor(audio_input, return_tensors="pt", sampling_rate=sample_rate).input_values

    # Create an OpenVINO tensor from the input data
    input_tensor = Tensor(input_tensor.numpy())

    # Perform inference
    infer_request = compiled_model.create_infer_request()
    infer_request.infer(inputs={0: input_tensor})
    output = infer_request.get_output_tensor(0).data

    # Print the shape of the output to debug
    print(f"Output shape: {output.shape}")

    # Ensure output length and convert to list for JSON serialization
    emotions_list = ["anger", "happiness", "excitement", "sadness", "frustration", "fear", "surprise", "other", "neutral"]
    # Flatten the output array and convert each element to float
    emotions = {emotion: float(output.flatten()[i]) if i < len(output.flatten()) else 0 for i, emotion in enumerate(emotions_list)}

    return emotions

def transcribe_audio(audio_data):
    # Implement the transcription logic or use an existing service like Google Speech-to-Text
    return "This is a transcribed text from the audio."

def combine_emotions(emotions_list):
    combined_emotions = {}
    for emotions in emotions_list:
        for emotion, value in emotions.items():
            if emotion not in combined_emotions:
                combined_emotions[emotion] = 0
            combined_emotions[emotion] += value

    total = sum(combined_emotions.values())
    for emotion in combined_emotions:
        combined_emotions[emotion] /= total
    
    return combined_emotions

def summarize_text(text):
    # Implement text summarization logic here
    return text[:100] + "..."

def cleanup_sessions():
    current_time = time.time()
    expired_sessions = [session_id for session_id, data in sessions.items() 
                        if current_time - data['last_activity'] > SESSION_TIMEOUT]
    for session_id in expired_sessions:
        del sessions[session_id]

def cleanup_sessions_periodically():
    while True:
        cleanup_sessions()
        time.sleep(SESSION_TIMEOUT)

# Start the cleanup thread
cleanup_thread = Thread(target=cleanup_sessions_periodically)
cleanup_thread.daemon = True
cleanup_thread.start()

if __name__ == '__main__':
    app.run(debug=True)