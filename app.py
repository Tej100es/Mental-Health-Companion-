# back.py

from flask import Flask, render_template, request, jsonify, send_file
import torch
import time
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import librosa
from pydub import AudioSegment
from pydub.silence import split_on_silence
import speech_recognition as sr
import requests
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import uuid
import os

app = Flask(__name__)

# Initialize emotion detection model
model_name = "superb/wav2vec2-base-superb-er"
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

SAMPLERATE = 44100 
CHANNELS = 1

# ElevenLabs API Setup
API_KEY = "sk_605fa7b267f418f577686c1996ac40a6b1b93e9a89387e37"
VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
def speech_to_text(audio_path):
    print(f"Processing audio file: {audio_path}")  # Log the file being processed
    try:
        with sr.AudioFile(audio_path) as source:
            print("Recognizing speech...")  # Log recognition start
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)  # Can also use other engines like recognize_sphinx()
            print(f"Recognized text: {text}")  # Log the recognized text
            return text
    except Exception as e:
        print(f"Error in speech recognition: {e}")  # Log any errors
        return None


def record_audio():
    duration = 6  # seconds
    print("Recording... Speak now!")
    recording = sd.rec(int(duration * SAMPLERATE), samplerate=SAMPLERATE, channels=CHANNELS, dtype='float32')
    sd.wait()
    filename = "temp_recording.wav"
    wav.write(filename, SAMPLERATE, (recording * 32767).astype(np.int16))  # convert to int16
    return filename

def clean_audio(path):
    sound = AudioSegment.from_wav(path)
    audio_chunks = split_on_silence(sound, min_silence_len=500, silence_thresh=sound.dBFS-14, keep_silence=200)
    cleaned_audio = AudioSegment.empty()
    for chunk in audio_chunks:
        cleaned_audio += chunk
    cleaned_path = "cleaned_audio.wav"
    cleaned_audio.export(cleaned_path, format="wav")
    return cleaned_path

def emotion_detector(audio_path):
    audio, sample_rate = librosa.load(audio_path, sr=16000)
    inputs = feature_extractor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    emotion_labels = ["Anger", "Disgust", "Fear", "Happy", "Sadness", "Surprise", "Neutral"]
    return emotion_labels[predicted_class]

def speech_text(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        try:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language="en-US", show_all=False)
            return text
        except sr.UnknownValueError:
            app.logger.warning("Google Speech Recognition could not understand audio")
            return ""
        except sr.RequestError as e:
            app.logger.error(f"Could not request results from Google Speech Recognition service; {e}")
            return ""

def generate_response(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral:7b-instruct-q4_0",
                "prompt": f"[INST] <<SYS>>Respond as an empathetic mental health coach<</SYS>> {prompt} [/INST]",
                "stream": False,
                "options": {
                    "temperature": 0.7
                }
            }
        )
        return response.json().get("response", "I'm here for you. Could you elaborate?")
    except Exception as e:
        return "Let's focus on breathing. Try this: 4-second inhale, 6-second exhale."

def speak(text):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "xi-api-key": API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.75,
            "similarity_boost": 0.75
        }
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        filename = f"response_{uuid.uuid4()}.mp3"
        filepath = os.path.join("static", filename)
        with open(filepath, "wb") as f:
            f.write(response.content)
        return filepath
    else:
        print("Error from ElevenLabs:", response.text)
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/process_voice', methods=['POST'])
def process_voice():
    
    try:
        audio_path = record_audio()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    cleaned_path = clean_audio(audio_path)
    user_input = speech_text(cleaned_path)
    
    if not user_input:
        return jsonify({
            "error": "Could not understand audio",
            "fallback": "Please try speaking again clearly"
        }), 400

    emotion = emotion_detector(cleaned_path)
    prompt = f"User input: {user_input}\nUser emotion: {emotion}\nRespond empathetically."

    try:
        response = generate_response(prompt)
        audio_path = speak(response)
        
        if audio_path:
            return send_file(audio_path, mimetype="audio/mpeg")
        return jsonify({
            "emotion": emotion,
            "user_input": user_input,
            "response": response
        })
    except Exception as e:
        app.logger.error(f"Processing failed: {str(e)}")
        return jsonify({
            "error": "Processing error",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
    
