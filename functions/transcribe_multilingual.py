import whisper
import librosa
import requests
import os
import mimetypes
from dotenv import load_dotenv

load_dotenv()

def sarvam_speech_to_text(audio_file_path, model='saaras:v2', prompt=None, with_diarization=False):
    url = "https://api.sarvam.ai/speech-to-text-translate"
    sarvam_api_key = os.getenv('SARVAM_API_KEY')
    if not sarvam_api_key:
        raise ValueError("SARVAM_API_KEY not found in environment variables.")

    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    mime_type, _ = mimetypes.guess_type(audio_file_path)
    if mime_type not in ['audio/mpeg', 'audio/wav', 'audio/wave', 'audio/x-wav']:
        raise ValueError(f"Unsupported audio file type: {mime_type}")

    with open(audio_file_path, 'rb') as audio_file:
        files = {
            'file': (os.path.basename(audio_file_path), audio_file, mime_type)
        }

        data = {
            'model': model,
            'with_diarization': str(with_diarization)
        }

        if prompt:
            data['prompt'] = prompt

        headers = {
            'api-subscription-key': sarvam_api_key
        }

        response = requests.post(url, files=files, data=data, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")


def transcribe_multilingual(audio_file_path, languages=["en"]):

    try:
        model = whisper.load_model("tiny")
        audio, sr = librosa.load(audio_file_path, sr=16000)
        audio = audio.astype("float32")
        result = model.transcribe(audio, fp16=False)
        detected_language = result.get("language", "").lower()

        print(f"Detected language: {detected_language}")

        if detected_language not in ["en", "english"] and detected_language in languages:
            try:
                sarvam_result = sarvam_speech_to_text(audio_file_path)
                translated_text = sarvam_result.get("transcript")
                if translated_text:
                    print(f"Sarvam AI Transcription: {translated_text}")
                    return translated_text
            except Exception as e:
                print(f"Sarvam AI Error: {e}")
                print(f"Fallback to Whisper: {result['text']}")
                return result["text"]
        else:
            print(f"Whisper Transcription: {result['text']}")
            return result["text"]

    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

if __name__ == "__main__":
    audio_file = "/Users/kartik/Desktop/vs/freelance/Audio-System/audio_segments/segment-5.wav"  
    supported_languages = ["en", "hi"] 
    if os.path.exists(audio_file):
      transcribe_multilingual(audio_file, supported_languages)
    else:
        print("Audio file does not exist")