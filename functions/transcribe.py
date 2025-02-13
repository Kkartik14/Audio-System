import whisper
import librosa
import os

def transcribe_with_whisper(audio_file_path):

    try:
        model = whisper.load_model("tiny")  
        audio, sr = librosa.load(audio_file_path, sr=16000)  
        audio = audio.astype("float32")
        result = model.transcribe(audio, fp16=False) 

        print(result["text"])
        return result["text"] 

    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

if __name__ == "__main__":
    audio_file = "/Users/kartik/Desktop/vs/freelance/Audio-System/audio_segments/segment-3.wav" 
    if os.path.exists(audio_file):
        transcribe_with_whisper(audio_file)
    else:
        print("Audio file does not exist")