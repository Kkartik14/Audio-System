"this code is for the approach-1 but it makes splits/chunks as per user defined time duration and divides the chunks based on conclusion of sentences/words in real time"
import sounddevice as sd
import numpy as np
import wave
import threading
import time
import queue
import platform
from datetime import datetime
import whisper
import librosa
import os
import signal

class AudioCapturer:
    def __init__(self, chunk_duration=3, sample_rate=44100):
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.is_recording = False
        self.is_processing = False
        self.output_folder = "audio_segments"
        self.os_type = platform.system()
        self.segment_counter = 1

        os.makedirs(self.output_folder, exist_ok=True)
        
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        self.processing_queue = queue.Queue()
        
        self.select_audio_devices()
        self.whisper_model = whisper.load_model("tiny")
        self.original_buffer = np.empty((0, 2), dtype=np.float32)
        self.lock = threading.Lock()
        self.get_split_preferences()

    def get_split_preferences(self):
      while True:
          try:
              self.min_split = float(input("Enter minimum split time (seconds, default 5): "))
              if self.min_split <= 0:
                  print("Minimum split time must be greater than zero.")
              else:
                break
          except ValueError:
            print("Invalid input. Please enter a number.")

      while True:
          try:
              self.max_split = float(input("Enter maximum split time (seconds, default 8): "))
              if self.max_split <= self.min_split:
                  print("Maximum split time must be greater than the minimum.")
              else:
                break
          except ValueError:
              print("Invalid input. Please enter a number.")

    def get_input_devices(self):
        devices = sd.query_devices()
        input_devices = []
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append((i, device))
        return input_devices

    def get_output_devices(self):
        devices = sd.query_devices()
        output_devices = []
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:  
                output_devices.append((i, device))
        return output_devices

    def select_audio_devices(self):
        print("\nAvailable microphone devices:")
        input_devices = self.get_input_devices()
        for idx, (device_id, device) in enumerate(input_devices):
            print(f"{idx}: {device['name']} (inputs: {device['max_input_channels']})")

        while True:
            try:
                selection = int(input("\nEnter microphone device number: "))
                if 0 <= selection < len(input_devices):
                    self.mic_device = input_devices[selection][0]
                    self.mic_channels = min(2, input_devices[selection][1]['max_input_channels'])
                    break
                else:
                    print("Invalid selection. Please try again.")
            except (ValueError, IndexError):
                print("Invalid input. Try again.")

        print("\nAvailable system audio devices (with loopback capability):")
        output_devices = self.get_output_devices()
        for idx, (device_id, device) in enumerate(output_devices):
            print(f"{idx}: {device['name']} (inputs: {device['max_input_channels']})")

        while True:
            try:
                selection = int(input("\nEnter system audio device number: "))
                if 0 <= selection < len(output_devices):
                    self.system_device = output_devices[selection][0]
                    self.system_channels = min(2, output_devices[selection][1]['max_input_channels'])
                    break
                else:
                    print("Invalid selection. Please try again.")
            except (ValueError, IndexError):
                print("Invalid input. Try again.")

    def mic_callback(self, indata, frames, time, status):
        if status:
            print(f"Mic status: {status}")
        self.processing_queue.put((indata.copy(), 'mic'))

    def system_callback(self, indata, frames, time, status):
        if status:
            print(f"System status: {status}")
        self.processing_queue.put((indata.copy(), 'system'))

    def mix_audio(self, mic_data, system_data):
        mic_stereo = mic_data if mic_data.shape[1] == 2 else np.column_stack((mic_data, mic_data))
        system_stereo = system_data if system_data.shape[1] == 2 else np.column_stack((system_data, system_data))
        return (system_stereo * 0.7 + mic_stereo * 0.3)

    def process_audio(self):
        while self.is_processing:
            try:
                mic_data = system_data = None
                while True:
                    data, dtype = self.processing_queue.get(timeout=0.5)
                    if dtype == 'mic':
                        mic_data = data
                    else:
                        system_data = data
                    
                    if mic_data is not None and system_data is not None:
                        break
                mixed = self.mix_audio(mic_data, system_data)
                
                with self.lock:
                    self.original_buffer = np.vstack((self.original_buffer, mixed))
                    current_duration = len(self.original_buffer) / self.sample_rate
                    if current_duration < 0.5:  
                        continue
                    mono_audio = np.mean(self.original_buffer, axis=1)
                    resampled = librosa.resample(
                        mono_audio.astype(np.float32),
                        orig_sr=self.sample_rate,
                        target_sr=16000
                    )

                    result = self.whisper_model.transcribe(resampled, fp16=False)
                    segments = result['segments']

                    split_time = None
                    for seg in reversed(segments):
                        end = seg['end'] * (self.sample_rate / 16000)
                        if self.min_split <= end <= self.max_split:
                            split_time = end
                            break

                    if not split_time and current_duration >= self.max_split:
                        split_time = self.max_split
                    
                    if split_time:
                        split_sample = int(split_time * self.sample_rate)
                        split_sample = min(split_sample, len(self.original_buffer))
                        self.save_segment(self.original_buffer[:split_sample])
                        self.original_buffer = self.original_buffer[split_sample:]
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {str(e)}")

    def save_segment(self, data):
        filename = f"{self.output_folder}/segment-{self.segment_counter}.wav"
        int_data = (data * 32767).astype(np.int16)
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(int_data.tobytes())
        print(f"Saved segment: {filename}")
        self.segment_counter += 1

    def start(self):
        self.is_recording = True
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self.process_audio)
        self.processing_thread.start()
        with sd.InputStream(device=self.mic_device,
                          channels=self.mic_channels,
                          samplerate=self.sample_rate,
                          callback=self.mic_callback,
                          blocksize=self.chunk_size), \
             sd.InputStream(device=self.system_device,
                          channels=self.system_channels,
                          samplerate=self.sample_rate,
                          callback=self.system_callback,
                          blocksize=self.chunk_size):
            
            print("\nPress Ctrl+C to stop recording...")
            print("Recording started...")
            while self.is_recording:
                time.sleep(0.1)

    def stop(self):
        self.is_recording = False
        self.is_processing = False
        self.processing_thread.join()
        with self.lock:
            if len(self.original_buffer) > 0:
                self.save_segment(self.original_buffer)

def signal_handler(signum, frame):
    print("\nStopping recording...")
    if 'capturer' in globals():
        capturer.stop()
    print("Recording stopped. Check audio_segments folder.")
    exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    try:
        capturer = AudioCapturer(chunk_duration=3)
        capturer.start()
    except Exception as e:
        print(f"Error: {str(e)}")
        if 'capturer' in globals():
            capturer.stop()

