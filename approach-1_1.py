"this code is for the approach-1 but it makes splits/chunks as per user defined timespan in real time"
import sounddevice as sd
import numpy as np
import wave
import threading
import time
import os
import queue
import platform
from datetime import datetime

class AudioCapturer:
    def __init__(self):
        self.sample_rate = 44100
        self.is_recording = False
        self.output_folder = "audio_chunks"
        self.os_type = platform.system()
        print(f"Detected OS: {self.os_type}")
        while True:
            try:
                self.chunk_duration = float(input("Enter chunk duration in seconds (recommended: 1-10): "))
                if 0.1 <= self.chunk_duration <= 60:  
                    break
                else:
                    print("Please enter a value between 0.1 and 60 seconds.")
            except ValueError:
                print("Please enter a valid number.")
        
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        
        self.mic_queue = queue.Queue()
        self.system_queue = queue.Queue()
        
        self.select_audio_devices()
        
        self.chunk_number = 0
    
    def select_audio_devices(self):
        """Let user manually select input devices"""
        devices = sd.query_devices()
        print("\nAvailable audio devices:")
        for i, device in enumerate(devices):
            print(f"{i}: {device['name']} (inputs: {device['max_input_channels']}, outputs: {device['max_output_channels']})")
        
        while True:
            try:
                mic_id = int(input("\nEnter the number for your microphone device: "))
                if 0 <= mic_id < len(devices):
                    if devices[mic_id]['max_input_channels'] > 0:
                        self.mic_device = mic_id
                        self.mic_channels = min(2, devices[mic_id]['max_input_channels'])
                        print(f"Selected microphone: {devices[mic_id]['name']}")
                        break
                    else:
                        print("Error: Selected device has no input channels. Please choose an input device.")
                else:
                    print("Error: Invalid device number. Please try again.")
            except ValueError:
                print("Error: Please enter a valid number.")
        
        while True:
            try:
                system_id = int(input("\nEnter the number for your system audio device: "))
                if 0 <= system_id < len(devices):
                    self.system_device = system_id
                    self.system_channels = min(2, max(devices[system_id]['max_input_channels'], 
                                                    devices[system_id]['max_output_channels']))
                    print(f"Selected system audio: {devices[system_id]['name']}")
                    break
                else:
                    print("Error: Invalid device number. Please try again.")
            except ValueError:
                print("Error: Please enter a valid number.")
        
        print(f"\nSelected devices:")
        print(f"Microphone: {devices[self.mic_device]['name']} ({self.mic_channels} channels)")
        print(f"System Audio: {devices[self.system_device]['name']} ({self.system_channels} channels)")
        
    def mic_callback(self, indata, frames, time, status):
        """Callback for microphone data"""
        if status:
            print(f"Mic status: {status}")
        if indata.shape[1] == 1:
            stereo_data = np.column_stack((indata, indata))
        else:
            stereo_data = indata
        self.mic_queue.put(stereo_data.copy())
        
    def system_callback(self, indata, frames, time, status):
        """Callback for system audio data"""
        if status:
            print(f"System status: {status}")
        if indata.shape[1] == 1:
            stereo_data = np.column_stack((indata, indata))
        else:
            stereo_data = indata
        self.system_queue.put(stereo_data.copy())
        
    def save_audio_chunk(self, mic_data, system_data):
        """Save mixed audio data to a WAV file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_folder}/chunk_{timestamp}_{self.chunk_number}.wav"
        mixed_data = (system_data * 0.7 + mic_data * 0.3)
        mixed_int = (mixed_data * 32767).astype(np.int16)
        
        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(2)  
            wav_file.setsampwidth(2)  
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(mixed_int.tobytes())
        print(f"Saved chunk {self.chunk_number}")
        self.chunk_number += 1
            
    def record(self):
        self.is_recording = True
        print("\nRecording started. Press Ctrl+C to stop.")
        
        try:
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
                
                while self.is_recording:
                    try:
                        mic_data = self.mic_queue.get(timeout=1)
                        system_data = self.system_queue.get(timeout=1)
                        self.save_audio_chunk(mic_data, system_data)
                    except queue.Empty:
                        continue
                    
        except KeyboardInterrupt:
            print("\nRecording stopped.")
        except Exception as e:
            print(f"\nError during recording: {str(e)}")
        finally:
            self.is_recording = False
            
    def start(self):
        """Start recording in a separate thread"""
        self.recording_thread = threading.Thread(target=self.record)
        self.recording_thread.start()
        
    def stop(self):
        """Stop recording"""
        self.is_recording = False
        if hasattr(self, 'recording_thread'):
            self.recording_thread.join()

if __name__ == "__main__":
    try:
        capturer = AudioCapturer()
        input("\nPress Enter to start recording...")
        capturer.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping recording...")
        capturer.stop()
        print("Recording stopped. Check the audio_chunks folder for the recorded files.")
    except Exception as e:
        print(f"\nError: {str(e)}")