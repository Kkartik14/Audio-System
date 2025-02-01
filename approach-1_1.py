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
    
    def get_microphone_devices(self):
        """Filter and return only microphone devices"""
        devices = sd.query_devices()
        mic_devices = []
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                mic_keywords = ['mic', 'microphone', 'input', 'headset']
                device_name = device['name'].lower()
                if any(keyword in device_name for keyword in mic_keywords) or device.get('default_input', False):
                    mic_devices.append((i, device))
        
        return mic_devices
    
    def get_system_audio_device(self):
        """Automatically select system audio device based on OS"""
        devices = sd.query_devices()
        
        if self.os_type == 'Windows':
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    name = device['name'].lower()
                    if 'stereo mix' in name or 'what u hear' in name or 'loopback' in name:
                        return i, device

            default_output = sd.default.device[1]
            if default_output is not None:
                return default_output, devices[default_output]
        
        elif self.os_type == 'Darwin':  
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    name = device['name'].lower()
                    if 'blackhole' in name or 'soundflower' in name:
                        return i, device
        
        elif self.os_type == 'Linux':
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    name = device['name'].lower()
                    if 'monitor' in name or 'pulse' in name or 'pipewire' in name:
                        return i, device
        
        default_input = sd.default.device[0]
        return default_input, devices[default_input]
    
    def select_audio_devices(self):
        """Let user select microphone and automatically set system audio"""
        mic_devices = self.get_microphone_devices()
        
        print("\nAvailable microphones:")
        for i, (device_id, device) in enumerate(mic_devices):
            print(f"{i}: {device['name']}")
        
        while True:
            try:
                selection = int(input("\nEnter the number for your microphone: "))
                if 0 <= selection < len(mic_devices):
                    device_id, device = mic_devices[selection]
                    self.mic_device = device_id
                    self.mic_channels = min(2, device['max_input_channels'])
                    print(f"Selected microphone: {device['name']}")
                    break
                else:
                    print("Error: Invalid selection. Please try again.")
            except ValueError:
                print("Error: Please enter a valid number.")
        
        system_id, system_device = self.get_system_audio_device()
        self.system_device = system_id
        self.system_channels = min(2, max(system_device['max_input_channels'], 
                                        system_device.get('max_output_channels', 0)))
        print(f"\nAutomatically selected system audio: {system_device['name']}")
        
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
        
    def normalize_audio(self, audio_data):
        """Normalize audio data to prevent clipping"""
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            return audio_data / max_val
        return audio_data
        
    def save_audio_chunk(self, mic_data, system_data):
        """Save mixed audio data to a WAV file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_folder}/chunk_{timestamp}_{self.chunk_number}.wav"
        mic_normalized = self.normalize_audio(mic_data)
        system_normalized = self.normalize_audio(system_data)
        mixed_data = (system_normalized * 0.5 + mic_normalized * 0.5)  
        gain = 1.0 
        mixed_data = mixed_data * gain
        mixed_data = np.clip(mixed_data, -1.0, 1.0)
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
