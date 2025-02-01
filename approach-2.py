"this is according to the approach 2nd as mentioned in scope of work defined"
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
import contextlib 

class AudioCapturer:
    def __init__(self, chunk_duration=3, sample_rate=44100):
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.is_recording = False
        self.is_processing = False
        self.base_folder = "audio_segments"
        self.os_type = platform.system()
        self.segment_counter = {'mic': 1, 'system': 1, 'mixed': 1}

        os.makedirs(self.base_folder, exist_ok=True)
        
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        self.processing_queue = queue.Queue()
        
        self.get_stream_preference()
        self.select_audio_devices()
        self.whisper_model = whisper.load_model("tiny")
        
        self.buffers = {
            'mic': np.empty((0, 2), dtype=np.float32),
            'system': np.empty((0, 2), dtype=np.float32),
            'mixed': np.empty((0, 2), dtype=np.float32)
        }
        self.lock = threading.Lock()
        self.get_split_preferences()
        
        self.setup_directories()

    def setup_directories(self):
        if self.process_stream == 'both':
            self.output_folders = {
                'mic': os.path.join(self.base_folder, 'microphone'),
                'system': os.path.join(self.base_folder, 'system'),
                'mixed': os.path.join(self.base_folder, 'mixed')
            }
        elif self.process_stream == 'mic':
            self.output_folders = {
                'mic': self.base_folder
            }
        else:  # system
            self.output_folders = {
                'system': self.base_folder
            }
            
        for folder in self.output_folders.values():
            os.makedirs(folder, exist_ok=True)

    def get_stream_preference(self):
        while True:
            preference = input("Process only 'mic', only 'system', or 'both' streams? (mic/system/both): ").lower()
            if preference in ['mic', 'system', 'both']:
                self.process_stream = preference
                break
            else:
                print("Invalid input. Please enter 'mic', 'system', or 'both'.")

    def get_split_preferences(self):
        while True:
            try:
                self.min_split = float(input("Enter minimum split time (seconds, default 5): ") or "5")
                if self.min_split <= 0:
                    print("Minimum split time must be greater than zero.")
                else:
                    break
            except ValueError:
                print("Invalid input. Please enter a number.")

        while True:
            try:
                self.max_split = float(input("Enter maximum split time (seconds, default 8): ") or "8")
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

    def get_system_audio_device(self):
        devices = sd.query_devices()
        
        if self.os_type == 'Windows':
            default_output = sd.default.device[1]
            device = sd.query_devices(default_output)
            if device['max_output_channels'] > 0:
                return default_output, min(2, device['max_output_channels'])
                
        else:  # MacOS or Linux
            virtual_devices = ['blackhole', 'soundflower']
            for i, device in enumerate(devices):
                if any(v in device['name'].lower() for v in virtual_devices):
                    return i, min(2, device['max_input_channels'])
        
        raise RuntimeError("No suitable system audio device found. Please install BlackHole (Mac) or check audio settings (Windows)")

    def select_audio_devices(self):
        if self.process_stream != 'system':  
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

        if self.process_stream in ['system', 'both']:
            try:
                self.system_device, self.system_channels = self.get_system_audio_device()
                device_info = sd.query_devices(self.system_device)
                print(f"\nAutomatically selected system audio device: {device_info['name']}")
            except RuntimeError as e:
                print(f"\nError: {str(e)}")
                raise

    def mic_callback(self, indata, frames, time, status):
        if status:
            print(f"Mic status: {status}")
        if self.process_stream in ['mic', 'both']:
            self.processing_queue.put((indata.copy(), 'mic'))

    def system_callback(self, indata, frames, time, status):
        if status:
            print(f"System status: {status}")
        if self.process_stream in ['system', 'both']:
            self.processing_queue.put((indata.copy(), 'system'))

    def mix_audio(self, mic_data, system_data):
        def normalize_audio(audio):
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                return audio / max_val
            return audio

        mic_stereo = mic_data if mic_data.shape[1] == 2 else np.column_stack((mic_data, mic_data))
        system_stereo = system_data if system_data.shape[1] == 2 else np.column_stack((system_data, system_data))
        
        mic_stereo = normalize_audio(mic_stereo)
        system_stereo = normalize_audio(system_stereo)
        
        return (system_stereo * 0.8 + mic_stereo * 0.2)

    def process_audio(self):
        while self.is_processing:
            try:
                mic_data = system_data = None
                while True:
                    try:
                        data, dtype = self.processing_queue.get(timeout=0.5)

                        if dtype == 'mic':
                            mic_data = data
                        elif dtype == 'system':
                            system_data = data

                        if self.process_stream == 'both':
                            if mic_data is not None and system_data is not None:
                                break
                        elif self.process_stream == 'mic' and mic_data is not None:
                            break
                        elif self.process_stream == 'system' and system_data is not None:
                            break
                    except queue.Empty:
                        if self.process_stream == 'both' and not (mic_data and system_data):
                            continue
                        if self.process_stream == 'mic' and not mic_data:
                            continue
                        if self.process_stream == 'system' and not system_data:
                            continue
                        break

                with self.lock:
                    if self.process_stream == 'both':
                        if mic_data is not None:
                            mic_stereo = mic_data if mic_data.shape[1] == 2 else np.column_stack((mic_data, mic_data))
                            self.buffers['mic'] = np.vstack((self.buffers['mic'], mic_stereo))
                        
                        if system_data is not None:
                            system_stereo = system_data if system_data.shape[1] == 2 else np.column_stack((system_data, system_data))
                            self.buffers['system'] = np.vstack((self.buffers['system'], system_stereo))
                        
                        mixed = self.mix_audio(mic_data, system_data)
                        self.buffers['mixed'] = np.vstack((self.buffers['mixed'], mixed))
                        
                        for buffer_type in ['mic', 'system', 'mixed']:
                            self.process_buffer(buffer_type)
                            
                    elif self.process_stream == 'mic':
                        mic_stereo = mic_data if mic_data.shape[1] == 2 else np.column_stack((mic_data, mic_data))
                        self.buffers['mic'] = np.vstack((self.buffers['mic'], mic_stereo))
                        self.process_buffer('mic')
                        
                    elif self.process_stream == 'system':
                        system_stereo = system_data if system_data.shape[1] == 2 else np.column_stack((system_data, system_data))
                        self.buffers['system'] = np.vstack((self.buffers['system'], system_stereo))
                        self.process_buffer('system')
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {str(e)}")

    def process_buffer(self, buffer_type):
        current_duration = len(self.buffers[buffer_type]) / self.sample_rate
        if current_duration < 0.5:
            return

        mono_audio = np.mean(self.buffers[buffer_type], axis=1)
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
            split_sample = min(split_sample, len(self.buffers[buffer_type]))
            self.save_segment(self.buffers[buffer_type][:split_sample], buffer_type)
            self.buffers[buffer_type] = self.buffers[buffer_type][split_sample:]

    def save_segment(self, data, buffer_type):
        max_val = np.max(np.abs(data))
        if max_val > 0:
            normalized_data = data / max_val
        else:
            normalized_data = data
            
        int_data = (normalized_data * 32767).astype(np.int16)
        
        filename = f"{self.output_folders[buffer_type]}/segment-{self.segment_counter[buffer_type]}.wav"
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(int_data.tobytes())
        print(f"Saved {buffer_type} segment: {filename}")
        self.segment_counter[buffer_type] += 1

    def start(self):
        self.is_recording = True
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self.process_audio)
        self.processing_thread.start()
        
        streams = []
        
        if self.process_stream in ['mic', 'both']:
            streams.append(sd.InputStream(device=self.mic_device,
                          channels=self.mic_channels,
                          samplerate=self.sample_rate,
                          callback=self.mic_callback,
                          blocksize=self.chunk_size))
            
        if self.process_stream in ['system', 'both']:
            if self.os_type == 'Windows':
                try:
                    streams.append(sd.InputStream(
                        device=self.system_device,
                        channels=self.system_channels,
                        samplerate=self.sample_rate,
                        callback=self.system_callback,
                        blocksize=self.chunk_size,
                        wasapi_loopback=True
                    ))
                except Exception as e:
                    print(f"Failed to initialize WASAPI loopback: {e}")
                    print("Falling back to default input...")
                    streams.append(sd.InputStream(
                        device=self.system_device,
                        channels=self.system_channels,
                        samplerate=self.sample_rate,
                        callback=self.system_callback,
                        blocksize=self.chunk_size
                    ))
            else:
                streams.append(sd.InputStream(
                    device=self.system_device,
                    channels=self.system_channels,
                    samplerate=self.sample_rate,
                    callback=self.system_callback,
                    blocksize=self.chunk_size
                ))
            
        if streams:
            with contextlib.ExitStack() as stack:
                for stream in streams:
                    stack.enter_context(stream)
                print("\nRecording started...")
                print("Press 'x' and Enter to stop recording...")
                while self.is_recording:
                    time.sleep(0.1)

    def stop(self):
        self.is_recording = False
        self.is_processing = False
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join()
        with self.lock:
            if self.process_stream == 'both':
                for buffer_type in ['mic', 'system', 'mixed']:
                    if len(self.buffers[buffer_type]) > 0:
                        self.save_segment(self.buffers[buffer_type], buffer_type)
            else:
                buffer_type = self.process_stream
                if len(self.buffers[buffer_type]) > 0:
                    self.save_segment(self.buffers[buffer_type], buffer_type)

if __name__ == "__main__":
    try:
        capturer = AudioCapturer(chunk_duration=3)
        capturer.start()
        while True:
            if input().lower() == 'x':
                break
        capturer.stop()
        print("Recording stopped. Check audio_segments folder.")
    except Exception as e:
        print(f"Error: {str(e)}")
        try:
            capturer.stop()
        except:
            pass
