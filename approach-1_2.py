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
import contextlib

class AudioCapturer:
    def __init__(self, chunk_duration=3, sample_rate=44100):
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.is_recording = False
        self.is_processing = False
        self.output_folder = "audio_segments"
        os.makedirs(self.output_folder, exist_ok=True)
        self.os_type = platform.system()
        self.segment_counter = 1

        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        self.processing_queue = queue.Queue()
        self.get_stream_preference()

        if self.process_stream == 'both':
            while True:
                choice = input("For storing splits, type 'mixed' for mixed audio or 'separate' for separate mic and system folders: ").lower()
                if choice in ['mixed', 'separate']:
                    self.store_mode = choice
                    break
                else:
                    print("Invalid input. Please enter 'mixed' or 'separate'.")
        else:
            self.store_mode = None

        if self.process_stream == 'both' and self.store_mode == 'separate':
            self.output_folder_mic = os.path.join(self.output_folder, "mic")
            self.output_folder_system = os.path.join(self.output_folder, "system")
            os.makedirs(self.output_folder_mic, exist_ok=True)
            os.makedirs(self.output_folder_system, exist_ok=True)
            self.mic_segment_counter = 1
            self.system_segment_counter = 1
            self.mic_buffer = np.empty((0, 2), dtype=np.float32)
            self.system_buffer = np.empty((0, 2), dtype=np.float32)
        else:
            self.original_buffer = np.empty((0, 2), dtype=np.float32)

        self.select_audio_devices()
        self.whisper_model = whisper.load_model("tiny")
        self.lock = threading.Lock()
        self.get_split_preferences()

    def get_stream_preference(self):
        """Prompt user for a recording stream preference."""
        while True:
            preference = input("Record only 'mic', only 'system', or 'both' streams? (mic/system/both): ").lower()
            if preference in ['mic', 'system', 'both']:
                self.process_stream = preference
                break
            else:
                print("Invalid input. Please enter 'mic', 'system', or 'both'.")

    def get_split_preferences(self):
        """Prompt user for minimum and maximum split times."""
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
        """Return a list of available input devices."""
        devices = sd.query_devices()
        input_devices = []
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append((i, device))
        return input_devices

    def select_system_audio_device(self):
        """
        Select a candidate system audio device. For Windows, attempt to find candidates like 'stereo mix'.
        For other systems, look for virtual devices.
        """
        devices = sd.query_devices()
        system_candidates = []
        if self.os_type == 'Windows':
            for i, device in enumerate(devices):
                if device['max_output_channels'] > 0 and ('stereo mix' in device['name'].lower() or 'what u hear' in device['name'].lower()):
                    system_candidates.append((i, device))
            if not system_candidates:
                default_output = sd.default.device[1]
                system_candidates.append((default_output, sd.query_devices(default_output)))
        else:
            virtual_keywords = ['blackhole', 'soundflower', 'monitor', 'pulse']
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0 and any(v in device['name'].lower() for v in virtual_keywords):
                    system_candidates.append((i, device))
            if not system_candidates:
                system_candidates = [(i, device) for i, device in enumerate(devices) if device['max_input_channels'] > 0]
        chosen_id, chosen_device = system_candidates[0]
        if self.os_type == 'Windows':
            channels = min(2, chosen_device['max_output_channels'])
        else:
            channels = min(2, chosen_device['max_input_channels'])
        print(f"\nSelected system audio device: {chosen_device['name']}")
        return chosen_id, channels

    def select_audio_devices(self):
        """Select microphone and/or system audio devices based on the stream preference."""
        if self.process_stream in ['mic', 'both']:
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
                self.system_device, self.system_channels = self.select_system_audio_device()
            except RuntimeError as e:
                print(f"\nError: {str(e)}")
                raise

    def mic_callback(self, indata, frames, time_info, status):
        """Callback for microphone input stream."""
        if status:
            print(f"Mic status: {status}")
        if indata.ndim == 2 and indata.shape[1] == 1:
            indata = np.column_stack((indata, indata))
        self.processing_queue.put((indata.copy(), 'mic'))

    def system_callback(self, indata, frames, time_info, status):
        """Callback for system audio input stream."""
        if status:
            print(f"System status: {status}")
        if indata.ndim == 2 and indata.shape[1] == 1:
            indata = np.column_stack((indata, indata))
        self.processing_queue.put((indata.copy(), 'system'))

    def mix_audio(self, mic_data, system_data):
        """
        Normalize and mix mic and system audio data.
        Both inputs are forced to stereo if needed.
        """
        def normalize_audio(audio):
            max_val = np.max(np.abs(audio))
            return audio / max_val if max_val > 0 else audio
        
        if mic_data.shape[1] != 2:
            mic_data = np.column_stack((mic_data, mic_data))
        if system_data.shape[1] != 2:
            system_data = np.column_stack((system_data, system_data))
        
        mic_stereo = normalize_audio(mic_data)
        system_stereo = normalize_audio(system_data)
        mixed = (system_stereo * 0.5 + mic_stereo * 0.5)
        gain = 1.2
        mixed = mixed * gain
        return np.clip(mixed, -1.0, 1.0)

    def process_buffer_generic(self, buffer, save_callback):
        """
        Process a given buffer: if its duration is sufficient, use Whisper to transcribe,
        determine a split point, and then save the segment using the provided callback.
        Returns the remaining buffer after the split.
        """
        current_duration = len(buffer) / self.sample_rate
        if current_duration < 0.5:
            return buffer

        mono_audio = np.mean(buffer, axis=1)
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
            split_sample = min(split_sample, len(buffer))
            save_callback(buffer[:split_sample])
            return buffer[split_sample:]
        return buffer

    def save_segment(self, data):
        """Save an audio segment (used for single or mixed stream)."""
        filename = f"{self.output_folder}/segment-{self.segment_counter}.wav"
        int_data = (data * 32767).astype(np.int16)
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(int_data.tobytes())
        print(f"Saved segment: {filename}")
        self.segment_counter += 1

    def save_mic_segment(self, data):
        """Save a microphone segment to its dedicated folder."""
        filename = f"{self.output_folder_mic}/segment-{self.mic_segment_counter}.wav"
        int_data = (data * 32767).astype(np.int16)
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(int_data.tobytes())
        print(f"Saved microphone segment: {filename}")
        self.mic_segment_counter += 1

    def save_system_segment(self, data):
        """Save a system audio segment to its dedicated folder."""
        filename = f"{self.output_folder_system}/segment-{self.system_segment_counter}.wav"
        int_data = (data * 32767).astype(np.int16)
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(int_data.tobytes())
        print(f"Saved system segment: {filename}")
        self.system_segment_counter += 1

    def process_audio(self):
        """
        Process audio data from the queue, transcribe using Whisper,
        and save segments when the criteria are met.
        """
        while self.is_processing:
            try:
                if self.process_stream == 'both' and self.store_mode == 'mixed':
                    mic_data = system_data = None
                    while True:
                        data, dtype = self.processing_queue.get(timeout=0.5)
                        if dtype == 'mic':
                            mic_data = data
                        elif dtype == 'system':
                            system_data = data
                        if mic_data is not None and system_data is not None:
                            break
                    mixed = self.mix_audio(mic_data, system_data)
                    with self.lock:
                        self.original_buffer = np.vstack((self.original_buffer, mixed))
                        self.original_buffer = self.process_buffer_generic(self.original_buffer, self.save_segment)
                elif self.process_stream == 'both' and self.store_mode == 'separate':
                    data, dtype = self.processing_queue.get(timeout=0.5)
                    with self.lock:
                        if dtype == 'mic':
                            self.mic_buffer = np.vstack((self.mic_buffer, data))
                        elif dtype == 'system':
                            self.system_buffer = np.vstack((self.system_buffer, data))
                        mic_len = len(self.mic_buffer)
                        system_len = len(self.system_buffer)
                        min_len = min(mic_len, system_len)
                        duration = min_len / self.sample_rate
                        if duration >= 0.5:  
                            combined_audio = (
                                np.mean(self.mic_buffer[:min_len], axis=1) +
                                np.mean(self.system_buffer[:min_len], axis=1)
                            ) / 2
                            resampled = librosa.resample(
                                combined_audio.astype(np.float32),
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
                            if not split_time and duration >= self.max_split:
                                split_time = self.max_split
                            if split_time:
                                split_sample = int(split_time * self.sample_rate)
                                split_sample = min(split_sample, min_len)
                                mic_segment = self.mic_buffer[:split_sample]
                                system_segment = self.system_buffer[:split_sample]
                                self.save_mic_segment(mic_segment)
                                self.save_system_segment(system_segment)
                                self.mic_buffer = self.mic_buffer[split_sample:]
                                self.system_buffer = self.system_buffer[split_sample:]
                else:
                    expected_type = self.process_stream
                    while True:
                        data, dtype = self.processing_queue.get(timeout=0.5)
                        if dtype == expected_type:
                            break
                    with self.lock:
                        self.original_buffer = np.vstack((self.original_buffer, data))
                        self.original_buffer = self.process_buffer_generic(self.original_buffer, self.save_segment)
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {str(e)}")

    def start(self):
        """Start recording and processing the audio streams."""
        self.is_recording = True
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self.process_audio)
        self.processing_thread.start()
        
        system_stream = None
        if self.process_stream in ['system', 'both']:
            if self.os_type == 'Windows':
                hostapi_info = sd.query_hostapis(sd.default.hostapi)
                if hostapi_info['name'] == 'Windows WASAPI':
                    extra_settings = {
                        'channel_map': list(range(self.system_channels)),
                        'wasapi_exclusive': False
                    }
                    try:
                        system_stream = sd.InputStream(
                            device=self.system_device,
                            channels=self.system_channels,
                            samplerate=self.sample_rate,
                            callback=self.system_callback,
                            blocksize=self.chunk_size,
                            extra_settings=extra_settings
                        )
                    except Exception as e:
                        print(f"Failed to initialize WASAPI settings: {e}")
                        system_stream = sd.InputStream(
                            device=self.system_device,
                            channels=self.system_channels,
                            samplerate=self.sample_rate,
                            callback=self.system_callback,
                            blocksize=self.chunk_size
                        )
                else:
                    system_stream = sd.InputStream(
                        device=self.system_device,
                        channels=self.system_channels,
                        samplerate=self.sample_rate,
                        callback=self.system_callback,
                        blocksize=self.chunk_size
                    )
            else:
                system_stream = sd.InputStream(
                    device=self.system_device,
                    channels=self.system_channels,
                    samplerate=self.sample_rate,
                    callback=self.system_callback,
                    blocksize=self.chunk_size
                )

        mic_stream = None
        if self.process_stream in ['mic', 'both']:
            mic_stream = sd.InputStream(
                device=self.mic_device,
                channels=self.mic_channels,
                samplerate=self.sample_rate,
                callback=self.mic_callback,
                blocksize=self.chunk_size
            )

        streams = []
        if mic_stream:
            streams.append(mic_stream)
        if system_stream:
            streams.append(system_stream)

        with contextlib.ExitStack() as stack:
            for stream in streams:
                stack.enter_context(stream)
            print("\nPress Ctrl+C to stop recording...")
            print("Recording started...")
            while self.is_recording:
                time.sleep(0.1)

    def stop(self):
        """Stop recording and finish processing."""
        self.is_recording = False
        self.is_processing = False
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join()
        with self.lock:
            if self.process_stream == 'both' and self.store_mode == 'separate':
                if len(self.mic_buffer) > 0:
                    self.save_mic_segment(self.mic_buffer)
                if len(self.system_buffer) > 0:
                    self.save_system_segment(self.system_buffer)
            else:
                if len(self.original_buffer) > 0:
                    self.save_segment(self.original_buffer)

def signal_handler(signum, frame):
    """Handle CTRL+C signal for graceful shutdown."""
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