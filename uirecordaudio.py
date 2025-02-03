import tkinter as tk
from tkinter import filedialog
import wave
import sys
import pyaudio
import time
import threading
import pygame
from pygame import mixer

def start_recording():
    global stream, wf, p, recording
    start_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)
    pause_recording_button.config(state=tk.NORMAL)
    resume_recording_button.config(state=tk.DISABLED)

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1 if sys.platform == 'darwin' else 2
    RATE = 44100
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    fileName = f'meeting_{timestamp}.wav'
    
    wf = wave.open(fileName, 'wb')
    p = pyaudio.PyAudio()
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True)
    
    def record():
        print('Recording...')
        while recording:
            if not paused:
                wf.writeframes(stream.read(CHUNK))
        print('Done')
        stream.close()
        p.terminate()
        wf.close()

    recording = True
    global paused
    paused = False
    threading.Thread(target=record).start()

def stop_recording():
    global recording
    recording = False
    stop_button.config(state=tk.DISABLED)
    start_button.config(state=tk.NORMAL)
    pause_recording_button.config(state=tk.DISABLED)
    resume_recording_button.config(state=tk.DISABLED)

def pause_recording():
    global paused
    paused = True
    pause_recording_button.config(state=tk.DISABLED)
    resume_recording_button.config(state=tk.NORMAL)

def resume_recording():
    global paused
    paused = False
    pause_recording_button.config(state=tk.NORMAL)
    resume_recording_button.config(state=tk.DISABLED)

def play_audio():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
    if file_path:
        mixer.init()
        mixer.music.load(file_path)
        mixer.music.play()
        play_button.config(state=tk.DISABLED)
        pause_button.config(state=tk.NORMAL)
        stop_play_button.config(state=tk.NORMAL)
        mixer.music.set_endevent(pygame.USEREVENT)

def stop_audio():
    mixer.music.stop()
    play_button.config(state=tk.NORMAL)
    pause_button.config(state=tk.DISABLED)
    stop_play_button.config(state=tk.DISABLED)

def pause_audio():
    mixer.music.pause()
    pause_button.config(state=tk.DISABLED)
    play_button.config(state=tk.NORMAL)

def check_playback_end():
    for event in pygame.event.get():
        if event.type == pygame.USEREVENT:
            play_button.config(state=tk.NORMAL)
            pause_button.config(state=tk.DISABLED)
            stop_play_button.config(state=tk.DISABLED)
    root.after(100, check_playback_end)

# Initialize pygame
pygame.init()

# Create the main window
root = tk.Tk()
root.title("Audio Recorder")
root.geometry("400x300")  # Set the window size to 400x300 pixels

# Create the Start button
start_button = tk.Button(root, text="Start Recording", command=start_recording)
start_button.pack(pady=10)

# Create the Stop button
stop_button = tk.Button(root, text="Stop Recording", command=stop_recording, state=tk.DISABLED)
stop_button.pack(pady=10)

# Create the Pause Recording button
pause_recording_button = tk.Button(root, text="Pause Recording", command=pause_recording, state=tk.DISABLED)
pause_recording_button.pack(pady=10)

# Create the Resume Recording button
resume_recording_button = tk.Button(root, text="Resume Recording", command=resume_recording, state=tk.DISABLED)
resume_recording_button.pack(pady=10)

# Create the Play button
play_button = tk.Button(root, text="Play Audio", command=play_audio)
play_button.pack(pady=10)

# Create the Pause button
pause_button = tk.Button(root, text="Pause Audio", command=pause_audio, state=tk.DISABLED)
pause_button.pack(pady=10)

# Create the Stop Play button
stop_play_button = tk.Button(root, text="Stop Audio", command=stop_audio, state=tk.DISABLED)
stop_play_button.pack(pady=10)

# Check for playback end
root.after(100, check_playback_end)

# Run the application
root.mainloop()