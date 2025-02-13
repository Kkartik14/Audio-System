import asyncio
import io
import logging
import queue
import time
import wave
import numpy as np
import uvicorn
import sounddevice as sd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middlewa1re.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Audio Recorder API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecordingConfig(BaseModel):
    target_url: str
    min_split: float = 5.0
    max_split: float = 8.0
    sample_rate: int = 44100
    channels: int = 1  


class AudioRecorder:
    def __init__(self):
        self.recording = False
        self.sample_rate = 44100
        self.channels = 1  
        self.chunk_size = int(self.sample_rate * 0.1) 
        self.audio_queue = asyncio.Queue(
            maxsize=100
        )  

        self.device_info = {}
        self.mic_buffer = np.array([], dtype=np.float32)  
        self.system_buffer = np.array([], dtype=np.float32) 
        self.buffer = np.array([], dtype=np.float32)
        self.setup_devices()

    def setup_devices(self):
        try:
            devices = sd.query_devices()
            default_input = sd.default.device[0]
            system_device = None
            for i, device in enumerate(devices):
                if device["max_input_channels"] > 0:
                    if any(
                        name in device["name"].lower()
                        for name in ["stereo mix", "what u hear", "blackhole", "soundflower"]
                    ):
                        system_device = i
                        break

            self.device_info = {
                "mic": 2,
                "system": 1,
            }

        except Exception as e:
            logger.error(f"Error setting up devices: {e}")
            raise

    def _create_wav_data(self, audio_data):
        with io.BytesIO() as wav_buffer:
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())
            return wav_buffer.getvalue()

    def start_recording(self, mode: str, websocket: WebSocket, config: RecordingConfig):
        self.recording = True
        self.audio_queue = asyncio.Queue(maxsize=100) 
        self.websocket = websocket
        self.config = config
        self.channels = config.channels 
        self.mic_buffer = np.array([], dtype=np.float32)
        self.system_buffer = np.array([], dtype=np.float32)
        self.buffer = np.array([], dtype=np.float32)
        self.mode = mode

        try:
            if mode == "system":
                stream = sd.InputStream(
                    device=self.device_info["system"],
                    channels=self.channels,
                    samplerate=self.sample_rate,
                    callback=self._audio_callback,
                )
                stream.start()
                return [stream]

            elif mode == "mic":
                stream = sd.InputStream(
                    device=self.device_info["mic"],
                    channels=self.channels,
                    samplerate=self.sample_rate,
                    callback=self._audio_callback,
                )
                stream.start()
                return [stream]

            elif mode == "mixed":
                mic_stream = sd.InputStream(
                    device=self.device_info["mic"],
                    channels=self.channels,
                    samplerate=self.sample_rate,
                    callback=lambda indata, frames, time, status: self._mixed_callback(
                        indata, "mic", status
                    ),
                )
                system_stream = sd.InputStream(
                    device=self.device_info["system"],
                    channels=self.channels,
                    samplerate=self.sample_rate,
                    callback=lambda indata, frames, time, status: self._mixed_callback(
                        indata, "system", status
                    ),
                )
                mic_stream.start()
                system_stream.start()
                return [mic_stream, system_stream]

            else:
                raise ValueError(f"Invalid recording mode: {mode}")

        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            raise

    def _audio_callback(self, indata, frames, time, status):
        if status:
            logger.warning(f"Audio callback status: {status}")
        if self.recording:
            self._process_and_enqueue(indata.copy())

    def _mixed_callback(self, indata, source, status):
        if status:
            logger.warning(f"Audio callback status: {status}")

        if self.recording:
            if source == "mic":
                self.mic_buffer = np.append(self.mic_buffer, indata)
            elif source == "system":
                self.system_buffer = np.append(self.system_buffer, indata)
            if (
                len(self.mic_buffer)
                >= int(self.config.min_split * self.sample_rate)
                and len(self.system_buffer)
                >= int(self.config.min_split * self.sample_rate)
            ):
                self._mix_and_enqueue()

    def _mix_and_enqueue(self):
        min_len = min(len(self.mic_buffer), len(self.system_buffer))
        mixed_audio = (self.mic_buffer[:min_len] + self.system_buffer[:min_len]) / 2.0
        self.mic_buffer = self.mic_buffer[min_len:]
        self.system_buffer = self.system_buffer[min_len:]
        self._process_and_enqueue(mixed_audio)

    def _process_and_enqueue(self, audio_chunk):
        try:

            split_samples = int(self.config.min_split * self.sample_rate)
            max_samples = int(self.config.max_split * self.sample_rate)

            self.buffer = np.append(self.buffer, audio_chunk)

            while len(self.buffer) >= split_samples:  
                current_split = min(
                    len(self.buffer), max_samples
                )  
                wav_data = self._create_wav_data(self.buffer[:current_split])
                try:
                    self.audio_queue.put_nowait(
                        (wav_data, time.time())
                    )  
                except queue.Full:
                    logger.warning("Audio queue full, dropping data.")  
                self.buffer = self.buffer[current_split:]  

        except Exception as e:
            logger.error(f"Error in audio processing: {e}")

    def stop_recording(self):
        self.recording = False
        self.mic_buffer = np.array([], dtype=np.float32)
        self.system_buffer = np.array([], dtype=np.float32)
        self.buffer = np.array([], dtype=np.float32)


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)


manager = ConnectionManager()
audio_recorder = AudioRecorder()


@app.websocket("/ws/{mode}")
async def websocket_endpoint(websocket: WebSocket, mode: str):
    if mode not in ["system", "mic", "mixed"]:
        await websocket.close(code=4000, reason="Invalid recording mode")
        return

    await manager.connect(websocket)

    try:
        config_str = await websocket.receive_text()
        config = RecordingConfig.model_validate_json(config_str)
        streams = audio_recorder.start_recording(mode, websocket, config)

        try:
            while audio_recorder.recording:
                try:
                    (
                        audio_data,
                        timestamp,
                    ) = await asyncio.wait_for(  
                        audio_recorder.audio_queue.get(), timeout=0.1
                    )
                    await websocket.send_bytes(audio_data)
                except asyncio.TimeoutError:
                    pass

                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                    if data == "stop":
                        break
                except asyncio.TimeoutError:
                    continue

        except WebSocketDisconnect:
            logger.info("Client disconnected")
        finally:
            audio_recorder.stop_recording()
            for stream in streams:
                stream.stop()
                stream.close()
            manager.disconnect(websocket)

    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}")
        manager.disconnect(websocket)
        await websocket.close(code=4000, reason=str(e))


@app.get("/devices")
async def get_devices():
    """Get available audio devices."""
    devices = sd.query_devices()
    return {
        "devices": [
            {
                "id": i,
                "name": device["name"],
                "channels": device["max_input_channels"],
                "default": i in [sd.default.device[0], sd.default.device[1]],
            }
            for i, device in enumerate(devices)
            if device["max_input_channels"] > 0
        ],
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, ws="websockets")