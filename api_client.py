import asyncio
import wave
import json
import os
import time
import sys
from websockets.client import connect
import argparse  

async def record_and_split(mode, duration, min_split, max_split, sample_rate, channels, output_folder):
    uri = f"ws://localhost:8000/ws/{mode}"  

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    config = {
        "target_url": "dummy", 
        "min_split": min_split,
        "max_split": max_split,
        "sample_rate": sample_rate,
        "channels": channels
    }

    async with connect(uri) as websocket:
        await websocket.send(json.dumps(config))
        print(f"Recording started ({mode} mode).  Recording for {duration} seconds...")

        start_time = time.time()
        chunk_counter = 1

        while time.time() - start_time < duration:
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                if isinstance(message, bytes):
                    filename = os.path.join(output_folder, f"chunk_{chunk_counter:03d}.wav")
                    with wave.open(filename, 'wb') as wf:
                        wf.setnchannels(channels)
                        wf.setsampwidth(2)  
                        wf.setframerate(sample_rate)
                        wf.writeframes(message)

                    print(f"Saved chunk {chunk_counter} to {filename}")
                    chunk_counter += 1
                else:
                    print(f"Received unexpected message: {message}")

            except asyncio.TimeoutError:
                continue

        await websocket.send("stop")
        print("Recording stopped.")


def main():

    parser = argparse.ArgumentParser(description="Audio Recorder Client")
    parser.add_argument(
        "mode",
        choices=["mic", "system", "mixed"],
        help="Recording mode: 'mic', 'system', or 'mixed'",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        default=10,
        help="Recording duration in seconds (default: 10)",
    )
    parser.add_argument(
        "--min_split",
        type=float,
        default=2.0,
        help="Minimum split time in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--max_split",
        type=float,
        default=5.0,
        help="Maximum split time in seconds (default: 5.0)",
    )
    parser.add_argument(
        "-sr",
        "--sample_rate",
        type=int,
        default=44100,
        help="Sample rate (default: 44100)",
    )

    parser.add_argument(
        "-c",
        "--channels",
        type=int,
        default=1,
        help="Number of channels (default: 1, mono)",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        default="recorded_chunks",
        help="Output folder for saved chunks (default: recorded_chunks)",
    )


    args = parser.parse_args()
    asyncio.run(record_and_split(args.mode, args.duration, args.min_split, args.max_split, args.sample_rate,args.channels, args.output_folder))
    print(f"Audio chunks saved to the '{args.output_folder}' folder.")



if __name__ == "__main__":
    main()

# python api_client.py system -d 30 --min_split 3 --max_split 8
# python api_client.py mic -d 15
# python api_client.py mixed -o my_mixed_audio