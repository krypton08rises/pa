import sounddevice as sd 
import soundfile as sf
from jetson_nano_asr.common import RecordingConfig
from loggy import logger

import threading 
from queue import Queue 


# Implement a queue to hold audio chunks and a callback function to read audio data from the stream and put it into the queue.
class AudioStream:
    def __init__(self, config:RecordingConfig):
        self.queue = Queue()
        self.stream = None
        self.config = config
        self.counter = 0 # Chunks seen so far
        self.use_file = config.file_path is not None
        self.event = threading.Event()       
        sd.default.device = self.config.device



    def audio_callback(self, indata, frames, time, status):
        if status:
            logger.debug("Captured audio chunk with {frames} frames at {t:.2f}s", frames=frames, t=time.inputBufferAdcTime)

        self.queue.put(indata.copy())


    def start_stream(self):
        if self.use_file:
            with sf.SoundFile(self.config.file_path) as file_stream:
                logger.info("Streaming audio from file: {fp}", fp=self.config.file_path)
                while True:
                    data = file_stream.read(self.config.chunk_size, dtype='float32')
                    if not data.size:
                        break
                    logger.info("Data size: {ds} | Total chunks: {qsize}", ds=data.size, qsize=self.queue.qsize())
                    self.queue.put(data)

                logger.info("Finished streaming audio from file | Total chunks: {chunks}", chunks=self.queue.qsize())
        else:
            self.stream = sd.InputStream(
                device=self.config.device,
                channels=self.config.channels,
                samplerate=self.config.sample_rate,
                blocksize=self.config.chunk_size,
                dtype='float32',
                callback=self.audio_callback
            )
            self.stream.start()
    
    def stop_stream(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def read(self):
        # return self.queue.get()
        return self.queue.get().reshape(-1, ) if self.queue.qsize() > 0 else None 
        
    def __enter__(self):
        self.start_stream()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_stream()


# Example usage:
if __name__ == "__main__":
    
    import argparse 

    parser = argparse.ArgumentParser(description="Audio Stream Example")
    parser.add_argument("--test-file", type=bool, default=False, help="Use a test audio file instead of live recording")    
    args = parser.parse_args()

    config = RecordingConfig()
    with AudioStream(config) as audio_stream:
        print("Recording audio...")
        for _ in range(5):  # Read 5 chunks of audio data
            audio_chunk = audio_stream.read(config.chunk_size)
            print(f"Received audio chunk of shape: {audio_chunk.shape}")

