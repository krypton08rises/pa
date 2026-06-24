from silero_vad import load_silero_vad, read_audio, VadIterator


from stream import AudioStream

"""
Using a drio

"""

def main(
    onnx:bool = False, 
):
    model = load_silero_vad(onnx=onnx)