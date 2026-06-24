import numpy as np
import librosa



def stereo_to_mono_for_vad(interleaved_buffer):
    # Initialize the mono buffer
    mono_buffer = []
    
    # Iterate through the buffer step by 2 (Left, Right)
    for i in range(0, len(interleaved_buffer), 2):
        left_sample = interleaved_buffer[i]
        right_sample = interleaved_buffer[i+1]
        
        # Simple arithmetic average to prevent clipping
        mono_sample = (left_sample + right_sample) / 2.0
        mono_buffer.append(mono_sample)
        
    return mono_buffer
    

def cast_sampling(waveform, target_sample_rate):
    
    pass
    