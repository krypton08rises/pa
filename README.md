# pa

1. Producer Consumer queue for audio chunk.
2. This chunk is passed down to the VAD to detect speech.


-- CHANNELS
1. MONO
2. STEREO - average to get mono
3. Surround (5.1) -
    L / R: Front Left and Front Right channels.
    C: Center channel (contains the main dialogue).
    Ls / Rs: Left Surround and Right Surround channels.
    ((L + R) / sqrt(2)) +C +((Ls + Rs) / sqrt(2))
