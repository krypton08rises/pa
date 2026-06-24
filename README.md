# pa

1. Producer Consumer queue for audio chunk.
2. This chunk is passed down to the VAD to detect speech.

-- CHUNKING (`stream.py`)

Silero needs exactly `min_required_chunk_size` samples at target_sr (512 @
16k, 256 @ 8k). Read enough samples at the SOURCE rate so that after
resampling you land on exactly that - `source_blocksize` is derived from
`min_required_chunk_size * original_sr / target_sr`, not hardcoded, so mic
(48k -> 1536) and 16k file (-> 512) both work through the same math. Short
tail chunks (end of file) are zero-padded up to `min_required_chunk_size`
AFTER resampling, since that's the length the model actually sees.

-- CHANNELS

1. MONO - used as-is.
2. STEREO - averaged across the channel axis (`np.mean(_chunk, axis=1)`)
   before resampling, to get mono. Flattening interleaved L/R samples
   instead of averaging corrupts the signal and doubles the effective
   sample count after resampling - must downmix first, not reshape.
3. Surround (5.1) -
    L / R: Front Left and Front Right channels.
    C: Center channel (contains the main dialogue).
    Ls / Rs: Left Surround and Right Surround channels.
    ((L + R) / sqrt(2)) +C +((Ls + Rs) / sqrt(2))

-- UTTERANCE ASSEMBLER (`vad.py`, `VadConfig` in `common.py`)

VAD assembles raw chunks into utterances before handing them to Whisper.
State machine lives in `VadChunkStream.__iter__`:

1. `pre_roll_q`: `deque(maxlen=preroll_ring_size)` - every chunk pushed in,
   speech or silence, every iteration.
2. `consecutive_speech_count`: +1 on speech, resets on silence. Onset fires
   at `>= onset_confirm_chunks`: seed `utterance_buffer` from `pre_roll_q`,
   `in_utterance = True`.
3. `consecutive_silence_count`: +1 on silence, resets on speech. Endpoint
   fires at `>= speech_complete_timeout_ms / chunk_duration_ms`: concatenate
   + yield `utterance_buffer`, reset state.
4. Hard cap (`max_utterance_seconds`): force-flush regardless of silence.
   Does NOT reset `in_utterance`/`consecutive_speech_count` - only
   clears+reseeds the buffer from `pre_roll_q`, so a forced cutoff
   mid-sentence doesn't drop audio.

Tuned values: `onset_confirm_chunks=10`, `speech_complete_timeout_ms=1024`,
`preroll_ring_size=10`, `max_utterance_seconds=30`. `onset_confirm_chunks`
must equal `preroll_ring_size` so the ring can't evict the confirming
chunks before they're dumped into the buffer - enforced by a try/except
around the assert (`VadConfig.__init__`, `common.py`) that logs a warning
and auto-corrects `preroll_ring_size` to match, rather than crashing, since
a stripped `-O` assert would otherwise fail silently instead of fast.

Flushing (`_flush_utterance`, `vad.py`) retries `utterance_queue.put()` on
`queue.Full` (bounded by `VadConfig.thread_timeout_seconds`) instead of
blocking forever, checking `stop_event` between retries so a dead/stalled
ASR consumer can't hang the VAD thread. Only the hard-cap flush reseeds
`utterance_buffer` from `pre_roll_q` after clearing - the other two flush
sites (silence endpoint, end of stream) don't, since those represent the
utterance actually ending rather than a forced mid-sentence cutoff.

-- FILE-MODE THREADING (`stream.py`)

File source runs on its own producer thread (`threadable_filestream`,
daemon), mirroring the mic path (PortAudio already runs its own callback
thread). The bounded queue gives real backpressure - producer blocks on
`put()` until the consumer catches up.

-- MIC CALLBACK BACKPRESSURE (`stream.py`, deferred)

`audio_callback` runs on PortAudio's real-time callback thread, which the
driver expects back promptly every ~32ms. It currently does a blocking
`self.queue.put(indata.copy())` with no timeout - if the VAD consumer falls
behind and the bounded queue fills, this blocks *inside the driver's
callback*, risking input overflow/xruns (audio dropouts) since the callback
can't return on schedule.

A manual bounded wait before `put()` doesn't fix this - any blocking in a
real-time callback, even capped, still risks the same xruns, just for a
shorter window instead of unbounded. The correct pattern for real-time
audio callbacks is to never block: `put_nowait()` and on `queue.Full`, drop
a chunk and log/count it rather than wait, evicting the oldest queued
chunk first (`get_nowait()` then `put_nowait()`) rather than dropping the
incoming one - keeps the queue representing the most recent audio, which
matters more for VAD onset detection than an unbroken backlog of stale
audio does.

Decided the *direction* (drop-oldest), deliberately NOT implemented yet -
`max_queue_size=1024` gives ~32s of headroom before `put()` would ever
actually block, and no backpressure from Whisper/Gemma has been observed
in practice. Revisit once `audio_stream.queue.qsize()` (already logged
under `verbose=True` in `_threadable_vad`) is actually seen climbing
toward the cap - that's the trigger, not a hypothetical. Note even once
implemented, eviction only prevents a driver-level crash; if Whisper/Gemma
is the real bottleneck, the actual fix is upstream throughput (prompt
cache reuse, speculative decoding, two-tier routing - see the LLM
orchestration work on `feat/orchestrate-llm`), not this queue's eviction
policy.

-- TODO
1. Real `utterance_queue` + separate ASR thread, so a slow Whisper call
   never stalls VAD chunk draining. (`stop_stream()`'s `file_thread.join()`
   is already bounded via `timeout=5`.)
