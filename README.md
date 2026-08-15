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
cache reuse, speculative decoding, two-tier routing - see "LLM ROUTING"
below), not this queue's eviction policy.

-- TODO

Both prior items resolved: `stop_stream()`'s `file_thread.join()` is
bounded via `timeout=5`, and ASR now has its own `speech_q` + `asr_thread`
(`WhisperTranscriber`, `asr.py`) so a slow Whisper call doesn't stall VAD
chunk draining. Nothing currently open here.

-- CODE REVIEW FINDINGS (parked 2026-08-12, from /code-review on
   feat/threading-prod-cons before merge to development)

STATUS (2026-08-16): all 10 numbered findings below resolved this
session - see `_flush_utterance` (`vad.py`), the `loggy` import + assert
try/except (`common.py`), the removed dead block (`asr.py`), `.gitignore`,
and `pyproject.toml`. The "lower priority" items in the closing paragraph
(dead code in `utils.py`, unused fields, the unimplemented 5.1 formula)
are NOT addressed - still open. List kept as historical record of what
was found and why.

1. `asr.py:60` - `if chunk is not None:` after the try/except in `main()` is
   unreachable. `chunk` is forced `None` on both `except` branches and is
   also `None` whenever the while loop exits normally - the "process final
   chunk" block never runs.
2. `vad.py:201` - hard-cap flush (`max_utterance_seconds_reached()`)
   reseeds `utterance_buffer` from `pre_roll_q`, which still holds the same
   trailing chunks just flushed - duplicates ~320ms of audio (repeated
   words) at every 30s cutoff on a long continuous utterance.
3. `pyproject.toml:9` - `librosa`, `soundfile`, `loggy` are imported
   (`common.py`, `stream.py`, all three modules respectively) but missing
   from `[project.dependencies]`. Fresh install on the actual Nano will
   `ModuleNotFoundError` on first import.
4. `vad.py:187` - `utterance_queue.put()` blocks indefinitely on the
   bounded queue (maxsize=10) and never checks `stop_event` while blocked -
   VAD thread can hang past `vad_thread.join(timeout=5)` in `asr.py` if the
   ASR consumer has fallen behind or errored out.
5. `stream.py:55` - `audio_callback` does a blocking `Queue.put()` from
   inside the real-time PortAudio callback thread, no drop/timeout
   fallback - risks input overflow/xruns if the consumer falls behind.
6. `vad.py:93` - flat `time.sleep(5)` after an empty poll adds up to 5s of
   latency to utterance delivery; own inline comment marks it as leftover
   debug code ("comment this out later").
7. `vad.py:165` - silence chunks inside an ongoing utterance (below
   endpoint timeout) are never appended to `utterance_buffer`, only speech
   chunks are - mid-sentence pauses get jump-cut instead of preserved,
   clipping adjacent word boundaries.
8. `.gitignore` - rewritten version dropped `.env`, `.venv`/`venv`,
   `build/`, `dist/`, `.pytest_cache/` and other prior entries; only
   `.claude/`, `**/data/`, `__pycache__/`, `*.py[cod]`, `*.egg-info/`
   remain covered now.
9. `common.py:230` - `onset_confirm_chunks == preroll_ring_size` invariant
   is a bare `assert`, stripped under `-O`/`PYTHONOPTIMIZE` - would fail
   silently instead of fast on a future config change.
10. `vad.py:30` - `VadChunkStream.__init__`'s `config: VadConfig =
    VadConfig()` is a mutable default, one instance shared by every
    `VadChunkStream` that doesn't pass its own `config=`. Harmless today
    (nothing mutates it post-init) but a latent trap for runtime tuning
    later.

Also flagged, lower priority: `utils.py`'s `stereo_to_mono_for_vad`/
`cast_sampling` are dead code and assume a shape incompatible with the 2D
downmix `stream.py` actually uses; unused fields (`RecordingConfig.
device_name`/`chunk_size`, `vad.py`'s `noise_flag`/`held_chunk`/
`last_chunk_timestamp`); the 5.1 surround downmix formula documented above
isn't actually implemented (`stream.py` only handles the >1-channel case
via `np.mean`, no per-channel weighting).

-- PROJECT SPLIT: ASSISTANT vs RESEARCH (agreed 2026-08-15)

One 20GB/70W card can't serve an always-on latency-sensitive assistant and
run hackable GPU research at once - split into two projects rather than
force one stack to do both.

PROJECT A - THE ASSISTANT (this repo continues here, `feat/orchestrate-llm`)

Always-on, batch-1, latency-sensitive, wants stability. Serving runtime is
`llama.cpp`/`llama-server` - GGML's kernels are already hand-written
CUDA/C++, no Triton in this path, no reason to write inference by hand
here. The barebones/by-hand instinct is redirected to the orchestrator
(thin Python loop, tool definitions, routing) instead of the inference
engine itself.

Latency budget (voice-in, ~30B model, this card):
  ASR                            ~0.3s
  prefill (sys + wiki + tools)   2-15s   <- the killer
  decode 200 tok @ 18 tok/s      ~11s
  TTS first chunk                ~0.3s

Mitigations, in order of impact:
1. Prompt cache reuse (`llama-server --cache-reuse` + persistent slots) -
   system prompt + wiki context prefilled once, stays resident. Worth more
   than any model choice.
2. Speculative decoding (`--model-draft` + `--draft-max 16`) - the reason
   DFlash/Gemma 4's MTP drafter matter on a bandwidth-starved card.
3. Two-tier routing - small model (Gemma 4 E4B) for conversational turns
   and tool-arg filling, the 26B/30B only when the task actually warrants
   it. Most turns are "what's on my calendar," not SWE-Bench.
4. Stream TTS from the first sentence instead of waiting for generation to
   finish (Kokoro/Piper, small enough to run alongside).

ASR->LLM prefill overlap (start prefilling the growing user-utterance text
as it's transcribed, instead of waiting for the VAD endpoint) is the SAME
mechanism as (1) above, applied incrementally - a refinement layered on
top of cache-reuse once that's working, not a step-1 concern. The static
system-prompt block dwarfs a single utterance in token count, so that's
where the real win is.

Wiki: markdown files + SQLite FTS5 + a grep-shaped retrieval tool, model
does its own retrieval across a 128-256K window. NOT a vector DB first -
FTS5 fails legibly (you can see why a miss happened), a vector index fails
silently. Add embeddings (EmbeddingGemma-308M, ~300MB) only once there's a
logged set of retrieval failures FTS5 provably can't handle - same
gold-set-first discipline as the judge work.

Tools: MCP servers (search/scrape/wiki) + a thin Python orchestrator
holding the loop. Not LangChain - the value is in the tool definitions;
both Qwen3.6 and Glimmer are already trained against agentic/schema-based
function calling. Scraper output is a real injection surface (AgentDojo
threat model - Glimmer's own card shows 28% attack success on the BETTER
model) - keep it behind a boundary: separate context, no tool access on
that turn.

Look into AWS Athena over Common Crawl's public columnar index (cc-index,
S3 + Parquet) as a first lookup before live-scraping a target URL - if a
page is already in a recent crawl, querying the cached copy avoids the
live network request entirely (cheaper, faster, no SSRF-shaped surface on
whatever URL the model picks). Doesn't remove the injection risk from page
*content* itself - cached text can still carry a prompt injection - so the
same boundary/isolation rule above still applies regardless of source.
Live scraping still needed as a fallback for anything not yet crawled.

Containerized (Docker/Podman) for reproducibility and future hosting -
wired in as the rest of Project A gets built, not bolted on at the end.

Model shortlist, ranked for this card (RTX 4000 SFF, 20GB):
1. Gemma 4 26B-A4B @ Q4 + MTP - best speed/quality/fit balance. STARTING
   MODEL.
2. Muse Glimmer-30B @ community Q3/IQ4 + DFlash - best agentic
   orchestration + injection resistance, but below Meta's validated quant
   floor - a real gamble specifically on the property it's picked for,
   worth extra caution given the scraper/injection surface above.
3. Qwen3.6-35B-A3B @ Q4 with `--n-cpu-moe` - strongest on
   terminal/coding harnesses.
4. Qwen3.6-27B @ IQ4_XS - dense, slowest prefill, hardest to justify -
   also the only non-MoE option here, no expert-routing to reason about.

Build order:
1. `llama-server` + Gemma 4 26B-A4B + OpenAI-compatible endpoint. Nothing
   else. Prove it bare-metal first, containerize once the working
   model/quant/flags are known - don't debug GPU passthrough and
   model/quant choice at the same time.
2. Wire ASR -> endpoint -> TTS. Measure end-to-end latency. Fix that
   before adding features.
3. Add prompt cache reuse + a drafter. Re-measure. (ASR/LLM prefill
   overlap belongs here too, once the basics work.)
4. Add wiki as FTS5 + tool. Log every retrieval.
5. Add MCP tools one at a time, scraper isolated.
6. Only then start Project B, deliberately - separate machine/hours.

-- LLM ROUTING (`llm.py`, agreed 2026-08-15)

Wiring ASR -> `llama-server` is step 2 of the Project A build order above.
Decisions made:

1. Third consumer thread + queue (mirrors the VAD->ASR split, same
   reasoning): the LLM call is a slow, bursty stage like Whisper was, so it
   gets its own thread reading a new queue of completed utterance text,
   instead of blocking the ASR consumer. Without this, Whisper can't start
   transcribing utterance N+1 until Gemma finishes replying to utterance N
   - `utterance_queue` would back up for no reason since VAD keeps running
   regardless.
2. Stateful, in-session conversation: full message history resent each
   `/v1/chat/completions` call. Confirmed cheap in practice - manual
   testing showed llama-server's slot/LCP prompt-cache reuse hitting
   `f_sim_best ~0.98` with `graphs reused` climbing into the thousands, so
   prompt-eval stays ~13ms/token even as history grows. True cross-session
   persistent memory (survives a restart) is a separate retrieval problem -
   deferred to the wiki/RAG step (build order step 4), not part of the
   basic loop.
3. Streaming replies (`stream: true`, SSE) preferred over blocking,
   consistent with the latency-first framing above.
4. `llm.py` is its own module (client wrapper around llama-server's
   OpenAI-compatible endpoint - same wire format as OpenAI's API, not an
   actual OpenAI dependency, everything stays on loopback), mirroring how
   `vad.py`/`asr.py` are already split.

-- BUILD ENV: llama-server CUDA build on Fedora 43 (2026-08-15)

CUDA 12.9's `crt/math_functions.h` declares `rsqrt`/`rsqrtf`/`sinpi`/
`sinpif`/`cospi`/`cospif` without `noexcept`, conflicting with Fedora 43's
newer glibc which declares the same GNU-extension functions WITH
`noexcept(true)` - hard compile error ("exception specification is
incompatible") when nvcc's host compiler parses both. Known upstream issue
(ggml-org/llama.cpp#19100, NVIDIA forums). Two-part fix applied:

1. System `gcc`/`g++` is 15 (Fedora 43 default) - too new for CUDA 12.9's
   supported host-compiler ceiling of 14. Installed `gcc14`/`gcc14-c++` as
   a SIDE compiler (system default left untouched, since Fedora 43 itself
   is built against 15) and pointed only
   `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14` at it.
2. Patched `/usr/local/cuda-12.9/targets/x86_64-linux/include/crt/
   math_functions.h` in place (backed up as `.bak`), adding
   `noexcept (true)` to the 6 conflicting declarations to match glibc.
   Will need reapplying if the CUDA toolkit is reinstalled/upgraded.

PROJECT B - THE RESEARCH (deliberately separate track, not this repo)

GRPO, Triton kernels, quantization work - wants the GPU free, a hackable
stack, tolerates breakage. Runtime is PyTorch + Triton (kernels testable
standalone against a PyTorch reference), not llama.cpp - a fused Triton
kernel written here can't ship into llama.cpp without porting to CUDA.
Runs on rented H100 hours, or 4-8B locally when this card is free.

GRPO ceiling on this card (QLoRA - base weights + adapters + gradients +
optimizer state + rollout KV + reference logprobs, all at once):
  4B   - comfortable, real experiments
  8B   - tight but workable (Unsloth: gradient checkpointing,
         adapter-disable as reference model)
  14B+ - rent

Better target than "another point on AIME": RL the ROUTER and the
DRAFTER, not the main model. A 1-4B router (small-vs-large, which tools)
is small enough to train locally, has a reward signal loggable from real
usage, and moves the assistant's felt quality more than the main model
would. Same logic for fine-tuning a drafter on your own conversation
distribution - higher acceptance rate is a direct tok/s win on a
bandwidth-bound card.

Revisit vLLM only if a Project B kernel proves out and is wanted in
production - migrate serving deliberately at that point, not by accident.
Vanilla baseline first, optimize with numbers - same discipline as the
ASR/Whisper work above.
