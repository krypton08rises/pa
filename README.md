# pa

*Continuous voice-in → LLM-out assistant. Audio capture → VAD → Whisper ASR
→ `llama-server`-hosted Gemma. Runs on a workstation GPU today; Jetson Nano
offload deliberately parked (see [Roadmap](#roadmap)).*

## Contents

- [Audio Pipeline](#audio-pipeline)
  - [Chunking & Resampling](#chunking--resampling)
  - [Channel Downmixing](#channel-downmixing)
  - [Utterance Assembly (VAD)](#utterance-assembly-vad)
  - [File-Mode Threading](#file-mode-threading)
  - [Mic Callback Backpressure — deferred](#mic-callback-backpressure--deferred)
- [LLM Orchestration](#llm-orchestration)
  - [LLM Routing](#llm-routing)
  - [Context Compaction](#context-compaction)
- [Roadmap](#roadmap)
  - [Project Split: Assistant vs. Research](#project-split-assistant-vs-research)
  - [Build Order](#build-order)
- [Environment](#environment)
  - [CUDA Build Fix — Fedora 43](#cuda-build-fix--fedora-43)
- [Status](#status)
  - [Open](#open)
  - [Resolved Code Review Findings](#resolved-code-review-findings)

---

## Audio Pipeline

Producer/consumer queue of audio chunks, handed to VAD to detect speech
before ASR ever runs.

### Chunking & Resampling
`stream.py`

Silero needs exactly `min_required_chunk_size` samples at `target_sr`
(512 @ 16k, 256 @ 8k). `source_blocksize` is derived
(`min_required_chunk_size * original_sr / target_sr`), not hardcoded, so
mic (48k → 1536) and 16k file (→ 512) both resolve through the same math.
Short tail chunks (end of file) are zero-padded up to
`min_required_chunk_size` **after** resampling — that's the length the
model actually sees.

### Channel Downmixing

- **Mono** — used as-is.
- **Stereo** — averaged across the channel axis (`np.mean(chunk, axis=1)`)
  before resampling. Flattening interleaved L/R instead of averaging
  corrupts the signal and doubles the effective sample count post-resample
  — must downmix first, not reshape.
- **5.1 surround** (formula documented, not yet implemented — see
  [Status](#status)): `((L + R) / √2) + C + ((Ls + Rs) / √2)`

### Utterance Assembly (VAD)
`vad.py`, `VadConfig` in `common.py`

State machine in `VadChunkStream.__iter__`:

1. `pre_roll_q`: `deque(maxlen=preroll_ring_size)` — every chunk pushed
   in, speech or silence, every iteration.
2. `consecutive_speech_count`: +1 on speech, resets on silence. Onset
   fires at `>= onset_confirm_chunks`: seed `utterance_buffer` from
   `pre_roll_q`, `in_utterance = True`.
3. `consecutive_silence_count`: +1 on silence, resets on speech. Endpoint
   fires at `>= speech_complete_timeout_ms / chunk_duration_ms`:
   concatenate + yield `utterance_buffer`, reset state.
4. Hard cap (`max_utterance_seconds`): force-flush regardless of silence.
   Does **not** reset `in_utterance`/`consecutive_speech_count` — only
   clears + reseeds the buffer from `pre_roll_q`, so a forced cutoff
   mid-sentence doesn't drop audio.

Tuned: `onset_confirm_chunks=10`, `speech_complete_timeout_ms=1024`,
`preroll_ring_size=10`, `max_utterance_seconds=30`. `onset_confirm_chunks`
must equal `preroll_ring_size` (otherwise the ring evicts confirming
chunks before they're dumped into the buffer) — enforced by a try/except
around the assert (`VadConfig.__init__`) that auto-corrects
`preroll_ring_size` and warns instead of crashing, since a stripped `-O`
assert would fail silently instead of fast.

Flushing (`_flush_utterance`) retries `utterance_queue.put()` on
`queue.Full` (bounded by `VadConfig.thread_timeout_seconds`), checking
`stop_event` between retries so a dead/stalled ASR consumer can't hang the
VAD thread. Only the hard-cap flush reseeds `utterance_buffer` from
`pre_roll_q` — the other two flush sites (silence endpoint, end of
stream) represent the utterance actually ending, not a forced cutoff.

### File-Mode Threading
`stream.py`

File source runs on its own producer thread (`threadable_filestream`,
daemon), mirroring the mic path (PortAudio already runs its own callback
thread). The bounded queue gives real backpressure — producer blocks on
`put()` until the consumer catches up. `stop_stream()`'s
`file_thread.join()` is bounded (`timeout=5`), and ASR has its own
`speech_q`/`asr_thread` (`WhisperTranscriber`, `asr.py`) so a slow Whisper
call doesn't stall VAD chunk draining — both were open TODOs, now
resolved.

### Mic Callback Backpressure — deferred
`stream.py`

`audio_callback` runs on PortAudio's real-time callback thread, expected
back every ~32ms. It currently does a blocking
`self.queue.put(indata.copy())` — if the VAD consumer falls behind and the
queue fills, this blocks *inside the driver's callback*, risking xruns
(dropouts) since the callback can't return on schedule. A bounded wait
doesn't fix this — any blocking in a real-time callback risks the same
xruns, just for a shorter window.

Correct pattern: never block. `put_nowait()`; on `queue.Full`, evict the
oldest queued chunk (`get_nowait()` then `put_nowait()`) rather than drop
the incoming one — the queue should represent the most recent audio,
which matters more for VAD onset detection than an unbroken backlog of
stale audio.

Direction decided (drop-oldest), **not implemented** —
`max_queue_size=1024` gives ~32s of headroom, no backpressure observed in
practice yet. Revisit once `audio_stream.queue.qsize()` (logged under
`verbose=True`) is actually seen climbing toward the cap — that's the
trigger, not a hypothetical. Even once implemented, eviction only
prevents a driver-level crash; if Whisper/Gemma is the real bottleneck,
the fix is upstream throughput (see [Context Compaction](#context-compaction),
[LLM Routing](#llm-routing)), not this queue's eviction policy.

---

## LLM Orchestration

### LLM Routing
`llm.py`, agreed 2026-08-15

Wiring ASR → `llama-server` is step 2 of the [build order](#build-order).

1. **No third consumer thread needed.** `WhisperTranscriber` already owns
   its own `asr_thread`, independent of whatever calls
   `read_transcripts()` — the main thread blocks on the LLM call while ASR
   keeps transcribing new utterances in the background. `speech_q`'s
   bound gives backpressure if the LLM ever falls far behind. Mirrors the
   VAD→ASR split without needing a literal 3rd thread.
2. **Stateful, in-session conversation** — full message history resent
   every `/v1/chat/completions` call. Confirmed cheap: llama-server's
   slot/LCP prompt-cache reuse hits `f_sim_best ~0.98` with `graphs
   reused` in the thousands, so prompt-eval stays ~13ms/token even as
   history grows. True cross-session memory (survives a restart) is a
   separate retrieval problem — deferred to the wiki/RAG step ([build
   order](#build-order) step 4).
3. **Streaming replies** (`stream: true`, SSE) over blocking — consistent
   with the latency-first framing.
4. `llm.py` is its own module — thin client wrapper around llama-server's
   OpenAI-compatible endpoint (same wire format, not an actual OpenAI
   dependency; everything stays on loopback) — mirrors how
   `vad.py`/`asr.py` are split.

### Context Compaction
`llm_utils.py`, agreed 2026-08-21

The `conversation` list grows unboundedly. `deque`/sliding-window was
considered and rejected: it breaks the exact prompt-cache reuse property
LLM Routing point 2 depends on — a sliding window evicts from the front,
so the prefix differs on every request instead of staying stable, and
cache reuse breaks completely. Compaction keeps the prefix stable
*between* compaction events instead of changing it every turn — a
corollary of why full-resend was chosen, not a competing idea.

**Structure**, four blocks in prefix order:
1. System prompt — immutable, permanent cache hit.
2. Earlier-summary block — rewritten only at a compaction event.
3. Live turns — append-only since the last compaction.
   *(Facts/memory block deferred — see below.)*

**Trigger:** 60% of usable context → compact down to 30%. Big steps, rare
— one re-prefill roughly every 20-40 turns instead of every turn. Fires
two ways: threshold, or sustained-idle `speech_q` (empty for N seconds,
not just momentarily — a single empty poll just means between
utterances). Idle-trigger keeps compaction off the latency-critical path;
threshold is the backstop if the user keeps talking.

**Token counting:** `/apply-template` (renders through Gemma's real chat
template) + `/tokenize` for ground truth — `chars/4` undercounts,
template overhead isn't negligible. Costs a round trip, so keep a running
per-turn estimate and only hit the real endpoint periodically (~every 20
turns) to recalibrate.

**Anchored summaries, not recursive:** keep the last *k* compaction
summaries as separate immutable blocks, only merge the oldest two once
out of room. Recursively re-summarizing the summary drifts/degrades —
the main failure mode of naive rolling-summary designs, avoided by design
here rather than caught later.

**Explicitly deferred** (step 4/5 territory — [build order](#build-order)):
persistent fact store (SQLite+FTS5, reusing the wiki pattern), retrieval
over past sessions, model-writable memory via `remember`/`forget` tools,
hierarchical/graph-structured compaction. All real ideas, all premature
ahead of wiki (step 4) and tools (step 5).

**Also decided:** no `--ctx-shift` (llama.cpp's StreamingLLM-style
keep-first-N-slide-rest) — server-level version of the same mistake as
`deque` above. `--cache-type-k q8_0 --cache-type-v q8_0` (roughly halves
KV vs f16, V-quant needs flash attention on) and explicit `--n-predict`
headroom are cheap, independent wins worth doing regardless of
compaction, given the tight VRAM budget.

---

## Roadmap

### Project Split: Assistant vs. Research
agreed 2026-08-15

One 20GB/70W card can't serve an always-on latency-sensitive assistant
*and* run hackable GPU research at once — split into two projects rather
than force one stack to do both.

#### Project A — the assistant
this repo, `feat/orchestrate-llm`

Always-on, batch-1, latency-sensitive, wants stability. Serving runtime
is `llama.cpp`/`llama-server` — GGML's kernels are already hand-written
CUDA/C++, no reason to write inference by hand here. The barebones/by-hand
instinct is redirected to the orchestrator (thin Python loop, tool
definitions, routing), not the inference engine.

**Latency budget** (voice-in, ~30B model, this card):

| Stage | Time |
|---|---|
| ASR | ~0.3s |
| Prefill (sys + wiki + tools) | 2–15s ← the killer |
| Decode 200 tok @ 18 tok/s | ~11s |
| TTS first chunk | ~0.3s |

**Mitigations, in order of impact:**
1. Prompt cache reuse (`llama-server --cache-reuse` + persistent slots) —
   system prompt + wiki context prefilled once, stays resident. Worth
   more than any model choice.
2. Speculative decoding (`--model-draft` + `--draft-max 16`) — why
   DFlash/Gemma 4's MTP drafter matter on a bandwidth-starved card.
3. Two-tier routing — small model (Gemma 4 E4B) for conversational turns
   and tool-arg filling, 26B/30B only when warranted. Most turns are
   "what's on my calendar," not SWE-Bench.
4. Stream TTS from the first sentence instead of waiting for generation
   to finish (Kokoro/Piper, small enough to run alongside).

ASR→LLM prefill overlap (start prefilling the growing utterance text as
it's transcribed, instead of waiting for the VAD endpoint) is the same
mechanism as (1), applied incrementally — a refinement once cache-reuse
is working, not a step-1 concern.

**Wiki:** markdown files + SQLite FTS5 + a grep-shaped retrieval tool,
model does its own retrieval across a 128–256K window. Not a vector DB
first — FTS5 fails legibly, a vector index fails silently. Add embeddings
(EmbeddingGemma-308M, ~300MB) only once there's a logged set of retrieval
failures FTS5 provably can't handle.

**Tools:** MCP servers (search/scrape/wiki) + a thin Python orchestrator.
Not LangChain — the value is in the tool definitions; Qwen3.6 and Glimmer
are already trained against agentic/schema-based function calling.
Scraper output is a real injection surface (AgentDojo threat model —
Glimmer's own card shows 28% attack success on the *better* model) — keep
it behind a boundary: separate context, no tool access on that turn.

AWS Athena over Common Crawl's `cc-index` (S3+Parquet) as a first lookup
before live-scraping — avoids the live network hop/SSRF-shaped surface if
a page's already crawled. Doesn't remove injection risk from page
*content* — same isolation boundary still applies. Live scraping stays as
fallback.

Containerized (Docker/Podman) as the rest of Project A gets built, not
bolted on at the end.

**Model shortlist**, ranked for this card (RTX 4000 SFF, 20GB):
1. **Gemma 4 26B-A4B @ Q4 + MTP** — best speed/quality/fit balance.
   Starting model.
2. Muse Glimmer-30B @ community Q3/IQ4 + DFlash — best agentic
   orchestration + injection resistance, but below Meta's validated quant
   floor — a gamble on the exact property it's picked for.
3. Qwen3.6-35B-A3B @ Q4 with `--n-cpu-moe` — strongest on
   terminal/coding harnesses.
4. Qwen3.6-27B @ IQ4_XS — dense, slowest prefill, hardest to justify;
   only non-MoE option here.

#### Project B — the research
deliberately separate track, not this repo

GRPO, Triton kernels, quantization work — wants the GPU free, a hackable
stack, tolerates breakage. Runtime is PyTorch + Triton (kernels testable
standalone against a PyTorch reference), not llama.cpp — a fused Triton
kernel here can't ship into llama.cpp without porting to CUDA. Runs on
rented H100 hours, or 4–8B locally when the card is free.

**GRPO ceiling on this card** (QLoRA — base weights + adapters +
gradients + optimizer state + rollout KV + reference logprobs, all at
once):

| Size | Verdict |
|---|---|
| 4B | Comfortable, real experiments |
| 8B | Tight but workable (Unsloth: gradient checkpointing, adapter-disable as reference model) |
| 14B+ | Rent |

Better target than "another point on AIME": RL the **router** and the
**drafter**, not the main model. A 1–4B router (small-vs-large, which
tools) is small enough to train locally, has a loggable reward signal
from real usage, and moves felt quality more than the main model would.
Same logic for fine-tuning a drafter on your own conversation
distribution — higher acceptance rate is a direct tok/s win on a
bandwidth-bound card.

Revisit vLLM only if a Project B kernel proves out and is wanted in
production — migrate serving deliberately, not by accident. Vanilla
baseline first, optimize with numbers.

### Build Order

1. `llama-server` + Gemma 4 26B-A4B + OpenAI-compatible endpoint. Nothing
   else. Prove it bare-metal first, containerize once model/quant/flags
   are known.
2. Wire ASR → endpoint → TTS. Measure end-to-end latency. Fix that
   before adding features.
3. Add prompt cache reuse + a drafter. Re-measure. (ASR/LLM prefill
   overlap belongs here too.)
4. Add wiki as FTS5 + tool. Log every retrieval.
5. Add MCP tools one at a time, scraper isolated.
6. Only then start Project B, deliberately — separate machine/hours.

---

## Environment

**Python:** pyenv 3.13.1 (`.python-version`). System Fedora Python (3.14)
deliberately untouched.

### CUDA Build Fix — Fedora 43
2026-08-15

CUDA 12.9's `crt/math_functions.h` declares `rsqrt`/`rsqrtf`/`sinpi`/
`sinpif`/`cospi`/`cospif` without `noexcept`, conflicting with Fedora 43's
newer glibc, which declares the same GNU-extension functions *with*
`noexcept(true)` — hard compile error when nvcc's host compiler parses
both. Known upstream issue (`ggml-org/llama.cpp#19100`, NVIDIA forums).

1. System `gcc`/`g++` is 15 (Fedora 43 default) — too new for CUDA 12.9's
   host-compiler ceiling of 14. Installed `gcc14`/`gcc14-c++` as a
   **side** compiler (system default untouched, since Fedora 43 itself is
   built against 15), pointed only `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14`
   at it.
2. Patched `/usr/local/cuda-12.9/targets/x86_64-linux/include/crt/math_functions.h`
   in place (backed up `.bak`), adding `noexcept (true)` to the 6
   conflicting declarations. **Will need reapplying if CUDA is
   reinstalled/upgraded.**

---

## Status

### Open

- **Mic callback backpressure** — see
  [above](#mic-callback-backpressure--deferred), direction decided, not
  implemented.
- **Context compaction** — see [above](#context-compaction), design
  agreed, not implemented (`llm_utils.py`, in progress).
- **Lower-priority cleanup, still open:** `utils.py`'s
  `stereo_to_mono_for_vad`/`cast_sampling` are dead code, assume a shape
  incompatible with the 2D downmix `stream.py` actually uses; unused
  fields (`RecordingConfig.device_name`/`chunk_size`, `vad.py`'s
  `noise_flag`/`held_chunk`/`last_chunk_timestamp`); the 5.1 downmix
  formula (see [Channel Downmixing](#channel-downmixing)) isn't actually
  implemented — `stream.py` only handles the >1-channel case via
  unweighted `np.mean`.

### Resolved Code Review Findings

From `/code-review` on `feat/threading-prod-cons`, parked 2026-08-12, all
resolved 2026-08-16 (see `_flush_utterance` in `vad.py`, the `loggy`
import + assert try/except in `common.py`, the removed dead block in
`asr.py`, `.gitignore`, `pyproject.toml`). Kept as historical record:

1. `asr.py` — unreachable "process final chunk" block (`chunk` forced
   `None` on every exit path).
2. `vad.py` — hard-cap flush duplicated ~320ms of audio at every 30s
   cutoff (reseeded from `pre_roll_q` after already flushing it).
3. `pyproject.toml` — `librosa`, `soundfile`, `loggy` imported but
   missing from `[project.dependencies]`; fresh install would
   `ModuleNotFoundError`.
4. `vad.py` — `utterance_queue.put()` blocked indefinitely, never checked
   `stop_event` — could hang past `vad_thread.join(timeout=5)`.
5. `stream.py` — `audio_callback` blocking `Queue.put()` on the real-time
   thread, no drop/timeout fallback.
6. `vad.py` — flat `time.sleep(5)` after an empty poll, leftover debug
   code.
7. `vad.py` — silence chunks inside an ongoing utterance never appended
   to `utterance_buffer` — mid-sentence pauses jump-cut instead of
   preserved.
8. `.gitignore` — rewrite had dropped `.env`, `.venv`/`venv`, `build/`,
   `dist/`, `.pytest_cache/`.
9. `common.py` — `onset_confirm_chunks == preroll_ring_size` invariant
   was a bare `assert`, stripped under `-O`.
10. `vad.py` — `VadChunkStream.__init__`'s `config: VadConfig =
    VadConfig()` mutable default, shared across instances that don't pass
    their own.
