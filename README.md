# hermes-ctxfold

Deterministic, zero-LLM context compaction for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

Hermes's built-in `ContextCompressor` pays for compaction with a **second model
call**: at the boundary it ships the whole middle of the transcript to
`call_llm(task="compression")` and asks for a structured summary. Cost and
latency therefore scale with how much context you just built up — and
`micro-compaction`, when enabled, adds one such call per turn. That is the
"compaction is slow" experience.

This plugin replaces that pass with a **compiled C++ engine**: token counting,
anchor lifting, folding, gisting and budget allocation all run locally in a
worker process over a length-framed protocol. No network, no second model,
no summary prompt.

```
built-in:  transcript ──► huge LLM call ──► narrative summary   (seconds, $$)
ctxfold:   transcript ──► C++ worker     ──► folded/gisted list  (~0.2 s, $0)
```

## Numbers

2 M characters (~500 K tokens) of tool-heavy transcript, laptop CPU:

| | |
|---|---|
| full plan (in-process benchmarks) | ~210 ms |
| end to end through the plugin (subprocess + framing) | ~250 ms |
| deterministic | yes — same input, same output, every run |
| model calls | 0 |

## What it actually does

* **Keeps everything.** Messages are kept, folded, gisted or tombstoned —
  never deleted. Tool-call/tool-result pairs stay intact, so the host never
  has to repair the list.
* **Protects what matters.** The system prompt and the newest `recent`
  messages are emitted verbatim.
* **Anchor layer.** Paths, URLs, hashes, commands, dotted config keys,
  numbers+units and backticked code are lifted verbatim into a system-level
  `ANCHORS` block before the transcript. Facts from human/assistant turns win
  the budget; tool-output noise comes last.
* **Human turns are never blanked.** Old user messages degrade to a gist at
  worst — prose intent has no anchor substitute.
* **Fold, don't summarize.** A long tool result becomes its head/tail lines
  plus a `[…ctxfold: 折叠 N 行 (sha256:…)…]` marker, so the cut is verifiable.
* **Per-turn pruning** (`prune_tool_results_only`) truncates old tool payloads
  without touching the worker, so a long session stops re-sending megabytes
  long before compaction fires.
* **Fail-open.** Missing or crashed worker → cheap truncation pass, warning in
  the log, conversation continues. A broken plugin never blocks a turn.

## Install

```powershell
.\install.ps1          # copies the plugin into $HERMES_HOME/plugins/ctxfold
.\build.ps1            # only needed if bin/ctxfold.exe is missing
```

Then in `$HERMES_HOME/config.yaml`:

```yaml
context:
  engine: "ctxfold"
```

Hermes never auto-activates plugin engines — the config line above is the
switch. `hermes plugins` → *Context Engine* should now list `ctxfold`.

Windows users can use the committed `bin/ctxfold.exe`. On other platforms run
`./build.ps1` (needs `g++` or `cl.exe`).

## Knobs (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `CTXFOLD_THRESHOLD` | `0.70` | share of the context window that triggers compaction |
| `CTXFOLD_TARGET_RATIO` | `0.25` | post-compaction size, as a share of the trigger |
| `CTXFOLD_RECENT` | `20` | newest messages kept verbatim |
| `CTXFOLD_TOOL_FOLD` | `400` | fold a tool result longer than this (tokens) |
| `CTXFOLD_GIST_LEN` | `120` | gist length (code points) |
| `CTXFOLD_TIMEOUT` | `15` | worker timeout, seconds |
| `CTXFOLD_WORKER` | auto | explicit path to the worker binary |

## Layout

```
src/                       C++17 core, zero dependencies
  core.h/.cpp              tok, anchors, fold, gist, distill, entryId, recall
  compress.h/.cpp          two-pass budget allocation over a message list
  main.cpp                 worker protocol (length-framed stdin/stdout)
  sha256.h/.cpp            FIPS 180-4 SHA-256 (shared with the other ports)
plugins/context_engine/ctxfold/
  __init__.py              Hermes entry point: register(ctx)
  engine.py                ContextEngine subclass + fail-open policy
  worker.py                protocol client (bytes in, plans out — no algorithm)
  plugin.yaml              manifest
  bin/ctxfold.exe          prebuilt worker (Windows)
tests/
  test_worker.py           worker contract + JS golden parity
  test_engine.py           real Hermes seam: discovery, contract, fail-open
  bench.py                 latency probe for a 2 M-character transcript
```

The Python side is deliberately thin: it maps provider messages onto
`(kind, text)` records and turns the reply into a new message list. No
tokenization, no folding, no budget math runs in Python.

## Protocol (one line per record, byte-length framed)

```
CTXFOLD/1 compress <budget> <toolFold> <head> <tail> <recent> <gistLen> <n>
  <kind>\t<bytes>\n<content>            × n
→ CTXFOLD/1 OK <kept> <tomb> <gisted> <folded> <inTok> <outTok> <anchors>
  keep | tomb | rewrite\t<bytes>\n<content>   × n
  A\t<bytes>\n<anchor>                  × anchors
```

`distill`, `id` and `ping` use the same framing. Message bodies are never
re-serialized by the C++ side, so provider JSON round-trips untouched.

## Determinism and cross-engine parity

The C++ core is a port of the JS reference engine (`H:/ctxfold/ctxfold.js`),
which the Torre di Babele port (`src/ctxfold.cpp`) also uses. `tok`,
`extractAnchors`, `fold`, `gist`, `distillText` and `entryId` are checked
against the JS engine on golden input (`tests/test_worker.py`, skipped when
`node`/the JS checkout is unavailable). Byte-determinism is what lets the same
memory library be shared by every host.

Two deliberate differences from the JS engine, both documented in the code:

1. **No template mining.** The JS `templatize` collapsed every surviving tool
   line to a bare `⟨§id⟩`, destroying readable output.
2. **Tool scanning is bounded** to the first 1 KB of a tool result, so a
   `std::regex` pass over megabytes of JSON cannot dominate compaction.

## Tests

```powershell
python -m unittest discover -s tests -v
```

`test_engine.py` needs a Hermes checkout (`HERMES_AGENT_SRC`, default
`H:/hermes-agent`) and loads the plugin through Hermes's real discovery path.
`tests/bench.py` prints latency for a synthetic 2 M-character transcript.

## Roadmap

* **Memory recall** — the other half of the story. Hermes's built-in memory is
  `MEMORY.md`/`USER.md` plus per-turn `prefetch()`; the ctxfold memory library
  (Markdown authoritative layer + derived index, `distill`/`recall`/`curate`/
  `verify`) plugs in there as a `MemoryProvider`, so what compaction drops is
  distilled instead of forgotten.

## License

MIT. The deterministic core derives from `ctxfold` (MIT).
