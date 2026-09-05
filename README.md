# TraceMind

<p align="center"><img src="logo.svg" alt="TraceMind logo" width="500"></p>

An agentic AI tool that investigates network connectivity problems the way
an engineer would: given a goal ("why can't host A reach host B?"), it
decides which diagnostic to run (ping, traceroute, or read a VLAN/subnet
config), looks at the **real output**, and decides what to check next —
looping until it reaches a supported conclusion. Runs fully locally via
[Ollama](https://ollama.com).

## Why this is "agentic" and not RAG

| | RAG (retrieval) | This project (agentic) |
|---|---|---|
| Model gets | Pre-fetched text chunks | A goal, and tools it can call itself |
| Steps | One retrieval, one answer | Multi-step: act → observe → reason → repeat |
| Ground truth | Your notes (static) | Live command output (real ping/traceroute results) |
| Model's job | Summarize/answer from context | Decide *what to investigate next* |

## Architecture

```
goal ("why can't 192.168.20.50 reach 192.168.10.10?")
      │
      ▼
 ┌─────────────────────────────────────────────┐
 │  agent loop (agent.py)                       │
 │  1. Send goal + tool definitions to the LLM  │
 │  2. LLM picks a tool + arguments             │
 │  3. Tool actually runs (ping/traceroute/     │
 │     read config) — real subprocess output    │
 │  4. Result fed back to the LLM               │
 │  5. LLM decides: call another tool, or       │
 │     give a final diagnosis                   │
 │  (repeats until answer or step limit)        │
 └─────────────────────────────────────────────┘
```

This is a from-scratch ReAct-style loop using Ollama's native tool-calling
API — no LangChain, no agent framework — so every part of the control flow
is visible and explainable.

## Safety design (read this before running)

Letting an LLM decide which shell commands to run is a real risk surface.
This project locks it down in two ways:

1. **Host/IP validation** — `ping_host` and `traceroute_host` reject any
   input containing characters outside `[a-zA-Z0-9.-]`. This blocks
   command-injection attempts like `8.8.8.8; rm -rf /` before they ever
   reach `subprocess`. Commands are run as an argument list (never
   `shell=True`), which is the other half of injection-proofing.
2. **Sandboxed file access** — `read_config_file` can only read files
   inside the local `configs/` folder. Path traversal attempts
   (`../secret.txt`, `../../etc/passwd`) are resolved and checked against
   that boundary and refused.

Both are unit-tested — see the "Testing" section below.

## Setup

```bash
ollama pull gpt-oss:20b      # OpenAI's open-weight model, supports tool calling
pip install -r requirements.txt
```

**Hardware note:** `gpt-oss:20b` runs best with 16GB+ RAM. On CPU-only
hardware at that size, each reasoning step can take 1-5 minutes — the
script's default per-step timeout is 300 seconds (`--timeout` to adjust).
If your machine is lighter on RAM, `llama3.2` or `qwen3:4b` are faster
alternatives that also support tool calling.

Sample VLAN configs are included in `configs/` to give the agent something
realistic to investigate (an engineering VLAN and a guest VLAN with an ACL
blocking inter-VLAN traffic — a classic "why can't I reach that host"
scenario). Add your own `.txt` files there to describe your actual lab.

## Usage

```bash
python agent.py --goal "why can't 192.168.20.50 reach 192.168.10.10?" --model gpt-oss:20b
```

Or run it interactively:
```bash
python agent.py --model gpt-oss:20b
```

You'll see each step printed as it happens — which tool was called, with
what arguments, and the real output — followed by a final diagnosis once
the agent has enough evidence.

## Testing the safety guardrails

```bash
python3 -c "
from agent import _validate_host, read_config_file
print(_validate_host('8.8.8.8; rm -rf /'))   # should be refused
print(read_config_file('../../etc/passwd'))   # should be refused
"
```

## Known limitations

- Small local models occasionally call a tool with malformed arguments;
  the agent handles this gracefully (empty args → validation error →
  fed back to the model) rather than crashing.
- **Observed in testing:** the same goal, run twice, didn't always behave
  identically. In one run, the model gathered the correct evidence (found
  the ACL block in the VLAN config) but then kept pinging additional hosts
  instead of concluding, hitting the step limit with no final answer. The
  fix was making the system prompt explicit about *when* evidence counts
  as sufficient ("stop once you find a config/log entry that directly
  explains the problem") rather than leaving that judgment implicit. After
  that change, repeated test runs consistently reached a correct
  conclusion, though the model sometimes still probes for a couple of
  nonexistent config filenames before settling — reasonable investigative
  behavior, not a bug.
- Traceroute can take 10-30 seconds per hop chain; be patient on slower
  connections. `gpt-oss:20b` on CPU-only hardware adds further latency
  per reasoning step (see hardware note above).

## Tech stack

Python, [Ollama](https://ollama.com) tool-calling API, `gpt-oss:20b`
(OpenAI's open-weight model), `requests`. No agent framework — built
from first principles to demonstrate understanding of the ReAct loop
itself.
