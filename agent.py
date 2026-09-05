#!/usr/bin/env python3
"""
agent.py — Agentic network troubleshooting assistant.

Give it a goal like "why can't host A reach host B?" and it will decide
which diagnostic tool to run (ping, traceroute, or read a config file),
look at the real result, and decide the next step — repeating until it
reaches a conclusion or hits the step limit. This is a from-scratch
implementation of the ReAct-style agent loop using Ollama's tool-calling
API (no LangChain or agent framework).

Usage:
    python agent.py --goal "figure out why 10.0.0.5 is unreachable from here"
    python agent.py            (interactive: prompts you for a goal)

Requires an Ollama model that supports tool calling, e.g. llama3.2 or
llama3.1. Run `ollama pull llama3.2` if you haven't already.
"""

import argparse
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import requests

OLLAMA_HOST = "http://localhost:11434"
MODEL = "llama3.2"
MAX_STEPS = 8  # hard cap so a confused agent can't loop forever

# Config files the agent is allowed to read live only in this folder —
# this is a hard boundary, not a suggestion, enforced in read_config_file().
CONFIG_DIR = Path("configs")

# Only allow characters valid in a hostname/IP — blocks shell metacharacters
# like ; | & $ ` that could otherwise be smuggled into a subprocess call.
HOST_PATTERN = re.compile(r"^[a-zA-Z0-9\.\-]+$")

SYSTEM_PROMPT = """You are a network troubleshooting assistant with access \
to real diagnostic tools. Given a goal, investigate step by step: call a \
tool, look at its real output, and decide what to check next. Reason like \
an experienced network engineer using the OSI model — check physical/link \
reachability before assuming a routing or config problem.

IMPORTANT: Stop investigating as soon as you have a config or log entry \
that directly explains the problem (e.g. an ACL, firewall rule, or policy \
that explicitly blocks the traffic in question). That is conclusive \
evidence — do not keep pinging additional hosts "just to be thorough" \
once you've found the actual documented cause. Only keep investigating if \
the evidence so far is genuinely inconclusive or contradictory.

Once you have enough evidence, stop calling tools and give a clear final \
diagnosis explaining what you found and why, citing the specific command \
output or config line that supports your conclusion. Don't guess — only \
conclude what the evidence actually supports."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ping_host",
            "description": "Ping a host to check basic reachability (ICMP).",
            "parameters": {
                "type": "object",
                "properties": {
                    "host": {"type": "string", "description": "Hostname or IP address to ping"},
                },
                "required": ["host"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "traceroute_host",
            "description": "Trace the network path (hop by hop) to a host to find where connectivity breaks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "host": {"type": "string", "description": "Hostname or IP address to trace"},
                },
                "required": ["host"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_config_file",
            "description": "Read a network config file (e.g. VLAN or subnet definitions) from the local configs/ folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Filename inside the configs/ folder, e.g. 'vlan10.txt'"},
                },
                "required": ["filename"],
            },
        },
    },
]


def _validate_host(host: str) -> str | None:
    """Return an error string if host looks unsafe/invalid, else None."""
    if not host or not HOST_PATTERN.match(host) or len(host) > 253:
        return f"Refused: '{host}' is not a valid hostname/IP (only letters, digits, dots, hyphens allowed)."
    return None


def ping_host(host: str) -> str:
    err = _validate_host(host)
    if err:
        return err
    count_flag = "-n" if platform.system() == "Windows" else "-c"
    try:
        result = subprocess.run(
            ["ping", count_flag, "4", host],
            capture_output=True, text=True, timeout=15,
        )
        return (result.stdout or result.stderr).strip()[:2000]
    except subprocess.TimeoutExpired:
        return f"Ping to {host} timed out (no response within 15s)."
    except FileNotFoundError:
        return "Error: 'ping' command not found on this system."


def traceroute_host(host: str) -> str:
    err = _validate_host(host)
    if err:
        return err
    cmd = ["tracert", host] if platform.system() == "Windows" else ["traceroute", host]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return (result.stdout or result.stderr).strip()[:2000]
    except subprocess.TimeoutExpired:
        return f"Traceroute to {host} timed out."
    except FileNotFoundError:
        return f"Error: '{cmd[0]}' command not found on this system."


def read_config_file(filename: str) -> str:
    # Resolve against CONFIG_DIR only, then verify the result is still
    # inside CONFIG_DIR — blocks path traversal like "../../secrets.txt".
    CONFIG_DIR.mkdir(exist_ok=True)
    target = (CONFIG_DIR / filename).resolve()
    if CONFIG_DIR.resolve() not in target.parents and target != CONFIG_DIR.resolve():
        return f"Refused: '{filename}' is outside the configs/ folder."
    if not target.exists():
        return f"No such config file: {filename} (looked in {CONFIG_DIR}/)"
    return target.read_text(encoding="utf-8", errors="ignore")[:3000]


TOOL_FUNCTIONS = {
    "ping_host": lambda args: ping_host(args.get("host", "")),
    "traceroute_host": lambda args: traceroute_host(args.get("host", "")),
    "read_config_file": lambda args: read_config_file(args.get("filename", "")),
}


def call_ollama(messages: list, host: str, model: str, timeout: int = 120) -> dict:
    resp = requests.post(
        f"{host}/api/chat",
        json={"model": model, "messages": messages, "tools": TOOLS, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def run_agent(goal: str, host: str, model: str, max_steps: int, timeout: int = 120):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": goal},
    ]

    for step in range(1, max_steps + 1):
        print(f"\n--- Step {step} --- (waiting on model, this can take a while for larger models)")
        try:
            response = call_ollama(messages, host, model, timeout)
        except requests.RequestException as exc:
            print(f"[!] Ollama request failed: {exc}")
            return

        message = response.get("message", {})
        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            # No tool call means the model is giving its final answer.
            answer = message.get("content", "").strip()
            print(f"\n=== Final diagnosis ===\n{answer}")
            return

        messages.append(message)
        for call in tool_calls:
            fn_name = call["function"]["name"]
            fn_args = call["function"].get("arguments", {})
            if isinstance(fn_args, str):
                try:
                    fn_args = json.loads(fn_args)
                except json.JSONDecodeError:
                    fn_args = {}

            print(f"[tool call] {fn_name}({fn_args})")
            fn = TOOL_FUNCTIONS.get(fn_name)
            if fn is None:
                result = f"Unknown tool: {fn_name}"
            else:
                result = fn(fn_args)
            print(f"[result] {result[:300]}{'...' if len(result) > 300 else ''}")

            messages.append({
                "role": "tool",
                "content": result,
            })

    print(f"\n[!] Hit the {max_steps}-step limit without a final answer. "
          f"The agent may be stuck — check the steps above for where it looped.")


def main():
    parser = argparse.ArgumentParser(description="Agentic network troubleshooting assistant")
    parser.add_argument("--goal", help="What you want the agent to investigate")
    parser.add_argument("--model", default=MODEL, help=f"Ollama model (default: {MODEL})")
    parser.add_argument("--host", default=OLLAMA_HOST, help=f"Ollama host (default: {OLLAMA_HOST})")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS,
                         help=f"Max reasoning steps before giving up (default: {MAX_STEPS})")
    parser.add_argument("--timeout", type=int, default=300,
                         help="Seconds to wait per model response (default: 300, raise for slower/larger models)")
    args = parser.parse_args()

    goal = args.goal
    if not goal:
        try:
            goal = input("What should the agent investigate? ").strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)
    if not goal:
        print("No goal given, exiting.")
        sys.exit(1)

    run_agent(goal, args.host, args.model, args.max_steps, args.timeout)


if __name__ == "__main__":
    main()
