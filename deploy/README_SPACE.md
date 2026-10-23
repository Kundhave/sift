---
title: Sift
emoji: 🔍
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Local verification layer that ranks AI-agent diff hunks by intent misalignment
---

# Sift

Sift checks whether each hunk of an AI coding agent's diff serves what the developer asked for. It consolidates the chat
intent, scores every hunk with a ModernBERT cross-encoder (or the distilled INT8 MiniLM student), ranks hunks by
misalignment risk and explains the verdict. All inference runs inside this container on CPU; nothing is sent elsewhere.

Free CPU tier: the first request to the 149M-parameter main model takes a while (model load); the INT8 student is much faster.
Dataset limits: misaligned hunks are constructed from donor commits, Python/JS/TS only. Latencies belong to the host CPU.
