"""
Streams from Claude, batching sentences into "packets" of up to
PACKET_SENTENCE_LIMIT. A short reply (which is the default — see
SYSTEM_PROMPT) comes out as ONE packet, so it plays as a single TTS call
with zero internal gaps. A long reply gets split into packets so packet 1
can start playing while packet 2 is still generating/fetching.
"""
import re
from anthropic import Anthropic

import config

_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
_history = []


def _split_sentences(buffer: str):
    parts = re.split(r"(?<=[.!?])\s+", buffer)
    complete = parts[:-1]
    remainder = parts[-1] if parts else ""
    return complete, remainder


def ask_streaming_packets(user_text: str):
    _history.append({"role": "user", "content": user_text})
    trimmed = _history[-config.MAX_HISTORY_TURNS:]

    buffer = ""
    full_reply = ""
    pending = []

    with _client.messages.stream(
        model=config.CLAUDE_MODEL,
        max_tokens=1024,
        system=config.SYSTEM_PROMPT,
        messages=trimmed,
    ) as stream:
        for text in stream.text_stream:
            buffer += text
            full_reply += text
            complete, buffer = _split_sentences(buffer)
            for sentence in complete:
                if sentence.strip():
                    pending.append(sentence.strip())
            if len(pending) >= config.PACKET_SENTENCE_LIMIT:
                yield " ".join(pending)
                pending = []

    if buffer.strip():
        pending.append(buffer.strip())
    if pending:
        yield " ".join(pending)

    _history.append({"role": "assistant", "content": full_reply})