"""
Two backends. ElevenLabs now streams TRUE live audio — chunks are piped
directly into ffplay's stdin as they arrive from the network, so playback
starts as soon as the first bit of audio is ready instead of waiting for
the whole clip to download. This is the actual fix for the remaining
latency — afplay couldn't do this reliably, ffplay can.
"""
import subprocess
import requests

import config

_session = requests.Session()


def prepare(text: str):
    # No separate fetch step needed anymore — streaming happens inside play().
    return text


def play(text: str):
    if not text or not text.strip():
        return
    if config.TTS_BACKEND == "elevenlabs":
        _speak_elevenlabs_streaming(text)
    else:
        _speak_local(text)


def _speak_local(text: str):
    cmd = ["say"]
    if config.LOCAL_VOICE_NAME:
        cmd += ["-v", config.LOCAL_VOICE_NAME]
    cmd.append(text)
    subprocess.run(cmd)


def _speak_elevenlabs_streaming(text: str):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{config.ELEVENLABS_VOICE_ID}/stream"
    params = {"optimize_streaming_latency": 4}
    headers = {
        "xi-api-key": config.ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": config.ELEVENLABS_MODEL,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }

    response = _session.post(url, params=params, json=payload, headers=headers, stream=True)
    response.raise_for_status()

    # ffplay reads raw mp3 bytes straight from stdin and starts playing as
    # soon as it has enough buffered — this is real streaming, unlike the
    # named-pipe approach that broke afplay earlier.
    proc = subprocess.Popen(
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-i", "pipe:0"],
        stdin=subprocess.PIPE,
    )
    try:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                proc.stdin.write(chunk)
    finally:
        proc.stdin.close()
        proc.wait()