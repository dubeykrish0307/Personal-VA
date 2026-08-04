# JARVIS voice assistant

Wake word -> speech-to-text -> Claude -> text-to-speech, all local except
the Claude and (optional) ElevenLabs API calls.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY (and ELEVENLABS_API_KEY if using that backend)
```

## Enroll your voice (do this before running main.py)

```bash
python3 voice_id/record.py    # records you saying "hey jarvis" a few times + free speech
python3 voice_id/enroll.py    # builds voiceprint.npy from those recordings
```

Without this, the assistant will still run, but it'll respond to *anyone's*
voice saying "hey jarvis" — main.py prints a warning on startup if no
voiceprint is found.

## Run

```bash
python3 main.py
```

Say "hey jarvis" (not bare "jarvis" — the pretrained openWakeWord model
was trained on the full phrase and misses more often without "hey").
After it replies, you can keep talking without saying the wake word again
for `FOLLOWUP_WINDOW_SECONDS` (config.py).

## Tuning

All the knobs are in `config.py`, with comments. The ones most worth
touching first:
- `WAKE_WORD_THRESHOLD` — lower if it's not waking reliably, raise if it
  fires on background noise/TV.
- `VOICE_MATCH_THRESHOLD` — lower if your own voice gets rejected, raise
  if others are triggering it.
- `SILENCE_DURATION_MS` — how long a pause before it decides you're done
  talking. Lower feels snappier but risks cutting you off mid-sentence.
