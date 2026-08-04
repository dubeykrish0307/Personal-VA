"""
Run second: `python3 voice_id/enroll.py`

Builds your voiceprint from the recordings using Resemblyzer's pretrained
speaker-encoder — an already-trained model, not something we train fresh.
We're extracting YOUR voice's fingerprint from it and saving that
fingerprint for comparison later. Fast, needs little data, fully local.
"""
import os
import sys
import traceback
import numpy as np

RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), "recordings")
PROFILE_DIR = os.path.join(os.path.dirname(__file__), "profile")
PROFILE_PATH = os.path.join(PROFILE_DIR, "voiceprint.npy")


def main():
    if not os.path.isdir(RECORDINGS_DIR) or not os.listdir(RECORDINGS_DIR):
        print("No recordings found. Run voice_id/record.py first.")
        return

    print("Importing resemblyzer...")
    from resemblyzer import VoiceEncoder, preprocess_wav

    print("Loading speaker-encoder model (one-time download on first run)...")
    encoder = VoiceEncoder()

    embeddings = []
    files = sorted(f for f in os.listdir(RECORDINGS_DIR) if f.endswith(".wav"))
    print(f"Found {len(files)} recordings.")

    for fname in files:
        path = os.path.join(RECORDINGS_DIR, fname)
        wav = preprocess_wav(path)
        embeddings.append(encoder.embed_utterance(wav))
        print(f"  processed {fname}")

    if not embeddings:
        print("No embeddings produced — recordings may be invalid.")
        return

    voiceprint = np.mean(embeddings, axis=0)
    voiceprint = voiceprint / np.linalg.norm(voiceprint)

    os.makedirs(PROFILE_DIR, exist_ok=True)
    np.save(PROFILE_PATH, voiceprint)
    print(f"\nVoiceprint saved to {PROFILE_PATH}")
    print(f"Built from {len(embeddings)} clips.")
    print("Run main.py — the wake word will now only respond to your voice.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n=== ENROLLMENT FAILED ===")
        traceback.print_exc()
        sys.exit(1)