#!/usr/bin/env python3
"""Differential test: Rust audio output against the Python original.

Renders the same text with both implementations and compares the WAV samples.
Output is normally bit-identical: the two run the same ONNX graph on the same
weights, so a difference in length, or in more than the last bit, means the
port diverges somewhere upstream of the model.

A handful of samples in a long utterance can differ by one 16-bit LSB. Each
implementation is deterministic on its own; the two accumulate ONNX reductions
in a different order, and a result landing on a quantisation boundary rounds
the other way. That is ~90 dB below the signal and is tolerated here.

    cargo build --release
    ./scripts/diff_audio.py [path-to-kokoro-rs-binary]

Requires the kokoro3 virtualenv and its cached model files.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

KOKORO3 = Path(__file__).resolve().parents[2] / "kokoro3"
BINARY = Path(sys.argv[1] if len(sys.argv) > 1 else "target/release/kokoro-rs").resolve()
PYTHON = KOKORO3 / ".venv" / "bin" / "python"

# (text, voice, speed) — a spread of lengths, punctuation, voices and rates.
CASES = [
    ("Hello from Rust.", "af_heart", 1.0),
    ("One. Two. Three.", "af_heart", 1.0),
    ("Dr. Smith arrived, quickly.", "af_heart", 1.0),
    ("no punctuation at all here", "af_heart", 1.0),
    ("Wait... really?!", "af_heart", 1.0),
    ("A somewhat longer sentence, with a clause in the middle, to check that "
     "chunking and the inter-sentence gap behave the same way in both.", "af_heart", 1.0),
    ("Hello from Rust.", "bm_george", 1.0),
    ("Hello from Rust.", "af_bella", 1.0),
    ("Hello from Rust.", "am_michael", 1.0),
    ("Speaking a little faster now.", "af_heart", 1.5),
    ("Speaking a little slower now.", "af_heart", 0.75),
    ("Numbers like 3.5 and 42 should read the same.", "af_heart", 1.0),
    ("The quick brown fox jumps over the lazy dog. " * 6, "af_heart", 1.0),
]


def render_rust(out: Path, text: str, voice: str, speed: float) -> None:
    subprocess.run(
        [str(BINARY), "-q", "-o", str(out), "-v", voice, "-s", str(speed), text],
        check=True, capture_output=True,
    )


def render_python(out: Path, text: str, voice: str, speed: float) -> None:
    subprocess.run(
        [str(PYTHON), "-m", "kokoro3.cli", "-q", "-o", str(out),
         "-v", voice, "-s", str(speed), text],
        check=True, capture_output=True, cwd=KOKORO3,
    )


def main() -> int:
    failures = 0
    inexact = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i, (text, voice, speed) in enumerate(CASES):
            rust_wav, py_wav = tmp / f"r{i}.wav", tmp / f"p{i}.wav"
            render_rust(rust_wav, text, voice, speed)
            render_python(py_wav, text, voice, speed)

            r, _ = sf.read(rust_wav, dtype="int16")
            p, _ = sf.read(py_wav, dtype="int16")
            label = f"{voice} @{speed} {text[:44]!r}"

            if r.shape != p.shape:
                failures += 1
                print(f"MISMATCH length {label}: rust {r.shape} vs python {p.shape}")
                continue
            delta = np.abs(r.astype(int) - p.astype(int)) if len(r) else np.zeros(0, int)
            diff = int(delta.max()) if len(delta) else 0
            if diff > 1:
                failures += 1
                print(f"MISMATCH samples {label}: max diff {diff} LSB")
            elif diff == 1:
                inexact += 1
                print(f"ok~ {label}  ({len(r)} samples, {int((delta > 0).sum())} at 1 LSB)")
            else:
                print(f"ok  {label}  ({len(r)} samples)")

    exact = len(CASES) - failures - inexact
    print(f"\n{exact} bit-identical, {inexact} within 1 LSB, {failures} mismatched")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
