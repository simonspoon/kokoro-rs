#!/usr/bin/env python3
"""Differential test: Rust phonemisation against the Python original.

Checks both the IPA string and the resulting token IDs, since a divergence in
either changes the audio.

    cargo build --release --example dump_phonemes
    ./scripts/diff_phonemes.py [path-to-dump_phonemes-binary]

Requires the kokoro3 virtualenv, which supplies the reference implementation.
"""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

KOKORO3 = Path(__file__).resolve().parents[2] / "kokoro3"
sys.path.insert(0, str(KOKORO3 / "src"))

from kokoro_onnx.tokenizer import Tokenizer  # noqa: E402

BINARY = Path(sys.argv[1] if len(sys.argv) > 1 else "target/release/examples/dump_phonemes").resolve()

CASES = [
    "Hello from Rust.",
    "One. Two. Three.",
    "Dr. Smith arrived, quickly.",
    "no punctuation here",
    "Wait... really?!",
    '"Quoted text," she said; then left.',
    "A list: one, two, and three.",
    "(Parenthetical) remarks — and dashes.",
    "It cost 3.5 pounds, or $4.20.",
    "!?!",
    ".",
    "  spaced   out   words  ",
    "Ellipsis… and more",
    "e.g. i.e. etc.",
    "U.S. and U.K. relations",
    "CAPS LOCK SHOUTING!",
    "numbers 1 2 3 and 42",
    "hyphen-ated words",
    "it's a contraction, isn't it?",
    "Multi\nline\ntext",
    "«guillemets» and “smart quotes”",
    "Mixed: [brackets] {braces} (parens)",
    "trailing comma,",
    ",leading comma",
    "a",
    "The quick brown fox jumps over the lazy dog.",
    "¿Questions? ¡Exclamations!",
    "Tab\tseparated",
]

# Word soup, to catch placement bugs the curated cases miss.
WORDS = ["alpha", "beta", "gamma", "delta", "hello", "world", "test", "value", "one", "two"]
PUNCT = [".", ",", "!", "?", ";", ":", "...", " - ", '"', "(", ")", "—"]


def generated(rng: random.Random, n: int) -> list[str]:
    out = []
    for _ in range(n):
        parts = []
        for _ in range(rng.randint(1, 8)):
            parts.append(rng.choice(WORDS))
            if rng.random() < 0.4:
                parts.append(rng.choice(PUNCT))
            parts.append(" ")
        out.append("".join(parts).strip())
    return out


def main() -> int:
    tokenizer = Tokenizer()
    cases = CASES + generated(random.Random(20260729), 120)

    # One process for all cases: espeak initialisation dominates otherwise.
    result = subprocess.run(
        [str(BINARY), *cases], capture_output=True, text=True, check=True
    )
    lines = result.stdout.splitlines()
    if len(lines) != len(cases):
        print(f"expected {len(cases)} lines, got {len(lines)}")
        return 1

    failures = 0
    for text, line in zip(cases, lines, strict=True):
        rust_phonemes, _, rust_tokens = line.partition("\t")
        expected_phonemes = tokenizer.phonemize(text, "en-us")
        expected_tokens = str(tokenizer.tokenize(expected_phonemes))
        if rust_phonemes != expected_phonemes or rust_tokens != expected_tokens:
            failures += 1
            print(f"MISMATCH {text!r}")
            print(f"  python: {expected_phonemes!r}")
            print(f"  rust:   {rust_phonemes!r}")

    print(f"\n{len(cases) - failures}/{len(cases)} cases agree")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
