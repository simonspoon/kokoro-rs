#!/usr/bin/env python3
"""Differential test: the Rust chunker against the Python original.

Runs both implementations over a corpus of inputs and reports any divergence.
The Rust side is reached through the `dump_chunks` example:

    cargo build --example dump_chunks
    ./scripts/diff_chunking.py [path-to-dump_chunks-binary]

Requires the kokoro3 virtualenv, which supplies the reference implementation.
"""

from __future__ import annotations

import itertools
import json
import random
import subprocess
import sys
from pathlib import Path

KOKORO3 = Path(__file__).resolve().parents[2] / "kokoro3"
sys.path.insert(0, str(KOKORO3 / "src"))

from kokoro3.text import chunk_text, ends_sentence, stream_chunks  # noqa: E402

BINARY = Path(sys.argv[1] if len(sys.argv) > 1 else "target/debug/examples/dump_chunks").resolve()

# Fragments chosen to exercise every branch: sentence ends, abbreviations,
# initials, decimals, closing quotes, clause breaks, unbreakable tokens,
# newlines and multi-byte characters.
FRAGMENTS = [
    "Hello there.",
    "One. Two. Three.",
    "Dr. Smith arrived at 3.5 past noon.",
    "J. R. R. Tolkien wrote it.",
    '"Stop!" she said.',
    "Wait... really?",
    "alpha, beta; gamma: delta — epsilon",
    "See Fig. 4 and Vol. 2, i.e. the later ones.",
    "a" * 120,
    "one\ntwo\nthree",
    "Καλημέρα κόσμε. Πώς είσαι;",
    "Mixed 123 numbers and U.S. states etc. here!",
    "no terminal punctuation at all",
    "   ",
    "!?!?",
    "e.g. this, that; and the other thing which runs on and on and on",
]


def rust_chunks(mode: str, limit: int, chunk_chars: int, pieces: list[str]) -> list[str]:
    request = json.dumps({"mode": mode, "limit": limit, "chunk_chars": chunk_chars, "pieces": pieces})
    out = subprocess.run(
        [str(BINARY)], input=request, capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout)


def cases():
    """Yield (description, mode, limit, chunk_chars, pieces)."""
    rng = random.Random(20260729)

    for text, limit in itertools.product(FRAGMENTS, (5, 12, 40, 100, 300)):
        yield f"chunk_text({text!r}, {limit})", "chunk", limit, limit, [text]

    # Whole documents built from the fragments, fed as one blob.
    for n in range(40):
        text = " ".join(rng.choice(FRAGMENTS) for _ in range(rng.randint(1, 8)))
        yield f"doc#{n}", "chunk", rng.choice((20, 60, 100, 300)), 300, [text]

    # The same documents split into arbitrary pieces, exercising the streaming
    # path the way stdin arrives — the split points must not change the output.
    for n in range(60):
        text = " ".join(rng.choice(FRAGMENTS) for _ in range(rng.randint(1, 6)))
        pieces, i = [], 0
        while i < len(text):
            step = rng.randint(1, 25)
            pieces.append(text[i : i + step])
            i += step
        first = rng.choice((10, 40, 100))
        yield f"stream#{n}", "stream", first, rng.choice((60, 300)), pieces or [""]


def main() -> int:
    failures = 0
    total = 0
    for desc, mode, limit, chunk_chars, pieces in cases():
        total += 1
        if mode == "chunk":
            expected = list(chunk_text(pieces[0], limit))
        else:
            expected = list(stream_chunks(pieces, limit, chunk_chars))
        actual = rust_chunks(mode, limit, chunk_chars, pieces)
        if actual != expected:
            failures += 1
            print(f"MISMATCH {desc}\n  python: {expected}\n  rust:   {actual}")

    # ends_sentence must agree too: it decides where the inter-sentence gap goes.
    probes = [c for f in FRAGMENTS for c in chunk_text(f, 40)] + ["x", "", "a?", 'b." ']
    rust_flags = rust_chunks("ends_sentence", 0, 0, probes)
    for probe, flag in zip(probes, rust_flags, strict=True):
        total += 1
        if bool(flag) != ends_sentence(probe):
            failures += 1
            print(f"MISMATCH ends_sentence({probe!r}): python={ends_sentence(probe)} rust={flag}")

    print(f"\n{total - failures}/{total} cases agree")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
