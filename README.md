# kokoro-rs

Command-line text-to-speech using [Kokoro v1.0](https://huggingface.co/hexgrad/Kokoro-82M),
an 82M-parameter open-weight model. Takes text as an argument or on stdin and
streams the audio to your speakers as it is generated, or writes it to a file.

A Rust rewrite of [`kokoro3`](../kokoro3), the Python version. Output is
bit-identical to it for the same text, voice and speed — see
[Verification](#verification).

Everything runs locally on the CPU. After the first run — which downloads the
model — no network access is needed.

## Install

```sh
cargo build --release
```

The first invocation downloads the model (~326 MB), the voices (~28 MB) and the
ONNX Runtime shared library (~40 MB) into `~/.cache/kokoro-rs`. espeak-ng is
built from source at compile time and its data is embedded in the binary, so
there is nothing to install system-wide.

## Use

```sh
kokoro-rs "Hello from the command line."     # speak it
echo "text from a pipe" | kokoro-rs          # read stdin
kokoro-rs -o out.wav < book.txt              # write a file instead
kokoro-rs -o - "piped audio" | ffplay -      # stream WAV to stdout
tail -f app.log | kokoro-rs                  # speak a log as it grows
```

Options:

| Flag | Meaning |
| --- | --- |
| `-o, --output FILE` | Write audio to `FILE` instead of the speakers; `-` streams WAV to stdout. |
| `-v, --voice NAME` | Voice to use (default `af_heart`). See `--list-voices` for all 54. |
| `-s, --speed RATE` | Speech rate, 0.5–2.0 (default 1.0). |
| `-l, --lang LANG` | Language passed to the phonemiser (default `en-us`). |
| `-d, --device DEV` | Output device name or index; see `--list-devices`. |
| `--gap SECONDS` | Silence inserted between sentences (default 0.12). |
| `--chunk-chars N` | Max characters per synthesis chunk (default 300). |
| `--no-download` | Fail rather than fetch missing model files. |
| `-q, --quiet` | Suppress the progress line. |

Ctrl-C stops promptly, within about a second.

## How the streaming works

Text is split into chunks on sentence boundaries and synthesised one chunk at a
time. The first chunk is deliberately short (~100 characters) so sound starts
quickly; later chunks are larger, which gives the model more context and better
prosody. Playback runs on the audio callback behind a four-deep channel, so the
next chunk is being generated while the current one is still being heard. The
channel is bounded on purpose: synthesis outruns playback by roughly 3x, so an
unbounded one would hold an entire document's audio in memory.

Because chunks are synthesised separately, the model's natural inter-sentence
pause is clipped; `--gap` puts it back, and is applied only at real sentence
ends, never where an over-long sentence had to be split.

## Verification

The port is checked against the Python implementation at three levels, each
with a script under `scripts/` that runs both and compares:

| Script | Checks | Result |
| --- | --- | --- |
| `diff_chunking.py` | chunk boundaries, including incremental stdin | 204/204 identical |
| `diff_phonemes.py` | IPA strings and token IDs | 148/148 identical |
| `diff_audio.py` | rendered WAV samples | 12/13 bit-identical, 1 within 1 LSB |

They need the `kokoro3` virtualenv, which supplies the reference:

```sh
cargo build --release
cargo build --example dump_chunks
cargo build --release --example dump_phonemes
../kokoro3/.venv/bin/python scripts/diff_audio.py
```

In the one inexact audio case, 3 samples out of 408,192 differ by a single
16-bit LSB. Each implementation is deterministic on its own; the two accumulate
ONNX reductions in a different order, and a value landing on a quantisation
boundary rounds the other way. That is about 90 dB below the signal.

## Notes

- **Intel Macs.** `ort` ships no prebuilt ONNX Runtime for x86_64 macOS, so the
  binary loads it dynamically and fetches Microsoft's own 1.23.2 release — the
  last with an official Intel build, and the version `kokoro3` pinned for the
  same reason. Set `ORT_DYLIB_PATH` to use a different one.
- **Deliberate differences from the Python version.** File output is WAV only,
  where `soundfile` also offered FLAC and others; a non-WAV extension is
  rejected with a clear message rather than silently mis-encoded. Float samples
  are converted to 16-bit with libsndfile's convention everywhere, so the
  stdout stream now matches file output exactly — the Python version used a
  slightly different cast there and the two disagreed by a bit or two.
- **espeak-ng data** lives in `assets/espeak-ng-data.tar.gz`, vendored because
  cargo does not guarantee a dependency's build script has run before this
  crate's. Regenerate it with `scripts/vendor_espeak_data.sh` after bumping
  `espeak-rs-sys`.
- Model location can be overridden with `KOKORO_MODEL`, `KOKORO_VOICES`, or
  `KOKORO_HOME`.
