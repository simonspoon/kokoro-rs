# kokoro-rs

Command-line text-to-speech using [Kokoro v1.0](https://huggingface.co/hexgrad/Kokoro-82M),
an 82M-parameter open-weight model. Takes text as an argument or on stdin and
streams the audio to your speakers as it is generated, or writes it to a file.

A Rust rewrite of `kokoro3`, an unpublished Python implementation of the same
model. Output is bit-identical to it for the same text, voice and speed — see
[Verification](#verification).

Everything runs locally on the CPU. After the first run — which downloads the
model — no network access is needed.

## Install

```sh
brew install simonspoon/tap/kokoro-rs
```

### Build from source

espeak-ng is compiled from source as part of the build, so you need Rust
(edition 2024), CMake, a C and C++ compiler, and libclang for the bindings.
On Debian or Ubuntu:

```sh
sudo apt-get install cmake clang libclang-dev pkg-config libasound2-dev
```

`libasound2-dev` is for ALSA, which the audio output needs. On macOS the Xcode
command line tools plus `brew install cmake` cover it.

```sh
git clone https://github.com/simonspoon/kokoro-rs.git
cd kokoro-rs
scripts/install.sh
```

That builds in release mode and copies the binary to `~/.local/bin`. Override
the destination with `PREFIX` (installs into `$PREFIX/bin`) or `BINDIR` (an
exact directory):

```sh
PREFIX=/usr/local scripts/install.sh
BINDIR=/opt/bin scripts/install.sh
```

To build without installing, use `cargo build --release`; the binary lands in
`target/release/kokoro-rs`.

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
| `--chunk-chars N` | Optional cap on the characters per synthesis chunk; off by default. |
| `--chunk-phonemes N` | Max phonemes per synthesis chunk, the model reads 510 (default 500). |
| `--no-download` | Fail rather than fetch missing model files. |
| `-q, --quiet` | Suppress the progress line. |

Ctrl-C stops promptly, within about a second.

## How the streaming works

Text is split into chunks on sentence boundaries and synthesised one chunk at a
time. The first chunk is deliberately short so sound starts quickly, and each
one after it may be up to twice what the last actually came to, growing until
chunks are packed to the model's phoneme budget rather than a character count,
which gives it more context and better prosody. The ramp matters: going
straight from a two-second opening phrase to a full-sized chunk leaves playback
with nothing to say for several seconds while that chunk is synthesised.
Characters are only a proxy for the real limit — "1234567" is seven of them and
forty phonemes — so the text is phonemised once, up front, and measured
directly; the phonemes are carried through to the synthesiser rather than
derived a second time.

Playback runs on the audio callback behind a four-deep channel, so the next
chunk is being generated while the current one is still being heard. The
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
| `diff_chunking.py` | chunk boundaries, including incremental stdin | 204/204 identical, before the phoneme budget below |
| `diff_phonemes.py` | IPA strings and token IDs | 148/148 identical |
| `diff_audio.py` | rendered WAV samples | 12/13 bit-identical, 1 within 1 LSB |

Chunk boundaries now deliberately diverge from the Python original, which sizes
chunks in characters, so the 204/204 above describes the behaviour before the
phoneme budget was introduced.

The scripts expect `kokoro3` checked out beside this repo, with its virtualenv
built, since that is what supplies the reference output. That implementation is
not published, so these are recorded results rather than something you can
re-run from a fresh clone:

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

- **ONNX Runtime.** `ort` ships no prebuilt binary for x86_64 macOS, so rather
  than link one at build time kokoro-rs loads it dynamically and fetches
  Microsoft's own release for the platform it is running on. The version is
  pinned at 1.23.2 — the last with an official Intel Mac build, which is the
  constraint that set it; the other three platforms publish that version too,
  so one pin covers them all. Set `ORT_DYLIB_PATH` to use a different one.
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

## License

GPL-3.0-or-later — see [LICENSE](LICENSE).

The choice is not really a choice: kokoro-rs links espeak-ng (through
`espeak-rs-sys`) into the binary, and espeak-ng is GPL-3.0-or-later, so
anything built here is a derivative work under that licence. The vendored
`assets/espeak-ng-data.tar.gz` is espeak-ng's own data, under the same terms.

Two things are *not* covered by it and are fetched at runtime rather than
bundled: the [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) model and
its voices (Apache-2.0), and the ONNX Runtime shared library (MIT).

[THIRD-PARTY.md](THIRD-PARTY.md) has the copyright notices and the written
offer of corresponding source.
