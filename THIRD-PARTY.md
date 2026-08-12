# Third-party components

## espeak-ng — GPL-3.0-or-later

kokoro-rs uses espeak-ng as its phonemiser. Two pieces of it travel with this
project:

- **The library itself**, compiled from source and linked statically into the
  binary by the [`espeak-rs-sys`](https://crates.io/crates/espeak-rs-sys) crate,
  which vendors the espeak-ng sources.
- **Its data directory** — compiled dictionaries, phoneme tables, intonation
  tables and voice definitions — vendored here as
  `assets/espeak-ng-data.tar.gz` and embedded into the binary.

espeak-ng is:

> Copyright (C) 2005 to 2013 by Jonathan Duddington
> Copyright (C) 2013-2017 Reece H. Dunn
> Copyright (C) 2015-2024 by the espeak-ng contributors

licensed under the GNU General Public License, version 3 or later. Its full
text is in [LICENSE](LICENSE), which is this project's licence for the same
reason: linking it makes kokoro-rs a derivative work.

Upstream source: <https://github.com/espeak-ng/espeak-ng>

**Written offer of corresponding source.** The espeak-ng source corresponding
to the binary is the copy vendored in `espeak-rs-sys` 0.2.0, obtainable from
<https://crates.io/crates/espeak-rs-sys/0.2.0> and from the upstream repository
above. The exact revision built is whatever that crate version pins; the
version is recorded in `Cargo.lock`.

## Components downloaded at runtime

Neither is bundled or redistributed here — both are fetched from their
publishers on first run, into `~/.cache/kokoro-rs`.

| Component | Licence | Source |
| --- | --- | --- |
| Kokoro-82M model and voice packs | Apache-2.0 | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), via the [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) release |
| ONNX Runtime shared library | MIT | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) |

## Rust dependencies

The crates kokoro-rs builds against are permissively licensed (MIT, Apache-2.0
or MIT/Apache-2.0 dual), with `espeak-rs`/`espeak-rs-sys` the exception noted
above: the binding crates are MIT, but the espeak-ng code they compile is not.
`cargo tree` lists the full set.
