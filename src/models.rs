//! Locating and fetching the files the model needs at runtime.
//!
//! Three kinds of asset end up in the cache directory:
//!
//! * the Kokoro ONNX model and its voice pack, downloaded on first use;
//! * the ONNX Runtime shared library, likewise downloaded — `ort` publishes no
//!   prebuilt binaries for x86_64 macOS, so we fetch Microsoft's own release
//!   and load it dynamically;
//! * espeak-ng's data directory, which is embedded in this binary and unpacked.
//!
//! After the first run nothing here touches the network.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

const RELEASE: &str =
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0";

pub const MODEL_FILE: &str = "kokoro-v1.0.onnx";
pub const VOICES_FILE: &str = "voices-v1.0.bin";

/// The last ONNX Runtime release with an official x86_64 macOS build, and the
/// version the Python project pinned for the same reason. Every other platform
/// we ship is published under the same version, so one pin covers all four.
const ORT_VERSION: &str = "1.23.2";

/// The platform tag in Microsoft's ONNX Runtime release archives.
///
/// A binary only ever needs its own platform's build, so this is chosen at
/// compile time; an unsupported target fails to build rather than fetching an
/// archive that cannot be loaded.
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
const ORT_PLATFORM: &str = "osx-x86_64";
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const ORT_PLATFORM: &str = "osx-arm64";
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
const ORT_PLATFORM: &str = "linux-x64";
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
const ORT_PLATFORM: &str = "linux-aarch64";

/// The shared library's name inside that archive.
///
/// The two platforms disagree about more than the extension: macOS puts the
/// version before it (`libonnxruntime.1.23.2.dylib`), Linux after the soname
/// (`libonnxruntime.so.1.23.2`).
fn ort_lib_name() -> String {
    if cfg!(target_os = "macos") {
        format!("libonnxruntime.{ORT_VERSION}.dylib")
    } else {
        format!("libonnxruntime.so.{ORT_VERSION}")
    }
}

/// espeak-ng's dictionaries, phoneme tables and voice definitions.
///
/// Vendored rather than taken from the espeak-rs-sys build: cargo gives no
/// ordering guarantee that a dependency's build script has run before ours,
/// so reading its output from a build script is a race. Regenerate with
/// `scripts/vendor_espeak_data.sh`. Embedding it keeps the property the Python
/// version got from `espeakng-loader` — no system espeak-ng install required.
const ESPEAK_DATA: &[u8] = include_bytes!("../assets/espeak-ng-data.tar.gz");

pub fn cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("KOKORO_HOME") {
        return PathBuf::from(dir);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".cache").join("kokoro-rs")
}

/// Return (model_path, voices_path), downloading them on first use.
pub fn resolve_model_files(download: bool) -> Result<(PathBuf, PathBuf)> {
    let model = resolve("KOKORO_MODEL", MODEL_FILE, download)?;
    let voices = resolve("KOKORO_VOICES", VOICES_FILE, download)?;
    Ok((model, voices))
}

fn resolve(env_var: &str, filename: &str, download: bool) -> Result<PathBuf> {
    if let Ok(override_path) = std::env::var(env_var) {
        let path = expand_tilde(&override_path);
        if !path.is_file() {
            bail!("{env_var}={} does not exist", path.display());
        }
        return Ok(path);
    }

    let path = cache_dir().join(filename);
    if path.is_file() {
        return Ok(path);
    }
    if !download {
        bail!(
            "missing {}\n  download it from {RELEASE}/{filename}\n  \
             or run without --no-download to fetch it automatically",
            path.display()
        );
    }
    fetch(&format!("{RELEASE}/{filename}"), &path)?;
    Ok(path)
}

fn expand_tilde(path: &str) -> PathBuf {
    match path.strip_prefix("~/") {
        Some(rest) if std::env::var("HOME").is_ok() => {
            PathBuf::from(std::env::var("HOME").unwrap()).join(rest)
        }
        _ => PathBuf::from(path),
    }
}

/// Make espeak-ng's data available and point the library at it.
///
/// Must run before the first phonemisation: espeak-rs reads this environment
/// variable when it initialises the library, which it does only once. Callers
/// should invoke this early, before any threads exist; it is idempotent, so
/// `phonemes` can also call it defensively.
pub fn ensure_espeak_data() -> Result<()> {
    static ONCE: std::sync::OnceLock<std::result::Result<(), String>> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        let root = cache_dir();
        let data = root.join("espeak-ng-data");
        // `phontab` is required for initialisation, so its presence is a
        // reasonable signal that an earlier unpack ran to completion.
        if !data.join("phontab").is_file() {
            unpack_tar_gz(ESPEAK_DATA, &root)
                .context("unpacking the bundled espeak-ng data")
                .map_err(|e| format!("{e:#}"))?;
        }
        // SAFETY: the first call happens before any threads are spawned, and
        // the OnceLock makes every later call a no-op.
        unsafe { std::env::set_var("PIPER_ESPEAKNG_DATA_DIRECTORY", &root) };
        Ok(())
    })
    .as_ref()
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(())
}

/// Make the ONNX Runtime shared library available and point `ort` at it.
///
/// `ORT_DYLIB_PATH` is honoured if set, so a system install can be used.
pub fn ensure_onnxruntime(download: bool) -> Result<PathBuf> {
    if let Ok(path) = std::env::var("ORT_DYLIB_PATH") {
        let path = expand_tilde(&path);
        if !path.is_file() {
            bail!("ORT_DYLIB_PATH={} does not exist", path.display());
        }
        return Ok(path);
    }

    let dylib = cache_dir().join(ort_lib_name());
    if !dylib.is_file() {
        if !download {
            bail!(
                "missing {}\n  it is unpacked from the ONNX Runtime {ORT_VERSION} release\n  \
                 or run without --no-download to fetch it automatically",
                dylib.display()
            );
        }
        fetch_onnxruntime(&dylib)?;
    }
    // SAFETY: called before any threads are spawned.
    unsafe { std::env::set_var("ORT_DYLIB_PATH", &dylib) };
    Ok(dylib)
}

/// Download the ONNX Runtime release tarball and extract just the dylib.
fn fetch_onnxruntime(dest: &Path) -> Result<()> {
    let name = format!("onnxruntime-{ORT_PLATFORM}-{ORT_VERSION}");
    let url = format!(
        "https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/{name}.tgz"
    );
    let tarball = download_to_memory(&url, "onnxruntime")?;

    let decoder = flate2::read::GzDecoder::new(std::io::Cursor::new(tarball));
    let mut archive = tar::Archive::new(decoder);
    // Matched as a suffix: the macOS archives store paths as `./<name>/lib/...`
    // and the Linux ones as `<name>/lib/...`.
    let wanted = format!("lib/{}", ort_lib_name());

    for entry in archive
        .entries()
        .context("reading the ONNX Runtime archive")?
    {
        let mut entry = entry?;
        if !entry.path()?.to_string_lossy().ends_with(&wanted) {
            continue;
        }
        std::fs::create_dir_all(dest.parent().unwrap())?;
        let tmp = dest.with_extension("part");
        let mut out = std::fs::File::create(&tmp)?;
        std::io::copy(&mut entry, &mut out)?;
        out.sync_all()?;
        drop(out);
        std::fs::rename(&tmp, dest)?;
        return Ok(());
    }
    bail!("{wanted} was not present in {url}")
}

fn unpack_tar_gz(bytes: &[u8], dest: &Path) -> Result<()> {
    std::fs::create_dir_all(dest)?;
    let decoder = flate2::read::GzDecoder::new(std::io::Cursor::new(bytes));
    tar::Archive::new(decoder).unpack(dest)?;
    Ok(())
}

fn download_to_memory(url: &str, label: &str) -> Result<Vec<u8>> {
    eprintln!("kokoro-rs: downloading {label}");
    let response = ureq::get(url)
        .call()
        .with_context(|| format!("fetching {url}"))?;
    let total = content_length(&response);
    let mut reader = response.into_body().into_reader();
    let mut buf = Vec::new();
    copy_with_progress(&mut reader, &mut buf, total)?;
    Ok(buf)
}

/// Download `url` to `dest`, showing a progress bar on an interactive stderr.
fn fetch(url: &str, dest: &Path) -> Result<()> {
    std::fs::create_dir_all(dest.parent().unwrap())?;
    let name = dest
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    eprintln!(
        "kokoro-rs: downloading {name} -> {}",
        dest.parent().unwrap().display()
    );

    let response = ureq::get(url)
        .call()
        .with_context(|| format!("fetching {url}"))?;
    let total = content_length(&response);
    let mut reader = response.into_body().into_reader();

    // Write beside the target and rename, so an interrupted download never
    // leaves a truncated file that later runs would trust.
    let tmp = dest.with_extension("part");
    let mut file = std::fs::File::create(&tmp)?;
    let result = copy_with_progress(&mut reader, &mut file, total);
    if let Err(e) = result {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    file.sync_all()?;
    drop(file);
    std::fs::rename(&tmp, dest)?;
    Ok(())
}

fn content_length(response: &ureq::http::Response<ureq::Body>) -> Option<u64> {
    response
        .headers()
        .get("content-length")?
        .to_str()
        .ok()?
        .parse()
        .ok()
}

fn copy_with_progress(
    reader: &mut impl Read,
    writer: &mut impl Write,
    total: Option<u64>,
) -> Result<()> {
    let show_progress =
        total.is_some_and(|t| t > 0) && std::io::IsTerminal::is_terminal(&std::io::stderr());
    let mut buf = vec![0u8; 256 * 1024];
    let mut done: u64 = 0;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        writer.write_all(&buf[..n])?;
        done += n as u64;
        if show_progress {
            draw_progress(done, total.unwrap());
        }
    }
    if show_progress {
        eprintln!();
    }
    Ok(())
}

fn draw_progress(done: u64, total: u64) {
    const WIDTH: usize = 30;
    let done = done.min(total);
    let filled = (WIDTH as u64 * done / total) as usize;
    let bar: String = "#".repeat(filled) + &"-".repeat(WIDTH - filled);
    eprint!(
        "\r  [{bar}] {:5.1}%  {:6.1}/{:.1} MB",
        100.0 * done as f64 / total as f64,
        done as f64 / 1e6,
        total as f64 / 1e6,
    );
    let _ = std::io::stderr().flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_espeak_data_is_present() {
        // A truncated or empty archive would only surface at first run.
        assert!(ESPEAK_DATA.len() > 1_000_000, "{} bytes", ESPEAK_DATA.len());
    }

    #[test]
    fn cache_dir_follows_the_environment() {
        // KOKORO_HOME wins; otherwise the path sits under the user's cache.
        assert!(cache_dir().ends_with("kokoro-rs") || std::env::var("KOKORO_HOME").is_ok());
    }
}
