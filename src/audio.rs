//! Audio sinks. Each accepts float32 mono chunks as they are synthesised.

use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

pub trait Sink {
    fn write(&mut self, chunk: &[f32]) -> Result<()>;
    /// Finish normally, letting any buffered audio play out.
    fn close(&mut self) -> Result<()>;
    /// Give up now, discarding buffered audio (used on Ctrl-C).
    fn stop(&mut self);
}

/// Convert a float sample to 16-bit PCM the way libsndfile does: scale by
/// 32768, round towards negative infinity, and clamp.
///
/// The Python version used libsndfile for files but a plain `* 32767` cast for
/// the stdout stream; the two disagreed by a bit or two. This uses the
/// libsndfile convention for both, which uses the full negative range and
/// keeps the two outputs identical.
fn to_int16(sample: f32) -> i16 {
    let scaled = (sample.clamp(-1.0, 1.0) * 32768.0).floor();
    scaled.clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

/// Plays chunks through an output device.
///
/// Synthesis and playback run concurrently: the producer hands a chunk to the
/// audio callback through a bounded channel and goes straight back to
/// generating the next one. The channel is deliberately shallow — synthesis
/// outruns playback by roughly 3x, so an unbounded one would race ahead and
/// hold an entire document's audio in memory. Blocking on a full channel
/// applies backpressure instead.
pub struct SpeakerSink {
    _stream: cpal::Stream,
    sender: Option<SyncSender<Vec<f32>>>,
    state: Arc<PlaybackState>,
    interrupt: Arc<AtomicBool>,
}

/// Shared between the producer and the audio callback.
struct PlaybackState {
    /// Set once the producer will send nothing further.
    finished: AtomicBool,
    /// Set when the callback has drained everything after `finished`.
    drained: AtomicBool,
    /// Set to abandon playback immediately.
    stopping: AtomicBool,
    /// An error raised by the audio backend, surfaced on the next write.
    error: Mutex<Option<String>>,
}

/// How deep the handoff queue is, in chunks.
const QUEUE_DEPTH: usize = 4;
/// How often blocking waits re-check for an interrupt.
const POLL: Duration = Duration::from_millis(100);

impl SpeakerSink {
    pub fn new(sample_rate: u32, device: Option<&str>, interrupt: Arc<AtomicBool>) -> Result<Self> {
        let host = cpal::default_host();
        let device = match device {
            Some(spec) => find_device(&host, spec)?,
            None => host
                .default_output_device()
                .context("no default output device")?,
        };

        let config = output_config(&device, sample_rate)?;
        let channels = config.channels as usize;

        let (sender, receiver) = std::sync::mpsc::sync_channel::<Vec<f32>>(QUEUE_DEPTH);
        let state = Arc::new(PlaybackState {
            finished: AtomicBool::new(false),
            drained: AtomicBool::new(false),
            stopping: AtomicBool::new(false),
            error: Mutex::new(None),
        });

        let mut playback = Playback {
            receiver,
            current: Vec::new(),
            position: 0,
        };
        let callback_state = Arc::clone(&state);
        let error_state = Arc::clone(&state);

        let stream = device
            .build_output_stream(
                config,
                move |output: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    playback.fill(output, channels, &callback_state);
                },
                move |err| {
                    let mut slot = error_state.error.lock().unwrap();
                    if slot.is_none() {
                        *slot = Some(err.to_string());
                    }
                    error_state.stopping.store(true, Ordering::SeqCst);
                },
                None,
            )
            .context("opening the output stream")?;
        stream.play().context("starting playback")?;

        Ok(Self {
            _stream: stream,
            sender: Some(sender),
            state,
            interrupt,
        })
    }

    fn check_error(&self) -> Result<()> {
        if let Some(err) = self.state.error.lock().unwrap().as_ref() {
            bail!("audio output failed: {err}");
        }
        Ok(())
    }

    fn interrupted(&self) -> bool {
        self.interrupt.load(Ordering::SeqCst) || self.state.stopping.load(Ordering::SeqCst)
    }
}

impl Sink for SpeakerSink {
    fn write(&mut self, chunk: &[f32]) -> Result<()> {
        self.check_error()?;
        let Some(sender) = self.sender.as_ref() else {
            return Ok(());
        };

        let mut payload = chunk.to_vec();
        // Retry rather than block indefinitely, so Ctrl-C is noticed while the
        // queue is full — which is most of the time, playback being the slower
        // of the two.
        loop {
            if self.interrupted() {
                return self.check_error();
            }
            match sender.try_send(payload) {
                Ok(()) => return Ok(()),
                Err(TrySendError::Full(returned)) => {
                    payload = returned;
                    std::thread::sleep(POLL);
                }
                Err(TrySendError::Disconnected(_)) => return self.check_error(),
            }
        }
    }

    fn close(&mut self) -> Result<()> {
        // Dropping the sender lets the callback see the channel end.
        self.sender = None;
        self.state.finished.store(true, Ordering::SeqCst);

        // Wait for the queued audio to actually reach the speakers, but give up
        // promptly if interrupted.
        while !self.state.drained.load(Ordering::SeqCst) && !self.interrupted() {
            std::thread::sleep(POLL);
        }
        if self.interrupted() {
            self.stop();
        } else {
            // The device holds a buffer beyond what the callback has consumed;
            // let it play out rather than clipping the final syllable.
            std::thread::sleep(POLL * 2);
        }
        self.check_error()
    }

    fn stop(&mut self) {
        self.state.stopping.store(true, Ordering::SeqCst);
        self.sender = None;
    }
}

/// The audio callback's side of the handoff.
struct Playback {
    receiver: Receiver<Vec<f32>>,
    current: Vec<f32>,
    position: usize,
}

impl Playback {
    fn fill(&mut self, output: &mut [f32], channels: usize, state: &PlaybackState) {
        if state.stopping.load(Ordering::SeqCst) {
            output.fill(0.0);
            state.drained.store(true, Ordering::SeqCst);
            return;
        }

        for frame in output.chunks_mut(channels) {
            if self.position >= self.current.len() {
                match self.receiver.try_recv() {
                    Ok(chunk) => {
                        self.current = chunk;
                        self.position = 0;
                    }
                    Err(_) => {
                        // Nothing queued: silence. If the producer has finished
                        // and the channel is empty, playback is complete.
                        frame.fill(0.0);
                        if state.finished.load(Ordering::SeqCst) {
                            state.drained.store(true, Ordering::SeqCst);
                        }
                        continue;
                    }
                }
            }
            let sample = self.current[self.position];
            self.position += 1;
            // Mono is duplicated across whatever channel count the device wants.
            frame.fill(sample);
        }
    }
}

fn output_config(device: &cpal::Device, sample_rate: u32) -> Result<cpal::StreamConfig> {
    let rate: cpal::SampleRate = sample_rate;
    let supported = device
        .supported_output_configs()
        .context("querying output configurations")?
        .filter(|c| c.sample_format() == cpal::SampleFormat::F32)
        .find(|c| c.min_sample_rate() <= rate && rate <= c.max_sample_rate())
        .map(|c| c.with_sample_rate(rate));

    let config = match supported {
        Some(config) => config,
        // The model's 24 kHz is not universally supported; fall back to the
        // device default rather than failing. CoreAudio resamples for us.
        None => device
            .default_output_config()
            .context("no usable output configuration")?,
    };
    let mut config: cpal::StreamConfig = config.into();
    // A generous buffer: an underrun mid-sentence is far more audible than the
    // small extra latency this costs.
    config.buffer_size = cpal::BufferSize::Default;
    Ok(config)
}

fn find_device(host: &cpal::Host, spec: &str) -> Result<cpal::Device> {
    let devices: Vec<cpal::Device> = host
        .output_devices()
        .context("listing output devices")?
        .collect();

    if let Ok(index) = spec.parse::<usize>() {
        return devices
            .into_iter()
            .nth(index)
            .with_context(|| format!("no output device with index {index}"));
    }
    let lowered = spec.to_lowercase();
    devices
        .into_iter()
        .find(|d| device_name(d).to_lowercase().contains(&lowered))
        .with_context(|| format!("no output device matching {spec:?}"))
}

fn device_name(device: &cpal::Device) -> String {
    device
        .description()
        .map(|d| d.name().to_string())
        .unwrap_or_else(|_| "<unknown>".into())
}

/// Print the output devices, as `--list-devices` does.
pub fn list_devices() -> Result<()> {
    let host = cpal::default_host();
    let default = host.default_output_device().map(|d| device_name(&d));
    for (i, device) in host
        .output_devices()
        .context("listing output devices")?
        .enumerate()
    {
        let name = device_name(&device);
        let marker = if Some(&name) == default.as_ref() {
            "*"
        } else {
            " "
        };
        let rate = device
            .default_output_config()
            .map(|c| format!("{} Hz", c.sample_rate()))
            .unwrap_or_else(|_| "unavailable".into());
        println!("{marker} {i}  {name}  ({rate})");
    }
    Ok(())
}

/// Streams chunks into a WAV file, growing it as audio is produced.
pub struct FileSink {
    writer: Option<hound::WavWriter<std::io::BufWriter<std::fs::File>>>,
}

impl FileSink {
    pub fn new(path: &Path, sample_rate: u32) -> Result<Self> {
        if !path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("wav"))
        {
            bail!(
                "only WAV output is supported, but {} was requested",
                path.display()
            );
        }
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let writer = hound::WavWriter::create(path, spec)
            .with_context(|| format!("creating {}", path.display()))?;
        Ok(Self {
            writer: Some(writer),
        })
    }
}

impl Sink for FileSink {
    fn write(&mut self, chunk: &[f32]) -> Result<()> {
        let Some(writer) = self.writer.as_mut() else {
            return Ok(());
        };
        for &sample in chunk {
            writer.write_sample(to_int16(sample))?;
        }
        // Keep the header's length fields current, so a reader looking at a
        // partial file sees the audio written so far.
        writer.flush()?;
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        if let Some(writer) = self.writer.take() {
            writer.finalize().context("finalising the WAV file")?;
        }
        Ok(())
    }

    // Audio already written stays on disk; a partial file is the useful outcome.
    fn stop(&mut self) {
        let _ = self.close();
    }
}

/// Writes a streaming WAV to a non-seekable stream (stdout).
///
/// The RIFF sizes are unknown up front, so they are left at 0xFFFFFFFF — the
/// convention players use for piped WAV, and what `ffplay -` expects.
pub struct StdoutWavSink {
    stream: std::io::Stdout,
    closed: bool,
}

const UNKNOWN: u32 = 0xFFFF_FFFF;

impl StdoutWavSink {
    pub fn new(sample_rate: u32) -> Result<Self> {
        let mut stream = std::io::stdout();
        stream.write_all(&wav_header(sample_rate, 1, 16))?;
        stream.flush()?;
        Ok(Self {
            stream,
            closed: false,
        })
    }
}

fn wav_header(sample_rate: u32, channels: u16, bits: u16) -> Vec<u8> {
    let block_align = channels * bits / 8;
    let byte_rate = sample_rate * block_align as u32;
    let mut header = Vec::with_capacity(44);
    header.extend_from_slice(b"RIFF");
    header.extend_from_slice(&UNKNOWN.to_le_bytes());
    header.extend_from_slice(b"WAVE");
    header.extend_from_slice(b"fmt ");
    header.extend_from_slice(&16u32.to_le_bytes());
    header.extend_from_slice(&1u16.to_le_bytes()); // PCM
    header.extend_from_slice(&channels.to_le_bytes());
    header.extend_from_slice(&sample_rate.to_le_bytes());
    header.extend_from_slice(&byte_rate.to_le_bytes());
    header.extend_from_slice(&block_align.to_le_bytes());
    header.extend_from_slice(&bits.to_le_bytes());
    header.extend_from_slice(b"data");
    header.extend_from_slice(&UNKNOWN.to_le_bytes());
    header
}

impl Sink for StdoutWavSink {
    fn write(&mut self, chunk: &[f32]) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        let mut bytes = Vec::with_capacity(chunk.len() * 2);
        for &sample in chunk {
            bytes.extend_from_slice(&to_int16(sample).to_le_bytes());
        }
        self.stream.write_all(&bytes)?;
        self.stream.flush()?;
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        self.closed = true;
        // A closed pipe is how `| head` and friends end; not an error.
        if let Err(e) = self.stream.flush()
            && e.kind() != std::io::ErrorKind::BrokenPipe
        {
            return Err(e.into());
        }
        Ok(())
    }

    fn stop(&mut self) {
        let _ = self.close();
    }
}

/// Open the sink implied by `output`: the speakers, a file, or stdout.
pub fn open_sink(
    output: Option<&str>,
    sample_rate: u32,
    device: Option<&str>,
    interrupt: Arc<AtomicBool>,
) -> Result<Box<dyn Sink>> {
    Ok(match output {
        None => Box::new(SpeakerSink::new(sample_rate, device, interrupt)?),
        Some("-") => Box::new(StdoutWavSink::new(sample_rate)?),
        Some(path) => Box::new(FileSink::new(Path::new(path), sample_rate)?),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_declares_unknown_lengths() {
        let header = wav_header(24_000, 1, 16);
        assert_eq!(header.len(), 44);
        assert_eq!(&header[0..4], b"RIFF");
        assert_eq!(&header[4..8], &UNKNOWN.to_le_bytes());
        assert_eq!(&header[8..12], b"WAVE");
        assert_eq!(&header[40..44], &UNKNOWN.to_le_bytes());
        // Byte rate and block align for 24 kHz mono 16-bit.
        assert_eq!(
            u32::from_le_bytes(header[28..32].try_into().unwrap()),
            48_000
        );
        assert_eq!(u16::from_le_bytes(header[32..34].try_into().unwrap()), 2);
    }

    #[test]
    fn samples_are_clamped_before_conversion() {
        // Values checked against libsndfile via soundfile.
        assert_eq!(to_int16(0.0), 0);
        assert_eq!(to_int16(0.5), 16384);
        assert_eq!(to_int16(-0.123456), -4046);
        assert_eq!(to_int16(1.0), 32767);
        assert_eq!(to_int16(-1.0), -32768);
        // The model occasionally overshoots; wrapping would be an audible click.
        assert_eq!(to_int16(9.0), 32767);
        assert_eq!(to_int16(-9.0), -32768);
    }

    #[test]
    fn file_sink_rejects_formats_it_cannot_write() {
        let err = match FileSink::new(Path::new("out.flac"), 24_000) {
            Err(e) => e,
            Ok(_) => panic!("expected a FLAC path to be rejected"),
        };
        assert!(err.to_string().contains("only WAV output"), "{err}");
    }
}
