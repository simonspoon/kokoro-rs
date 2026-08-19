//! kokoro-rs — speak text with the Kokoro v1.0 TTS model, streaming as it goes.

use std::io::{BufRead, IsTerminal, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::Result;
use clap::Parser;

use crate::audio::{self, Sink};
use crate::models;
use crate::phonemes::{self, MAX_PHONEME_LENGTH};
use crate::synth::{SAMPLE_RATE, Synthesiser};
use crate::text::{
    self, Budget, CHUNK_PHONEMES, Chunk, ChunkStream, FIRST_CHUNK_CHARS, FIRST_CHUNK_PHONEMES,
};

const DEFAULT_VOICE: &str = "af_heart";
const DEFAULT_LANG: &str = "en-us";

#[derive(Parser)]
#[command(
    name = "kokoro-rs",
    about = "Speak text with the Kokoro v1.0 TTS model. \
             Reads from arguments or stdin; plays to the speakers unless -o is given.",
    after_help = "examples:\n  \
        kokoro-rs 'hello there'\n  \
        echo 'from a pipe' | kokoro-rs\n  \
        kokoro-rs -o out.wav < book.txt\n  \
        kokoro-rs -v bm_george -s 1.15 'a little faster'\n  \
        tail -f app.log | kokoro-rs\n",
    version,
    // So `--gap -1` reaches our own range check with the message the Python
    // version gave, rather than being rejected as an unknown flag.
    allow_negative_numbers = true
)]
pub struct Args {
    /// Text to speak (default: read stdin)
    text: Vec<String>,

    /// Write audio to FILE instead of the speakers ('-' streams WAV to stdout)
    #[arg(short, long, value_name = "FILE")]
    output: Option<String>,

    /// Voice to use
    #[arg(short, long, default_value = DEFAULT_VOICE)]
    voice: String,

    /// Speech rate between 0.5 and 2.0
    #[arg(short, long, default_value_t = 1.0, value_name = "RATE")]
    speed: f32,

    /// Language passed to the phonemiser
    #[arg(short, long, default_value = DEFAULT_LANG)]
    lang: String,

    /// Output audio device (name or index)
    #[arg(short, long)]
    device: Option<String>,

    /// List available voices and exit
    #[arg(long)]
    list_voices: bool,

    /// List audio devices and exit
    #[arg(long)]
    list_devices: bool,

    /// Optional cap on the characters per synthesis chunk
    #[arg(long, value_name = "N")]
    chunk_chars: Option<usize>,

    /// Max phonemes per synthesis chunk (the model reads 510)
    #[arg(long, default_value_t = CHUNK_PHONEMES, value_name = "N")]
    chunk_phonemes: usize,

    /// Silence inserted between sentences
    #[arg(long, default_value_t = 0.12, value_name = "SECONDS")]
    gap: f32,

    /// Fail instead of fetching missing model files
    #[arg(long)]
    no_download: bool,

    /// Suppress progress output
    #[arg(short, long)]
    quiet: bool,
}

/// Exit codes, matching the Python original.
const OK: i32 = 0;
const NOTHING_SPOKEN: i32 = 1;
const USAGE: i32 = 2;
const INTERRUPTED: i32 = 130;

pub fn main() -> i32 {
    let args = Args::parse();
    match run(args) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("kokoro-rs: {e:#}");
            NOTHING_SPOKEN
        }
    }
}

/// Turn Ctrl-C into a flag rather than an unwind.
///
/// The synthesis loop sits on top of ONNX Runtime and CoreAudio, both of which
/// hold native threads and buffers that must be torn down in order. The signal
/// just sets a flag that the loop and the audio sink poll.
fn install_interrupt_handler() -> Arc<AtomicBool> {
    let interrupt = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&interrupt);
    // A second Ctrl-C should still kill a wedged process, so leave the default
    // handler in place after the first.
    let _ = ctrlc::set_handler(move || {
        if flag.swap(true, Ordering::SeqCst) {
            std::process::exit(INTERRUPTED);
        }
    });
    interrupt
}

fn run(args: Args) -> Result<i32> {
    let interrupt = install_interrupt_handler();

    if args.list_devices {
        audio::list_devices()?;
        return Ok(OK);
    }

    if !(0.5..=2.0).contains(&args.speed) {
        eprintln!("kokoro-rs: --speed must be between 0.5 and 2.0");
        return Ok(USAGE);
    }
    if args.chunk_chars.is_some_and(|n| n < 20) {
        eprintln!("kokoro-rs: --chunk-chars must be at least 20");
        return Ok(USAGE);
    }
    if args.chunk_phonemes < 20 {
        eprintln!("kokoro-rs: --chunk-phonemes must be at least 20");
        return Ok(USAGE);
    }
    if args.chunk_phonemes > MAX_PHONEME_LENGTH {
        eprintln!("kokoro-rs: --chunk-phonemes cannot exceed {MAX_PHONEME_LENGTH}");
        return Ok(USAGE);
    }
    if args.gap < 0.0 {
        eprintln!("kokoro-rs: --gap cannot be negative");
        return Ok(USAGE);
    }

    // Both must be in place before the model loads or the first phonemisation.
    models::ensure_espeak_data()?;
    models::ensure_onnxruntime(!args.no_download)?;
    let (model_path, voices_path) = models::resolve_model_files(!args.no_download)?;
    if interrupt.load(Ordering::SeqCst) {
        return Ok(INTERRUPTED);
    }
    let mut synth = Synthesiser::load(&model_path, &voices_path)?;

    if args.list_voices {
        for name in synth.voices().names() {
            println!("{name}");
        }
        return Ok(OK);
    }
    if !synth.voices().contains(&args.voice) {
        eprintln!(
            "kokoro-rs: unknown voice {:?} (try --list-voices)",
            args.voice
        );
        return Ok(USAGE);
    }

    // Chunks are measured in phonemes, so the chunker does the phonemising and
    // hands the result to the synthesiser rather than it being done twice.
    let lang = args.lang.clone();
    let first = Budget {
        phonemes: FIRST_CHUNK_PHONEMES.min(args.chunk_phonemes),
        chars: args.chunk_chars.map(|c| FIRST_CHUNK_CHARS.min(c)),
    };
    let rest = Budget {
        phonemes: args.chunk_phonemes,
        chars: args.chunk_chars,
    };
    let mut stream = ChunkStream::new(first, rest, move |text: &str| {
        phonemes::phonemize(text, &lang)
    });

    let from_stdin = args.text.is_empty();
    if from_stdin && std::io::stdin().is_terminal() {
        eprintln!(
            "kokoro-rs: no text given; pass it as an argument or pipe it in (kokoro-rs --help)"
        );
        return Ok(USAGE);
    }

    let mut speaker = Speaker::new(&args, Arc::clone(&interrupt));

    if from_stdin {
        // Read line by line so streamed input is spoken as it arrives.
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            let mut line = line?;
            line.push('\n');
            for chunk in stream.push(&line)? {
                if !speaker.speak(&mut synth, &chunk)? {
                    break;
                }
            }
            if speaker.interrupted {
                break;
            }
        }
    } else {
        for chunk in stream.push(&args.text.join(" "))? {
            if !speaker.speak(&mut synth, &chunk)? {
                break;
            }
        }
    }
    if !speaker.interrupted {
        for chunk in stream.finish()? {
            if !speaker.speak(&mut synth, &chunk)? {
                break;
            }
        }
    }

    speaker.finish()
}

/// Synthesises chunks and feeds them to the sink, tracking progress.
struct Speaker<'a> {
    args: &'a Args,
    interrupt: Arc<AtomicBool>,
    sink: Option<Box<dyn Sink>>,
    verbose: bool,
    interrupted: bool,
    /// Whether the previous chunk ended a sentence, and so earns a gap.
    pending_gap: bool,
    spoken: f64,
    start: Instant,
    first_audio: Option<f64>,
}

impl<'a> Speaker<'a> {
    fn new(args: &'a Args, interrupt: Arc<AtomicBool>) -> Self {
        Self {
            verbose: !args.quiet && std::io::stderr().is_terminal(),
            args,
            interrupt,
            sink: None,
            interrupted: false,
            pending_gap: false,
            spoken: 0.0,
            start: Instant::now(),
            first_audio: None,
        }
    }

    /// Synthesise and play one chunk. Returns false once interrupted.
    fn speak(&mut self, synth: &mut Synthesiser, chunk: &Chunk) -> Result<bool> {
        if self.interrupt.load(Ordering::SeqCst) {
            self.interrupted = true;
            return Ok(false);
        }
        let samples =
            synth.create_from_phonemes(&chunk.phonemes, &self.args.voice, self.args.speed)?;
        if self.interrupt.load(Ordering::SeqCst) {
            self.interrupted = true;
            return Ok(false);
        }
        if samples.is_empty() {
            // Nothing in this chunk survived phonemisation — punctuation only.
            return Ok(true);
        }

        // The sink is opened only once there is audio to put in it, so a run
        // that produces nothing does not create an empty file or grab the
        // audio device.
        if self.sink.is_none() {
            self.sink = Some(audio::open_sink(
                self.args.output.as_deref(),
                SAMPLE_RATE,
                self.args.device.as_deref(),
                Arc::clone(&self.interrupt),
            )?);
            self.first_audio = Some(self.start.elapsed().as_secs_f64());
        }
        let sink = self.sink.as_mut().expect("sink was just opened");

        // Synthesising each chunk separately clips the pause the model would
        // have left between sentences; put it back.
        if self.pending_gap && self.args.gap > 0.0 {
            let frames = (self.args.gap * SAMPLE_RATE as f32) as usize;
            sink.write(&vec![0.0; frames])?;
            self.spoken += frames as f64 / SAMPLE_RATE as f64;
        }
        sink.write(&samples)?;
        self.pending_gap = text::ends_sentence(&chunk.text);
        self.spoken += samples.len() as f64 / SAMPLE_RATE as f64;

        if self.verbose {
            eprint!(
                "\r\x1b[Kkokoro-rs: {:5.1}s audio  (first sound in {:.2}s)",
                self.spoken,
                self.first_audio.unwrap_or(0.0)
            );
            let _ = std::io::stderr().flush();
        }
        Ok(true)
    }

    fn finish(mut self) -> Result<i32> {
        if self.verbose && self.first_audio.is_some() {
            eprintln!();
        }
        let Some(mut sink) = self.sink.take() else {
            if self.interrupted || self.interrupt.load(Ordering::SeqCst) {
                return Ok(INTERRUPTED);
            }
            eprintln!("kokoro-rs: no speakable text in input");
            return Ok(NOTHING_SPOKEN);
        };

        if self.interrupted {
            sink.stop();
            return Ok(INTERRUPTED);
        }
        // close() plays out what is queued, but returns early if Ctrl-C
        // arrives during the drain.
        sink.close()?;
        if self.interrupt.load(Ordering::SeqCst) {
            return Ok(INTERRUPTED);
        }
        Ok(OK)
    }
}
