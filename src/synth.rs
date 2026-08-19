//! Running the Kokoro model.

use std::path::Path;

use anyhow::{Context, Result, bail};
use ort::session::Session;
use ort::value::Tensor;

use crate::phonemes::{self, MAX_PHONEME_LENGTH};
use crate::trim;
use crate::voices::Voices;

/// Kokoro v1.0 always outputs at this rate.
pub const SAMPLE_RATE: u32 = 24_000;

pub struct Synthesiser {
    session: Session,
    voices: Voices,
    /// Newer exports of the model name the token input `input_ids`.
    token_input: &'static str,
}

impl Synthesiser {
    pub fn load(model_path: &Path, voices_path: &Path) -> Result<Self> {
        let session = Session::builder()
            .context("creating an ONNX session builder")?
            .commit_from_file(model_path)
            .with_context(|| format!("loading {}", model_path.display()))?;

        let token_input = if session.inputs().iter().any(|i| i.name() == "input_ids") {
            "input_ids"
        } else {
            "tokens"
        };

        let voices = Voices::load(voices_path)?;
        Ok(Self {
            session,
            voices,
            token_input,
        })
    }

    pub fn voices(&self) -> &Voices {
        &self.voices
    }

    /// Synthesise one chunk, returning mono float32 samples.
    ///
    /// The chunker has already phonemised the text to measure it against the
    /// model's context, so the phonemes come in ready-made rather than being
    /// derived from the text a second time.
    pub fn create_from_phonemes(
        &mut self,
        phonemes: &str,
        voice: &str,
        speed: f32,
    ) -> Result<Vec<f32>> {
        let mut audio = Vec::new();
        for batch in split_phonemes(phonemes) {
            let part = self.create_batch(&batch, voice, speed)?;
            // Each batch is synthesised separately and padded with silence by
            // the model; trimming it keeps the joins from sounding halting.
            audio.extend(trim::trim(part));
        }
        Ok(audio)
    }

    fn create_batch(&mut self, phonemes: &str, voice: &str, speed: f32) -> Result<Vec<f32>> {
        let tokens = phonemes::tokenize(phonemes);
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        debug_assert!(tokens.len() <= MAX_PHONEME_LENGTH);

        // The style vector is chosen by token count, before the padding below.
        let style = self.voices.style(voice, tokens.len())?;

        // The model expects a pad token either side of the phonemes.
        let mut padded = Vec::with_capacity(tokens.len() + 2);
        padded.push(0i64);
        padded.extend_from_slice(&tokens);
        padded.push(0i64);

        let token_tensor = Tensor::from_array((vec![1, padded.len() as i64], padded))?;
        let style_tensor = Tensor::from_array((vec![1, style.len() as i64], style))?;
        let speed_tensor = Tensor::from_array((vec![1i64], vec![speed]))?;

        let outputs = self
            .session
            .run(ort::inputs![
                self.token_input => token_tensor,
                "style" => style_tensor,
                "speed" => speed_tensor,
            ])
            .context("running the model")?;

        let (_, audio) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("reading the model's audio output")?;
        if audio.is_empty() {
            bail!("the model returned no audio for {phonemes:?}");
        }
        Ok(audio.to_vec())
    }
}

/// Split a phoneme string into batches that fit the model's context.
///
/// Chunks are already sized so this rarely fires, but a chunk dense in short
/// words can still exceed 510 phonemes. Splits land on punctuation where
/// possible, then on word boundaries, since a break mid-word is audible.
fn split_phonemes(phonemes: &str) -> Vec<String> {
    const BREAKS: [char; 5] = ['.', ',', '!', '?', ';'];

    // Split on punctuation, keeping the marks as their own parts.
    let mut parts: Vec<&str> = Vec::new();
    let mut prev = 0;
    for (i, c) in phonemes.char_indices() {
        if BREAKS.contains(&c) {
            parts.push(&phonemes[prev..i]);
            parts.push(&phonemes[i..i + c.len_utf8()]);
            prev = i + c.len_utf8();
        }
    }
    parts.push(&phonemes[prev..]);

    let mut batches = Vec::new();
    let mut current = String::new();
    for part in parts
        .into_iter()
        .flat_map(|part| split_run(part, MAX_PHONEME_LENGTH))
    {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        // Lengths are in characters, as in the Python original.
        if current.chars().count() + part.chars().count() + 1 >= MAX_PHONEME_LENGTH {
            // A part that fills the context on its own arrives with nothing
            // ahead of it, and an empty batch would synthesise nothing.
            if !current.is_empty() {
                batches.push(current.trim().to_string());
            }
            current = part.to_string();
        } else if part.chars().count() == 1 && BREAKS.contains(&part.chars().next().unwrap()) {
            current.push_str(part);
        } else {
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(part);
        }
    }
    if !current.is_empty() {
        batches.push(current.trim().to_string());
    }
    batches
}

/// Break a run of phonemes with no punctuation in it into pieces of at most
/// `limit` characters, at word boundaries where possible.
///
/// Text expands into phonemes unevenly — "1234567" becomes forty of them — so a
/// chunk sized in characters can still hold a punctuation-free run longer than
/// the context. Left whole, such a run is truncated by [`phonemes::tokenize`]
/// and the rest of the sentence is never spoken.
fn split_run(run: &str, limit: usize) -> Vec<&str> {
    if run.chars().count() <= limit {
        return vec![run];
    }

    let mut pieces = Vec::new();
    // Byte offset where the piece being built starts, how many characters it
    // holds, and where its last word break was.
    let mut start = 0;
    let mut count = 0;
    let mut space: Option<usize> = None;
    for (i, c) in run.char_indices() {
        if count == limit {
            // A cut mid-word is audible, so fall back to one only when the
            // piece is a single word longer than the whole context.
            let cut = space.unwrap_or(i);
            pieces.push(&run[start..cut]);
            start = if cut == i { i } else { cut + 1 };
            count = run[start..i].chars().count();
            space = None;
        }
        if c == ' ' {
            space = Some(i);
        }
        count += 1;
    }
    pieces.push(&run[start..]);
    pieces
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_phonemes_stay_in_one_batch() {
        assert_eq!(split_phonemes("həlˈoʊ fɹʌm ɹˈʌst."), ["həlˈoʊ fɹʌm ɹˈʌst."]);
    }

    #[test]
    fn punctuation_stays_attached_to_the_preceding_phonemes() {
        assert_eq!(split_phonemes("wˌʌn. tˈuː."), ["wˌʌn. tˈuː."]);
    }

    #[test]
    fn over_long_phonemes_are_split_at_punctuation() {
        let clause = "a".repeat(300);
        let batches = split_phonemes(&format!("{clause}. {clause}."));
        assert_eq!(batches.len(), 2);
        assert!(
            batches
                .iter()
                .all(|b| b.chars().count() < MAX_PHONEME_LENGTH)
        );
        assert!(batches[0].ends_with('.'));
    }

    #[test]
    fn over_long_runs_without_punctuation_are_split_at_words() {
        // Numbers and dates expand several-fold into phonemes, so a chunk sized
        // in characters can hold more than the model can read in one pass.
        let word = "wˈʌn ";
        let run = word.repeat(200);
        let batches = split_phonemes(run.trim());
        assert!(batches.len() > 1);
        assert!(
            batches
                .iter()
                .all(|b| b.chars().count() <= MAX_PHONEME_LENGTH),
            "{:?}",
            batches
                .iter()
                .map(|b| b.chars().count())
                .collect::<Vec<_>>()
        );
        // Nothing is dropped and no word is cut in half.
        assert_eq!(batches.join(" "), run.trim());
    }

    #[test]
    fn a_single_over_long_word_is_cut() {
        let run = "a".repeat(MAX_PHONEME_LENGTH * 2 + 5);
        let batches = split_phonemes(&run);
        assert_eq!(batches.len(), 3);
        assert!(
            batches
                .iter()
                .all(|b| b.chars().count() <= MAX_PHONEME_LENGTH)
        );
        assert_eq!(batches.concat(), run);
    }

    #[test]
    fn empty_input_produces_no_batches() {
        assert!(split_phonemes("").is_empty());
        assert!(split_phonemes("   ").is_empty());
    }
}
