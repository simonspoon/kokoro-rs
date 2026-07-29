//! Turning text into the phoneme token IDs the model expects.
//!
//! Text is phonemised to IPA by espeak-ng, then each character is looked up in
//! Kokoro's 114-entry vocabulary. Characters outside it — espeak emits a few
//! the model was never trained on — are dropped, as in the reference
//! implementation.

use std::collections::HashMap;
use std::sync::OnceLock;

use anyhow::{Context, Result};

/// The model's context length, in phonemes.
pub const MAX_PHONEME_LENGTH: usize = 510;

/// Kokoro v1.0's phoneme vocabulary, copied from `kokoro_onnx/config.json`.
/// The IDs are not contiguous: they index the model's embedding table, and the
/// gaps are phonemes the released checkpoint does not use.
#[rustfmt::skip]
const VOCAB: &[(char, i64)] = &[
    (';', 1), (':', 2), (',', 3), ('.', 4), ('!', 5), ('?', 6), ('—', 9), ('…', 10),
    ('"', 11), ('(', 12), (')', 13), ('“', 14), ('”', 15), (' ', 16), ('\u{303}', 17),
    ('ʣ', 18), ('ʥ', 19), ('ʦ', 20), ('ʨ', 21), ('ᵝ', 22), ('ꭧ', 23), ('A', 24), ('I', 25),
    ('O', 31), ('Q', 33), ('S', 35), ('T', 36), ('W', 39), ('Y', 41), ('ᵊ', 42), ('a', 43),
    ('b', 44), ('c', 45), ('d', 46), ('e', 47), ('f', 48), ('h', 50), ('i', 51), ('j', 52),
    ('k', 53), ('l', 54), ('m', 55), ('n', 56), ('o', 57), ('p', 58), ('q', 59), ('r', 60),
    ('s', 61), ('t', 62), ('u', 63), ('v', 64), ('w', 65), ('x', 66), ('y', 67), ('z', 68),
    ('ɑ', 69), ('ɐ', 70), ('ɒ', 71), ('æ', 72), ('β', 75), ('ɔ', 76), ('ɕ', 77), ('ç', 78),
    ('ɖ', 80), ('ð', 81), ('ʤ', 82), ('ə', 83), ('ɚ', 85), ('ɛ', 86), ('ɜ', 87), ('ɟ', 90),
    ('ɡ', 92), ('ɥ', 99), ('ɨ', 101), ('ɪ', 102), ('ʝ', 103), ('ɯ', 110), ('ɰ', 111),
    ('ŋ', 112), ('ɳ', 113), ('ɲ', 114), ('ɴ', 115), ('ø', 116), ('ɸ', 118), ('θ', 119),
    ('œ', 120), ('ɹ', 123), ('ɾ', 125), ('ɻ', 126), ('ʁ', 128), ('ɽ', 129), ('ʂ', 130),
    ('ʃ', 131), ('ʈ', 132), ('ʧ', 133), ('ʊ', 135), ('ʋ', 136), ('ʌ', 138), ('ɣ', 139),
    ('ɤ', 140), ('χ', 142), ('ʎ', 143), ('ʒ', 147), ('ʔ', 148), ('ˈ', 156), ('ˌ', 157),
    ('ː', 158), ('ʰ', 162), ('ʲ', 164), ('↓', 169), ('→', 171), ('↗', 172), ('↘', 173),
    ('ᵻ', 177),
];

fn vocab() -> &'static HashMap<char, i64> {
    static VOCAB_MAP: OnceLock<HashMap<char, i64>> = OnceLock::new();
    VOCAB_MAP.get_or_init(|| VOCAB.iter().copied().collect())
}

/// Phonemise `text` to IPA, keeping only vocabulary characters.
///
/// Punctuation is hidden from espeak-ng and spliced back into the phonemes
/// afterwards — see [`crate::punctuation`]. espeak drops it otherwise, and
/// Kokoro relies on it for pauses and sentence-final intonation.
pub fn phonemize(text: &str, lang: &str) -> Result<String> {
    crate::models::ensure_espeak_data()?;

    let phonemes = crate::punctuation::phonemize_preserving(text, |chunk| espeak(chunk, lang))?;
    Ok(phonemes
        .chars()
        .filter(|c| vocab().contains_key(c))
        .collect::<String>()
        .trim()
        .to_string())
}

/// Run espeak-ng over one punctuation-free chunk, returning space-separated
/// words of IPA phonemes.
fn espeak(chunk: &str, lang: &str) -> Result<String> {
    // espeak-ng keeps its parser state in globals — the active voice among
    // them — so concurrent calls interleave and return empty or mixed output.
    static ESPEAK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = ESPEAK.lock().unwrap_or_else(|e| e.into_inner());

    if chunk.trim().is_empty() {
        return Ok(String::new());
    }
    let sentences = espeak_rs::text_to_phonemes(chunk, lang, Some('_'))
        .map_err(|e| anyhow::anyhow!("{e}"))
        .with_context(|| format!("phonemising {chunk:?} as {lang}"))?;

    // The '_' phoneme separator is requested only to match what phonemizer
    // asks espeak for; it is discarded, as phonemizer discards it. Whitespace
    // is normalised to a single space per word break.
    let joined = sentences.join(" ").replace('_', "");
    Ok(joined.split_whitespace().collect::<Vec<_>>().join(" "))
}

/// Map phonemes to token IDs, truncating to the model's context length.
pub fn tokenize(phonemes: &str) -> Vec<i64> {
    phonemes
        .chars()
        .take(MAX_PHONEME_LENGTH)
        .filter_map(|c| vocab().get(&c).copied())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocab_has_the_expected_size() {
        assert_eq!(vocab().len(), 114);
    }

    #[test]
    fn tokenize_maps_known_phonemes_and_drops_others() {
        assert_eq!(tokenize("hɛlˈoʊ"), [50, 86, 54, 156, 57, 135]);
        // Digits are outside the vocabulary and are dropped rather than
        // mapped to a placeholder token.
        assert_eq!(tokenize("h3h"), [50, 50]);
    }

    #[test]
    fn tokenize_truncates_to_the_context_length() {
        let long = "a".repeat(MAX_PHONEME_LENGTH + 50);
        assert_eq!(tokenize(&long).len(), MAX_PHONEME_LENGTH);
    }

    #[test]
    fn phonemize_produces_vocabulary_only_output() {
        let phonemes = phonemize("Hello there.", "en-us").expect("phonemise");
        assert!(!phonemes.is_empty());
        assert!(
            phonemes.chars().all(|c| vocab().contains_key(&c)),
            "{phonemes:?}"
        );
        assert!(!phonemes.contains("  "));
    }

    #[test]
    fn phonemize_marks_stress() {
        // Stress marks are what distinguish this from a plain phoneme dump;
        // losing them flattens the prosody.
        assert!(phonemize("Hello there.", "en-us").unwrap().contains('ˈ'));
    }
}
