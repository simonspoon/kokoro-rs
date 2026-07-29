//! Trimming leading and trailing silence from a synthesised chunk.
//!
//! Kokoro pads every utterance with silence — roughly two seconds before the
//! first chunk and a fraction of a second after each one. Left in, those pauses
//! accumulate at every chunk boundary and the speech sounds halting, so the
//! reference implementation trims them and this port does the same.
//!
//! This is a port of `librosa.effects.trim` as vendored in `kokoro_onnx/trim.py`,
//! restricted to the mono, default-parameter case that Kokoro actually uses.

/// Samples per analysis frame.
const FRAME_LENGTH: usize = 2048;
/// Samples between analysis frames.
const HOP_LENGTH: usize = 512;
/// How far below the peak still counts as silence, in decibels.
const TOP_DB: f32 = 60.0;
/// librosa's floor for `amplitude_to_db`, squared for the power domain.
const AMIN: f32 = 1e-5 * 1e-5;

/// Return the range of `audio` that is not leading or trailing silence.
///
/// A uniform signal is left untouched, matching librosa: with `ref=max` every
/// frame sits at the peak, so none is more than `TOP_DB` below it. That
/// includes a wholly silent signal, where the `AMIN` floor applies equally to
/// the frames and the reference and the difference is zero.
pub fn trim_range(audio: &[f32]) -> (usize, usize) {
    let power = frame_power(audio);
    if power.is_empty() {
        return (0, audio.len());
    }

    // ref=np.max over the RMS amplitudes, squared back into the power domain.
    let peak = power.iter().copied().fold(0.0f32, f32::max);
    let ref_db = 10.0 * AMIN.max(peak).log10();

    let first = power
        .iter()
        .position(|&p| 10.0 * AMIN.max(p).log10() - ref_db > -TOP_DB);
    let Some(first) = first else { return (0, 0) };
    let last = power
        .iter()
        .rposition(|&p| 10.0 * AMIN.max(p).log10() - ref_db > -TOP_DB)
        .unwrap();

    let start = first * HOP_LENGTH;
    let end = audio.len().min((last + 1) * HOP_LENGTH);
    (start, end)
}

/// Trim `audio` in place, returning the surviving samples.
pub fn trim(audio: Vec<f32>) -> Vec<f32> {
    let (start, end) = trim_range(&audio);
    if start == 0 && end == audio.len() {
        return audio;
    }
    audio[start..end].to_vec()
}

/// Mean square per frame, over the centre-padded signal.
///
/// librosa's `rms` takes the square root and `amplitude_to_db` squares it
/// again; the two cancel, so the power is used directly.
fn frame_power(audio: &[f32]) -> Vec<f32> {
    // center=True pads by half a frame either side with zeros.
    let pad = FRAME_LENGTH / 2;
    let padded_len = audio.len() + 2 * pad;
    if padded_len < FRAME_LENGTH {
        return Vec::new();
    }
    let frames = 1 + (padded_len - FRAME_LENGTH) / HOP_LENGTH;

    // Indexing the padded signal lazily rather than materialising it.
    let at = |i: usize| -> f32 {
        if i < pad || i >= pad + audio.len() {
            0.0
        } else {
            audio[i - pad]
        }
    };

    (0..frames)
        .map(|f| {
            let start = f * HOP_LENGTH;
            let sum: f32 = (start..start + FRAME_LENGTH).map(|i| at(i) * at(i)).sum();
            sum / FRAME_LENGTH as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_all_silent_signal_is_left_alone() {
        // Nothing is below the peak when everything is the peak, so librosa
        // trims nothing here — verified against the Python implementation.
        assert_eq!(trim_range(&vec![0.0; 12_000]), (0, 12_000));
    }

    #[test]
    fn bounds_match_the_python_implementation() {
        // librosa.effects.trim on this signal returns (3072, 9216).
        let mut audio = vec![0.0f32; 12_000];
        for (i, s) in audio.iter_mut().enumerate().take(8_000).skip(4_000) {
            *s = ((i - 4_000) as f32 * 0.1).sin();
        }
        assert_eq!(trim_range(&audio), (3_072, 9_216));
    }

    #[test]
    fn a_signal_with_no_silence_is_left_alone() {
        let audio: Vec<f32> = (0..12_000).map(|i| (i as f32 * 0.1).sin()).collect();
        let (start, end) = trim_range(&audio);
        assert_eq!(start, 0);
        assert_eq!(end, audio.len());
    }

    #[test]
    fn a_signal_shorter_than_a_frame_is_left_alone() {
        let audio = vec![0.5f32; 100];
        assert_eq!(trim_range(&audio), (0, 100));
    }
}
