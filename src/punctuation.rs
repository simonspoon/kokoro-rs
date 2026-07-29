//! Preserving punctuation across phonemisation.
//!
//! espeak-ng drops punctuation silently, but Kokoro was trained on phoneme
//! strings that contain it — commas and full stops are what give the model its
//! pauses and final-syllable intonation. The Python implementation gets them
//! back through `phonemizer`'s `preserve_punctuation` option, which hides the
//! marks from the backend and splices them into the phonemes afterwards.
//!
//! This is a direct port of `phonemizer.punctuation.Punctuation`. It is
//! reproduced rather than simplified because the placement rules are load
//! bearing: `scripts/diff_phonemes.py` checks the whole pipeline against the
//! Python original, and small deviations here change the audio.

/// The punctuation marks phonemizer considers by default.
const MARKS: &[char] = &[
    ';', ':', ',', '.', '!', '?', '¡', '¿', '—', '…', '"', '«', '»', '“', '”', '(', ')', '{', '}',
    '[', ']',
];

/// Where a mark sat relative to the text around it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Position {
    /// At the start of the utterance.
    Begin,
    /// At the end of the utterance.
    End,
    /// Between two pieces of text.
    Inner,
    /// The utterance was nothing but punctuation.
    Alone,
}

#[derive(Debug, Clone)]
struct Mark {
    /// Index of the input line this mark came from.
    index: usize,
    mark: String,
    position: Position,
}

/// One run of punctuation, together with any surrounding whitespace.
///
/// Equivalent to phonemizer's `(\s*[marks]+\s*)+`: whitespace either side is
/// swallowed into the mark so it can be replayed verbatim on restore.
fn find_marks(line: &str) -> Vec<(usize, usize)> {
    let chars: Vec<(usize, char)> = line.char_indices().collect();
    let mut runs = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        // A run must contain at least one mark, so scan leading whitespace
        // only as a candidate prefix.
        let start = i;
        let mut j = i;
        while j < chars.len() && chars[j].1.is_whitespace() {
            j += 1;
        }
        if j >= chars.len() || !MARKS.contains(&chars[j].1) {
            i += 1;
            continue;
        }
        // Consume alternating runs of marks and whitespace, greedily.
        let mut end = j;
        loop {
            let mut k = end;
            while k < chars.len() && MARKS.contains(&chars[k].1) {
                k += 1;
            }
            if k == end {
                break;
            }
            end = k;
            while end < chars.len() && chars[end].1.is_whitespace() {
                end += 1;
            }
        }
        let byte_start = chars[start].0;
        let byte_end = if end < chars.len() {
            chars[end].0
        } else {
            line.len()
        };
        runs.push((byte_start, byte_end));
        i = end;
    }
    runs
}

/// The text with punctuation removed, plus what is needed to restore it.
struct Preserved {
    chunks: Vec<String>,
    marks: Vec<Mark>,
}

/// Remove punctuation from `lines`, allowing for later restoration.
///
/// `"hello, my world!"` becomes `["hello", "my world"]` with marks `[",", "!"]`.
fn preserve(lines: &[String]) -> Preserved {
    let mut chunks = Vec::new();
    let mut marks = Vec::new();
    for (num, line) in lines.iter().enumerate() {
        let (line_chunks, line_marks) = preserve_line(line, num);
        chunks.extend(line_chunks);
        marks.extend(line_marks);
    }
    Preserved {
        chunks: chunks.into_iter().filter(|c| !c.is_empty()).collect(),
        marks,
    }
}

fn preserve_line(line: &str, num: usize) -> (Vec<String>, Vec<Mark>) {
    let found = find_marks(line);
    if found.is_empty() {
        return (vec![line.to_string()], Vec::new());
    }
    // The line is made only of punctuation marks.
    if found.len() == 1 && &line[found[0].0..found[0].1] == line {
        return (
            Vec::new(),
            vec![Mark {
                index: num,
                mark: line.to_string(),
                position: Position::Alone,
            }],
        );
    }

    let last = found.len() - 1;
    let marks: Vec<Mark> = found
        .iter()
        .enumerate()
        .map(|(i, &(start, end))| {
            let mark = &line[start..end];
            let position = if i == 0 && line.starts_with(mark) {
                Position::Begin
            } else if i == last && line.ends_with(mark) {
                Position::End
            } else {
                Position::Inner
            };
            Mark {
                index: num,
                mark: mark.to_string(),
                position,
            }
        })
        .collect();

    // Split the line into sublines, each separated by a punctuation mark.
    // Each split works on what is left of the line, as in the original.
    let mut chunks = Vec::new();
    let mut rest = line.to_string();
    for mark in &marks {
        let parts: Vec<&str> = rest.split(mark.mark.as_str()).collect();
        let prefix = parts[0].to_string();
        let suffix = parts[1..].join(mark.mark.as_str());
        chunks.push(prefix);
        rest = suffix;
    }
    chunks.push(rest);
    (chunks, marks)
}

/// Splice the marks back into the phonemised chunks.
///
/// The reverse of [`preserve`]: `["hello", "my world"]` with `[",", "!"]`
/// becomes `["hello, my world!"]`.
fn restore(mut text: Vec<String>, mut marks: Vec<Mark>, word_sep: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut pos = 0usize;

    while !text.is_empty() || !marks.is_empty() {
        if marks.is_empty() {
            for line in text.drain(..) {
                // Ensure the final word still ends with a word separator.
                let line = if !word_sep.is_empty() && !line.ends_with(word_sep) {
                    line + word_sep
                } else {
                    line
                };
                out.push(line);
            }
            continue;
        }
        if text.is_empty() {
            // Nothing was phonemised: return the marks alone.
            out.push(
                marks
                    .iter()
                    .map(|m| m.mark.as_str())
                    .collect::<String>()
                    .replace(' ', word_sep),
            );
            marks.clear();
            continue;
        }

        if marks[0].index != pos {
            out.push(text.remove(0));
            pos += 1;
            continue;
        }

        let current = marks.remove(0);
        let mark = current.mark.replace(' ', word_sep);
        // Remove the trailing word separator before splicing.
        if !word_sep.is_empty() && text[0].ends_with(word_sep) {
            let keep = text[0].len() - word_sep.len();
            text[0].truncate(keep);
        }
        match current.position {
            Position::Begin => text[0] = mark + &text[0],
            Position::End => {
                let tail = if mark.ends_with(word_sep) {
                    ""
                } else {
                    word_sep
                };
                out.push(format!("{}{mark}{tail}", text.remove(0)));
                pos += 1;
            }
            Position::Alone => {
                let tail = if mark.ends_with(word_sep) {
                    ""
                } else {
                    word_sep
                };
                out.push(format!("{mark}{tail}"));
                pos += 1;
            }
            Position::Inner => {
                if text.len() == 1 {
                    // The final part of an inner mark was never phonemised.
                    text[0].push_str(&mark);
                } else {
                    let first = text.remove(0);
                    text[0] = format!("{first}{mark}{}", text[0]);
                }
            }
        }
    }
    out
}

/// Phonemise `text`, keeping its punctuation, using `backend` for the parts
/// between the marks.
///
/// Mirrors `phonemizer.phonemize(..., preserve_punctuation=True)` with the
/// default word separator and `strip=False`, which is how the Python
/// implementation calls it.
pub fn phonemize_preserving<F>(text: &str, mut backend: F) -> anyhow::Result<String>
where
    F: FnMut(&str) -> anyhow::Result<String>,
{
    const WORD_SEP: &str = " ";

    // phonemizer treats the input as a single line and drops it if blank.
    let lines: Vec<String> = if text.trim().is_empty() {
        Vec::new()
    } else {
        vec![text.to_string()]
    };
    if lines.is_empty() {
        return Ok(String::new());
    }

    let Preserved { chunks, marks } = preserve(&lines);

    let mut phonemised = Vec::with_capacity(chunks.len());
    for chunk in &chunks {
        let words = backend(chunk)?;
        // Every word is followed by a separator when `strip` is false; the
        // restore step relies on that trailing separator being present.
        phonemised.push(if words.is_empty() {
            words
        } else {
            words + WORD_SEP
        });
    }

    // Lines are joined with a newline, which is outside the vocabulary and so
    // disappears when the caller filters — matching `list2str`.
    Ok(restore(phonemised, marks, WORD_SEP).join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in backend: uppercases each word so the structure is visible.
    fn fake(chunk: &str) -> anyhow::Result<String> {
        Ok(chunk
            .split_whitespace()
            .map(|w| w.to_uppercase())
            .collect::<Vec<_>>()
            .join(" "))
    }

    fn run(text: &str) -> String {
        phonemize_preserving(text, fake).unwrap()
    }

    #[test]
    fn restores_a_trailing_mark() {
        assert_eq!(run("Hello from Rust."), "HELLO FROM RUST. ");
    }

    #[test]
    fn restores_marks_between_sentences() {
        // The inner ". " keeps its space; the final "." gains a word separator.
        assert_eq!(run("One. Two."), "ONE. TWO. ");
    }

    #[test]
    fn splits_and_rejoins_around_inner_marks() {
        assert_eq!(run("hello, my world!"), "HELLO, MY WORLD! ");
    }

    #[test]
    fn handles_a_leading_mark() {
        assert_eq!(run("\"quoted\" text"), "\"QUOTED\" TEXT ");
    }

    #[test]
    fn handles_text_that_is_only_punctuation() {
        // Nothing reaches the backend; the marks are returned on their own.
        assert_eq!(run("!?!"), "!?!");
    }

    #[test]
    fn blank_input_yields_nothing() {
        assert_eq!(run("   "), "");
    }

    #[test]
    fn marks_capture_surrounding_whitespace() {
        let (chunks, marks) = preserve_line("a, b", 0);
        assert_eq!(chunks, ["a", "b"]);
        assert_eq!(marks.len(), 1);
        assert_eq!(marks[0].mark, ", ");
        assert_eq!(marks[0].position, Position::Inner);
    }
}
