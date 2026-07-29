//! Splitting incoming text into synthesis-sized chunks.
//!
//! Kokoro's context is 510 phonemes, but we chunk far below that: shorter
//! chunks reach the speakers sooner, and the first chunk is kept shortest of
//! all so that audio starts while the rest of the text is still being
//! synthesised.

/// Titles and abbreviations whose trailing dot does not end a sentence.
/// Splitting on these would insert an audible pause mid-phrase ("Dr. | Smith").
const ABBREVIATIONS: &[&str] = &[
    "Mr", "Mrs", "Ms", "Mx", "Dr", "Prof", "Rev", "Hon", "Sr", "Jr", "St", "Mt", "Gen", "Col",
    "Sgt", "Capt", "Lt", "Cmdr", "Inc", "Ltd", "Co", "Corp", "Dept", "Est", "Fig", "No", "Vol",
    "Ch", "Sec", "Ref", "Approx", "vs", "etc", "al", "ca", "cf", "viz", "i.e", "e.g", "a.m", "p.m",
    "U.S", "U.K",
];

/// Terminal punctuation that can end a sentence.
const TERMINALS: [char; 4] = ['.', '!', '?', '…'];

/// Closing quotes and brackets allowed between the punctuation and the space.
const CLOSERS: [char; 6] = ['"', '\'', ')', ']', '\u{201d}', '\u{2019}'];

/// Punctuation that ends a clause — the preferred place to break a sentence
/// that is too long to synthesise in one piece.
const CLAUSE_ENDS: [char; 4] = [',', ';', ':', '—'];

pub const FIRST_CHUNK_CHARS: usize = 100;
pub const CHUNK_CHARS: usize = 300;

fn nchars(s: &str) -> usize {
    s.chars().count()
}

/// Whether the text ending at `end` is a sentence break rather than an
/// abbreviation or an initial.
///
/// `end` is the byte offset just past the terminal punctuation. The Python
/// original expressed this as negative lookbehinds on the split pattern;
/// Rust's `regex` has no lookbehind, so the vetoes are checked directly.
fn is_vetoed(text: &str, end: usize) -> bool {
    let head = &text[..end];
    if !head.ends_with('.') {
        // Only a dot is ambiguous — "!" and "?" always end a sentence.
        return false;
    }
    let before_dot = &head[..head.len() - 1];

    // A single initial, as in "J. Smith": one letter preceded by a
    // non-word character (or the start of the text).
    let mut tail = before_dot.chars().rev();
    if let Some(letter) = tail.next()
        && letter.is_ascii_alphabetic()
        && !tail.next().is_some_and(|c| c.is_alphanumeric() || c == '_')
    {
        return true;
    }

    // A known abbreviation. Matched case-insensitively and as a plain suffix,
    // mirroring the original's lookbehind group.
    let lowered = before_dot.to_lowercase();
    ABBREVIATIONS
        .iter()
        .any(|abbr| lowered.ends_with(&abbr.to_lowercase()))
}

/// Byte ranges of the separators at every sentence boundary in `text`.
///
/// A boundary is terminal punctuation, then an optional closing quote or
/// bracket, then whitespace. The returned range covers the closer and the
/// whitespace — the text either side of it is what a split yields. Decimals
/// like "3.5" never match, since the dot there is not followed by whitespace.
fn sentence_breaks(text: &str) -> Vec<(usize, usize)> {
    let mut breaks = Vec::new();
    for (i, c) in text.char_indices() {
        if !TERMINALS.contains(&c) {
            continue;
        }
        let start = i + c.len_utf8();
        if is_vetoed(text, start) {
            continue;
        }

        let mut cursor = start;
        let rest = &text[start..];
        let mut chars = rest.chars();
        // At most one closer, exactly as the original's `["\')\]”’]?`.
        if let Some(c) = chars.clone().next()
            && CLOSERS.contains(&c)
        {
            cursor += c.len_utf8();
            chars.next();
        }
        let whitespace: usize = chars
            .take_while(|c| c.is_whitespace())
            .map(char::len_utf8)
            .sum();
        if whitespace > 0 {
            breaks.push((start, cursor + whitespace));
        }
    }
    breaks
}

/// Split `text` at every sentence boundary, dropping blank pieces.
fn split_sentences(text: &str) -> Vec<&str> {
    let mut pieces = Vec::new();
    let mut prev = 0;
    for (start, end) in sentence_breaks(text) {
        // Overlapping matches cannot occur, but a boundary's whitespace can
        // swallow the start of the next candidate; skip anything behind us.
        if start < prev {
            continue;
        }
        pieces.push(&text[prev..start]);
        prev = end;
    }
    pieces.push(&text[prev..]);
    pieces
        .into_iter()
        .filter(|p| !p.trim().is_empty())
        .collect()
}

/// Split at the last sentence boundary, returning (head, separator, tail).
fn rsplit_sentence(text: &str) -> Option<(&str, &str, &str)> {
    let (start, end) = *sentence_breaks(text).last()?;
    Some((&text[..start], &text[start..end], &text[end..]))
}

/// Whether a chunk ends on sentence-final punctuation.
///
/// Synthesising each chunk separately clips the pause the model would have put
/// between sentences, so the caller reinstates one — but only here, never at a
/// boundary that merely split an over-long sentence.
pub fn ends_sentence(chunk: &str) -> bool {
    let trimmed = chunk.trim_end();
    let trimmed = match trimmed.chars().next_back() {
        Some(c) if CLOSERS.contains(&c) => &trimmed[..trimmed.len() - c.len_utf8()],
        _ => trimmed,
    };
    trimmed
        .chars()
        .next_back()
        .is_some_and(|c| TERMINALS.contains(&c))
}

/// Greedily pack `parts` into chunks of at most `limit` characters, rejoining
/// them with a single space.
fn pack<'a>(parts: impl Iterator<Item = &'a str>, limit: usize, out: &mut Vec<String>) {
    let mut buf = String::new();
    for part in parts {
        if !buf.is_empty() && nchars(&buf) + 1 + nchars(part) > limit {
            out.push(std::mem::take(&mut buf));
            buf.push_str(part);
        } else {
            if !buf.is_empty() {
                buf.push(' ');
            }
            buf.push_str(part);
        }
    }
    if !buf.is_empty() {
        out.push(buf);
    }
}

/// Break a single over-long sentence, preferring clause then word boundaries.
fn hard_split(text: &str, limit: usize, out: &mut Vec<String>) {
    if nchars(text) <= limit {
        out.push(text.to_string());
        return;
    }

    // Clause boundaries first: breaking at a comma is far less audible than
    // breaking between two arbitrary words.
    let clauses = split_after(text, &CLAUSE_ENDS);
    if clauses.len() >= 2 {
        pack(clauses.into_iter(), limit, out);
        return;
    }
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() >= 2 {
        pack(words.into_iter(), limit, out);
        return;
    }

    // A single unbroken token longer than the limit: cut it bluntly.
    let chars: Vec<char> = text.chars().collect();
    for piece in chars.chunks(limit) {
        out.push(piece.iter().collect());
    }
}

/// Split on the whitespace that follows any of `marks`, keeping the mark.
fn split_after<'a>(text: &'a str, marks: &[char]) -> Vec<&'a str> {
    let mut parts = Vec::new();
    let mut prev = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if !marks.contains(&c) {
            continue;
        }
        let start = i + c.len_utf8();
        let ws: usize = text[start..]
            .chars()
            .take_while(|c| c.is_whitespace())
            .map(char::len_utf8)
            .sum();
        if ws > 0 {
            parts.push(&text[prev..start]);
            prev = start + ws;
            while chars.peek().is_some_and(|(j, _)| *j < prev) {
                chars.next();
            }
        }
    }
    parts.push(&text[prev..]);
    parts.into_iter().filter(|p| !p.is_empty()).collect()
}

/// Split `text` into chunks of at most roughly `limit` characters.
pub fn chunk_text(text: &str, limit: usize) -> Vec<String> {
    let text = text.trim();
    let mut out = Vec::new();
    if text.is_empty() {
        return out;
    }

    // Newlines end a sentence whether or not they are punctuated, so blocks
    // are split first and sentence-split independently.
    let mut sentences: Vec<&str> = Vec::new();
    for block in text.split('\n') {
        let block = block.trim();
        if !block.is_empty() {
            sentences.extend(split_sentences(block));
        }
    }

    let mut buf = String::new();
    for sentence in sentences {
        let sentence = sentence.trim();
        if nchars(sentence) > limit {
            if !buf.is_empty() {
                out.push(std::mem::take(&mut buf));
            }
            hard_split(sentence, limit, &mut out);
            continue;
        }
        if !buf.is_empty() && nchars(&buf) + 1 + nchars(sentence) > limit {
            out.push(std::mem::replace(&mut buf, sentence.to_string()));
        } else {
            if !buf.is_empty() {
                buf.push(' ');
            }
            buf.push_str(sentence);
        }
    }
    if !buf.is_empty() {
        out.push(buf);
    }
    out
}

/// Chunks an incremental stream of text pieces (e.g. lines arriving on stdin).
///
/// Text is emitted as soon as a complete chunk is available rather than waiting
/// for the whole input, so `tail -f log | kokoro-rs` speaks as the log grows.
pub struct ChunkStream {
    limit: usize,
    chunk_chars: usize,
    pending: String,
}

impl ChunkStream {
    pub fn new(first_chunk_chars: usize, chunk_chars: usize) -> Self {
        Self {
            limit: first_chunk_chars,
            chunk_chars,
            pending: String::new(),
        }
    }

    /// Feed one piece of input, returning whatever chunks are now complete.
    pub fn push(&mut self, piece: &str) -> Vec<String> {
        self.pending.push_str(piece);

        // Only emit up to the last sentence boundary; the tail may still grow.
        let has_break = !sentence_breaks(&self.pending).is_empty();
        if !has_break && nchars(&self.pending) < self.limit {
            return Vec::new();
        }

        let emit = match rsplit_sentence(&self.pending) {
            Some((head, _, tail)) => {
                let (head, tail) = (head.to_string(), tail.to_string());
                self.pending = tail;
                head
            }
            None if nchars(&self.pending) < self.limit => return Vec::new(),
            None => std::mem::take(&mut self.pending),
        };
        self.emit(&emit)
    }

    /// Flush whatever is left once the input ends.
    pub fn finish(&mut self) -> Vec<String> {
        let rest = std::mem::take(&mut self.pending);
        self.emit(&rest)
    }

    fn emit(&mut self, text: &str) -> Vec<String> {
        let chunks = chunk_text(text, self.limit);
        // Only the very first chunk is kept short; once audio is playing,
        // larger chunks give the model more context and better prosody.
        if !chunks.is_empty() {
            self.limit = self.chunk_chars;
        }
        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunks(text: &str, limit: usize) -> Vec<String> {
        chunk_text(text, limit)
    }

    #[test]
    fn splits_on_sentence_ends() {
        // Sentences are repacked greedily, so two fit inside a limit of ten.
        assert_eq!(chunks("One. Two. Three.", 10), ["One. Two.", "Three."]);
        // "Three." exceeds a limit of five and is cut, having no inner break.
        assert_eq!(
            chunks("One. Two. Three.", 5),
            ["One.", "Two.", "Three", "."]
        );
    }

    #[test]
    fn packs_sentences_up_to_the_limit() {
        assert_eq!(chunks("One. Two. Three.", 300), ["One. Two. Three."]);
    }

    #[test]
    fn keeps_abbreviations_and_initials_intact() {
        for text in ["Dr. Smith arrived.", "J. Smith arrived.", "See Fig. 4 now."] {
            assert_eq!(chunks(text, 300).len(), 1, "split {text:?}");
        }
    }

    #[test]
    fn keeps_decimals_intact() {
        assert_eq!(chunks("It cost 3.5 pounds.", 300), ["It cost 3.5 pounds."]);
    }

    #[test]
    fn splits_after_a_closing_quote() {
        // The closing quote is part of the separator and is dropped with it,
        // matching the Python original.
        assert_eq!(
            chunks("\"Stop!\" she said. Then left.", 12),
            ["\"Stop!", "she said.", "Then left."]
        );
    }

    #[test]
    fn newlines_separate_sentences() {
        assert_eq!(chunks("one\ntwo", 300), ["one two"]);
        assert_eq!(chunks("one\ntwo", 4), ["one", "two"]);
    }

    #[test]
    fn long_sentence_breaks_at_clauses() {
        let text = "alpha beta gamma, delta epsilon zeta, eta theta iota";
        assert_eq!(
            chunks(text, 25),
            ["alpha beta gamma,", "delta epsilon zeta,", "eta theta iota"]
        );
    }

    #[test]
    fn long_sentence_falls_back_to_words() {
        assert_eq!(
            chunks("alpha beta gamma delta", 12),
            ["alpha beta", "gamma delta"]
        );
    }

    #[test]
    fn unbreakable_token_is_cut_bluntly() {
        assert_eq!(chunks(&"x".repeat(7), 3), ["xxx", "xxx", "x"]);
    }

    #[test]
    fn counts_characters_not_bytes() {
        // Five characters, ten bytes — must not be split at a five-char limit.
        assert_eq!(chunks("ααααα", 5), ["ααααα"]);
    }

    #[test]
    fn ends_sentence_detects_terminal_punctuation() {
        assert!(ends_sentence("Done."));
        assert!(ends_sentence("Really? "));
        assert!(ends_sentence("\"Stop!\""));
        assert!(!ends_sentence("a clause,"));
        assert!(!ends_sentence("mid sentence"));
    }

    #[test]
    fn stream_emits_the_first_chunk_early() {
        let mut stream = ChunkStream::new(10, 300);
        // No boundary and under the limit: nothing to say yet.
        assert!(stream.push("hello ").is_empty());
        // A sentence boundary releases the completed sentence, which at this
        // limit is itself split; the unterminated tail is held back.
        assert_eq!(stream.push("there. more"), ["hello", "there."]);
        assert_eq!(stream.finish(), ["more"]);
    }

    #[test]
    fn stream_widens_after_the_first_chunk() {
        let mut stream = ChunkStream::new(10, 300);
        assert_eq!(
            stream.push("one. two. three. four."),
            ["one. two.", "three."]
        );
        // Later chunks use the wider limit, so the tail stays in one piece.
        assert_eq!(stream.finish(), ["four."]);
    }

    #[test]
    fn stream_handles_input_with_no_punctuation() {
        let mut stream = ChunkStream::new(10, 20);
        let mut all: Vec<String> = Vec::new();
        for word in ["alpha ", "beta ", "gamma ", "delta "] {
            all.extend(stream.push(word));
        }
        all.extend(stream.finish());
        assert_eq!(all.concat().replace(' ', ""), "alphabetagammadelta");
    }

    #[test]
    fn blank_input_produces_nothing() {
        assert!(chunks("   \n  \n ", 300).is_empty());
        let mut stream = ChunkStream::new(10, 300);
        assert!(stream.push("  ").is_empty());
        assert!(stream.finish().is_empty());
    }
}
