//! Splitting incoming text into synthesis-sized chunks.
//!
//! The real constraint is Kokoro's context of 510 phonemes, so that is what
//! chunks are measured in: characters are only a proxy for it, and a poor one,
//! since "1234567" is seven characters and forty phonemes. Measuring the
//! phonemes directly lets whole sentences stay together far more often, which
//! the model turns into better prosody. The first chunk is kept deliberately
//! short so that audio starts while the rest of the text is still being
//! synthesised.
//!
//! Each piece of text is phonemised exactly once: the phonemes a chunk was
//! measured with are carried out with it, so the synthesiser can use them
//! directly. Joining them at punctuation is exact —
//! [`crate::punctuation::phonemize_preserving`] already splits there before
//! calling espeak, so phonemising two sentences (or two clauses) separately and
//! joining the results with a space gives the same string as phonemising the
//! joined text. Joining across a word boundary is not, which is why the pieces
//! [`hard_split`] cuts between words are never packed back together.

use anyhow::Result;

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

/// Phonemes per chunk. Kept under the model's 510 because `synth::split_phonemes`
/// packs up to but not including that figure, and a chunk it has to split again
/// is one this budget sized for nothing.
pub const CHUNK_PHONEMES: usize = 500;
pub const FIRST_CHUNK_PHONEMES: usize = 100;

fn nchars(s: &str) -> usize {
    s.chars().count()
}

/// One synthesis pass: the text as it will be spoken, and the phonemes it was
/// measured with — carried through so the synthesiser does not have to
/// phonemise it a second time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub text: String,
    pub phonemes: String,
}

/// How much a chunk may hold.
///
/// `phonemes` is the real limit; `chars` is an optional extra cap the user can
/// ask for with `--chunk-chars`, applied to the text before it is phonemised.
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    pub phonemes: usize,
    pub chars: Option<usize>,
}

impl Budget {
    /// Whether `piece` still fits once appended to `buf` with a joining space.
    fn fits(&self, buf: &Chunk, piece: &Chunk) -> bool {
        nchars(&buf.phonemes) + 1 + nchars(&piece.phonemes) <= self.phonemes
            && self
                .chars
                .is_none_or(|cap| nchars(&buf.text) + 1 + nchars(&piece.text) <= cap)
    }
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

/// Builds chunks out of already-phonemised pieces, in order.
///
/// Text and phonemes alike are rejoined with a single space. A piece too large
/// to fit on its own simply lands in a chunk by itself; only [`chunk_text`]
/// knows how to break one down further.
struct Packer {
    /// The budget in force. Closing a chunk widens it to `rest`: only the
    /// first chunk is held to the narrow budget that gets audio started.
    budget: Budget,
    rest: Budget,
    buf: Option<Chunk>,
    /// Whether the chunk being built may still take another piece.
    open: bool,
    out: Vec<Chunk>,
}

impl Packer {
    fn new(first: Budget, rest: Budget) -> Self {
        Self {
            budget: first,
            rest,
            buf: None,
            open: true,
            out: Vec::new(),
        }
    }

    /// Add `piece` to the chunk being built, closing that chunk off first if
    /// the piece no longer fits.
    ///
    /// `joinable` is false for a piece that must not share a chunk with its
    /// neighbours: [`hard_split`] can cut between two words, and words
    /// phonemised apart are not the phonemes of the two together — espeak
    /// reads each in the context of its neighbours, so "read" alone comes out
    /// as the present tense whatever the sentence around it said.
    fn push(&mut self, piece: Chunk, joinable: bool) {
        if let Some(held) = &self.buf
            && (!joinable || !self.open || !self.budget.fits(held, &piece))
        {
            self.close();
        }
        match &mut self.buf {
            Some(held) => {
                held.text.push(' ');
                held.text.push_str(&piece.text);
                held.phonemes.push(' ');
                held.phonemes.push_str(&piece.phonemes);
            }
            None => self.buf = Some(piece),
        }
        self.open = joinable;
    }

    /// Finish the chunk being built, if any, and widen the budget.
    fn close(&mut self) {
        if let Some(held) = self.buf.take() {
            self.out.push(held);
            self.budget = self.rest;
        }
        self.open = true;
    }

    fn finish(mut self) -> Vec<Chunk> {
        self.close();
        self.out
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

/// Split `text` into chunks, phonemising each piece once.
///
/// The first chunk is sized by `first` and everything after it by `rest`, so a
/// long run of text handed over in one go still starts with a short chunk and
/// then packs the remainder properly.
///
/// `phonemize` is called on every piece of text that survives the splitting, in
/// order, and never twice on the same piece — except where a sentence turns out
/// to be too long and has to be re-split at its clauses.
pub fn chunk_text<F>(text: &str, first: Budget, rest: Budget, phonemize: &mut F) -> Result<Vec<Chunk>>
where
    F: FnMut(&str) -> Result<String>,
{
    let text = text.trim();
    if text.is_empty() {
        return Ok(Vec::new());
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

    // Sentences are measured and packed in order, so that the budget in force
    // is the one that applies to the chunk each piece is landing in.
    let mut packer = Packer::new(first, rest);
    for sentence in sentences {
        let sentence = sentence.trim();

        // Over the narrow first-chunk cap usually just means the sentence
        // belongs in the next chunk, which is wider; close this one before
        // cutting the sentence up to meet a cap it need never have met. The
        // first chunk still honours the narrow cap, since a sentence that is
        // going into it arrives with nothing held back.
        if packer.buf.is_some() && packer.budget.chars.is_some_and(|cap| nchars(sentence) > cap) {
            packer.close();
        }

        // A character cap is a limit on the text itself, so it is applied
        // before phonemising; the phoneme budget can only be checked after.
        let mut parts = Vec::new();
        match packer.budget.chars {
            Some(cap) => hard_split(sentence, cap, &mut parts),
            None => parts.push(sentence.to_string()),
        }
        // A sentence the cap had to cut apart was cut between words, so its
        // pieces have to stay in the chunks they were measured for.
        let joinable = parts.len() == 1;

        for text in parts {
            let phonemes = phonemize(&text)?;
            if nchars(&phonemes) > packer.budget.phonemes && packer.buf.is_some() {
                // Over the narrow first-chunk budget usually just means the
                // piece belongs in the next chunk, which is wider; close this
                // one and measure again before going to the trouble — and the
                // extra phonemising — of re-splitting the sentence.
                packer.close();
            }
            if nchars(&phonemes) <= packer.budget.phonemes {
                packer.push(Chunk { text, phonemes }, joinable);
                continue;
            }

            // Anything that fills the budget on its own is re-split at its
            // clauses, the least audible place to break a sentence.
            let clauses = split_after(&text, &CLAUSE_ENDS);
            if clauses.len() < 2 {
                // A single clause that overruns the context is left whole for
                // `synth::split_phonemes` to break at a word boundary in the
                // phoneme string. Splitting the text into words and phonemising
                // them one by one would do the same job worse: espeak reads each
                // word in the context of its neighbours, and taken alone they
                // come out pronounced differently.
                packer.push(Chunk { text, phonemes }, joinable);
                continue;
            }
            for (i, clause) in clauses.into_iter().enumerate() {
                let piece = Chunk {
                    text: clause.to_string(),
                    phonemes: phonemize(clause)?,
                };
                // Clauses may rejoin each other — espeak splits at punctuation
                // anyway, so that is exact — but the first one inherits
                // whether the sentence itself could join what came before.
                packer.push(piece, i > 0 || joinable);
            }
        }
    }
    Ok(packer.finish())
}

/// Chunks an incremental stream of text pieces (e.g. lines arriving on stdin).
///
/// Text is emitted as soon as a complete chunk is available rather than waiting
/// for the whole input, so `tail -f log | kokoro-rs` speaks as the log grows.
pub struct ChunkStream<F> {
    /// The budget in force, narrow for the first chunk and `rest` after it.
    budget: Budget,
    rest: Budget,
    phonemize: F,
    pending: String,
}

impl<F: FnMut(&str) -> Result<String>> ChunkStream<F> {
    pub fn new(first: Budget, rest: Budget, phonemize: F) -> Self {
        Self {
            budget: first,
            rest,
            phonemize,
            pending: String::new(),
        }
    }

    /// Feed one piece of input, returning whatever chunks are now complete.
    pub fn push(&mut self, piece: &str) -> Result<Vec<Chunk>> {
        self.pending.push_str(piece);

        // Only emit up to the last sentence boundary; the tail may still grow.
        let has_break = !sentence_breaks(&self.pending).is_empty();
        if !has_break && nchars(&self.pending) < self.flush_at() {
            return Ok(Vec::new());
        }

        let emit = match rsplit_sentence(&self.pending) {
            Some((head, _, tail)) => {
                let (head, tail) = (head.to_string(), tail.to_string());
                self.pending = tail;
                head
            }
            None if nchars(&self.pending) < self.flush_at() => return Ok(Vec::new()),
            None => std::mem::take(&mut self.pending),
        };
        self.emit(&emit)
    }

    /// Flush whatever is left once the input ends.
    pub fn finish(&mut self) -> Result<Vec<Chunk>> {
        let rest = std::mem::take(&mut self.pending);
        self.emit(&rest)
    }

    /// How much unpunctuated text to hold before giving up on finding a
    /// sentence boundary. Only a trigger for emitting — the sizing of what is
    /// emitted is the phoneme budget, applied in [`chunk_text`] below — so
    /// counting characters here is good enough, and avoids phonemising the
    /// pending text on every keystroke to find out.
    fn flush_at(&self) -> usize {
        self.budget.chars.unwrap_or(self.budget.phonemes)
    }

    fn emit(&mut self, text: &str) -> Result<Vec<Chunk>> {
        let chunks = chunk_text(text, self.budget, self.rest, &mut self.phonemize)?;
        // Only the very first chunk is kept short; once audio is playing,
        // larger chunks give the model more context and better prosody.
        // `chunk_text` widens within a single call; this carries that across
        // the calls that a streamed input arrives in.
        if !chunks.is_empty() {
            self.budget = self.rest;
        }
        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phonemes::MAX_PHONEME_LENGTH;

    /// A stand-in phonemiser that leaves the text alone, so that phoneme counts
    /// are character counts and the expected chunks stay readable.
    fn identity() -> impl FnMut(&str) -> Result<String> {
        |text: &str| Ok(text.to_string())
    }

    fn budget(phonemes: usize) -> Budget {
        Budget {
            phonemes,
            chars: None,
        }
    }

    fn texts(chunks: Vec<Chunk>) -> Vec<String> {
        chunks.into_iter().map(|c| c.text).collect()
    }

    fn chunks(text: &str, limit: usize) -> Vec<String> {
        texts(chunk_text(text, budget(limit), budget(limit), &mut identity()).unwrap())
    }

    /// Chunking with `--chunk-chars` in force, capping text and phonemes alike.
    fn capped(text: &str, limit: usize) -> Vec<String> {
        let budget = Budget {
            phonemes: limit,
            chars: Some(limit),
        };
        texts(chunk_text(text, budget, budget, &mut identity()).unwrap())
    }

    #[test]
    fn splits_on_sentence_ends() {
        // Sentences are repacked greedily, so two fit inside a limit of ten.
        assert_eq!(chunks("One. Two. Three.", 10), ["One. Two.", "Three."]);
        assert_eq!(chunks("One. Two. Three.", 5), ["One.", "Two.", "Three."]);
        // Under a character cap of five, "Three." is over it and is cut.
        assert_eq!(
            capped("One. Two. Three.", 5),
            ["One.", "Two.", "Three", "."]
        );
    }

    #[test]
    fn only_the_first_chunk_is_held_to_the_narrow_budget() {
        // One call, several sentences: the first chunk gets audio started, and
        // the rest is packed to the wide budget rather than the narrow one.
        let text = "One. Two. Three. Four. Five. Six.";
        let chunks = texts(chunk_text(text, budget(5), budget(300), &mut identity()).unwrap());
        assert_eq!(chunks, ["One.", "Two. Three. Four. Five. Six."]);
    }

    #[test]
    fn packs_sentences_up_to_the_limit() {
        assert_eq!(chunks("One. Two. Three.", 300), ["One. Two. Three."]);
    }

    #[test]
    fn whole_sentences_are_never_split_when_they_fit() {
        let text = "The first sentence is fairly long. The second one is too. \
                    A third follows it, with a clause in the middle, and an end.";
        // Every chunk is made of whole sentences: none ends mid-sentence.
        for chunk in chunks(text, 60) {
            assert!(ends_sentence(&chunk), "split {chunk:?}");
        }
    }

    #[test]
    fn no_chunk_exceeds_the_phoneme_budget() {
        let text = "One sentence here. Another, with a clause, follows it. \
                    And a third, rather longer than the others, closes the lot.";
        for limit in [20, 40, 300] {
            for chunk in chunk_text(text, budget(limit), budget(limit), &mut identity()).unwrap() {
                // The one documented exception is a single clause with no
                // inner break, left whole for synth::split_phonemes.
                let indivisible = split_after(&chunk.text, &CLAUSE_ENDS).len() == 1;
                assert!(
                    nchars(&chunk.phonemes) <= limit || indivisible,
                    "{limit}: {:?}",
                    chunk.phonemes
                );
            }
        }
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
    fn over_long_clause_is_left_for_the_phoneme_splitter() {
        // No clause boundary to break at, and no character cap asking for a
        // word split, so the sentence is emitted whole; synth::split_phonemes
        // breaks the phoneme string at a word boundary instead.
        assert_eq!(
            chunks("alpha beta gamma delta", 12),
            ["alpha beta gamma delta"]
        );
    }

    #[test]
    fn long_sentence_falls_back_to_words_under_a_char_cap() {
        assert_eq!(
            capped("alpha beta gamma delta", 12),
            ["alpha beta", "gamma delta"]
        );
    }

    #[test]
    fn unbreakable_token_is_cut_bluntly() {
        assert_eq!(capped(&"x".repeat(7), 3), ["xxx", "xxx", "x"]);
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
        let mut stream = ChunkStream::new(budget(10), budget(300), identity());
        // No boundary and under the limit: nothing to say yet.
        assert!(stream.push("hello ").unwrap().is_empty());
        // A sentence boundary releases the completed sentence — over the limit
        // but with nowhere to break it — and holds back the unterminated tail.
        assert_eq!(texts(stream.push("there. more").unwrap()), ["hello there."]);
        assert_eq!(texts(stream.finish().unwrap()), ["more"]);
    }

    #[test]
    fn stream_widens_after_the_first_chunk() {
        let mut stream = ChunkStream::new(budget(10), budget(300), identity());
        assert_eq!(
            texts(stream.push("one. two. three. four.").unwrap()),
            ["one. two.", "three."]
        );
        // Later chunks use the wider budget, so the tail stays in one piece.
        assert_eq!(texts(stream.finish().unwrap()), ["four."]);
    }

    #[test]
    fn stream_handles_input_with_no_punctuation() {
        let mut stream = ChunkStream::new(budget(10), budget(20), identity());
        let mut all: Vec<String> = Vec::new();
        for word in ["alpha ", "beta ", "gamma ", "delta "] {
            all.extend(texts(stream.push(word).unwrap()));
        }
        all.extend(texts(stream.finish().unwrap()));
        assert_eq!(all.concat().replace(' ', ""), "alphabetagammadelta");
    }

    #[test]
    fn blank_input_produces_nothing() {
        assert!(chunks("   \n  \n ", 300).is_empty());
        let mut stream = ChunkStream::new(budget(10), budget(300), identity());
        assert!(stream.push("  ").unwrap().is_empty());
        assert!(stream.finish().unwrap().is_empty());
    }

    /// The carried phonemes must be what each chunk's text phonemises to: that
    /// is what the synthesiser speaks, and a mismatch is a mispronounced word.
    fn assert_phonemes_match(chunks: &[Chunk]) {
        let phonemize = |text: &str| crate::phonemes::phonemize(text, "en-us");
        for chunk in chunks {
            assert_eq!(
                chunk.phonemes,
                phonemize(&chunk.text).unwrap(),
                "{:?}",
                chunk.text
            );
        }
    }

    #[test]
    fn word_split_pieces_are_not_rejoined() {
        // Under a character cap the first chunk's narrow cap cuts this sentence
        // between words; the wider cap that comes into force underneath must
        // not put those pieces back together. espeak reads a word in the
        // context of its neighbours, so "read" measured alone is "reed" and
        // rejoining it would have the model say the wrong word.
        let mut phonemize = |text: &str| crate::phonemes::phonemize(text, "en-us");
        let text = "I have read a book and I have read a paper and then I went \
                    away and later I came back and I have read the apple once \
                    more today and I have read the news and I have read the \
                    letter and then I have read the apple again.";
        let first = Budget {
            phonemes: FIRST_CHUNK_PHONEMES,
            chars: Some(FIRST_CHUNK_CHARS),
        };
        let rest = Budget {
            phonemes: CHUNK_PHONEMES,
            chars: Some(300),
        };
        let chunks = chunk_text(text, first, rest, &mut phonemize).unwrap();
        assert!(chunks.len() > 1);
        assert_phonemes_match(&chunks);
    }

    #[test]
    fn clause_splits_and_the_widening_budget_keep_the_phonemes_honest() {
        let mut phonemize = |text: &str| crate::phonemes::phonemize(text, "en-us");
        let text = "Hi. A second sentence, with several clauses, that is far \
                    longer than a hundred phonemes, and keeps going, and going, \
                    and going for a while yet. A third one follows it. And a \
                    fourth, with a clause of its own, closes the lot.";
        let chunks = chunk_text(
            text,
            budget(FIRST_CHUNK_PHONEMES),
            budget(CHUNK_PHONEMES),
            &mut phonemize,
        )
        .unwrap();
        assert!(chunks.len() > 1);
        assert_phonemes_match(&chunks);
    }

    #[test]
    fn real_phonemes_fit_the_model_context() {
        let mut phonemize = |text: &str| crate::phonemes::phonemize(text, "en-us");
        let text = "The port keeps the same chunk boundaries as the original. \
                    Sentences are packed greedily, so a short one rides along \
                    with its neighbour, and a long one, full of clauses, is \
                    broken at the commas rather than between two arbitrary \
                    words. Numbers such as 1234567 expand several-fold into \
                    phonemes, which is exactly why the budget is counted there \
                    and not in characters. That is the whole idea.";
        let chunks = chunk_text(
            text,
            budget(FIRST_CHUNK_PHONEMES),
            budget(CHUNK_PHONEMES),
            &mut phonemize,
        )
        .unwrap();

        for chunk in &chunks {
            assert!(
                nchars(&chunk.phonemes) <= MAX_PHONEME_LENGTH,
                "{} phonemes",
                nchars(&chunk.phonemes)
            );
            // The carried phonemes must be what the chunk's text phonemises to,
            // since that is what the synthesiser will speak.
            assert_eq!(chunk.phonemes, phonemize(&chunk.text).unwrap());
        }

        let spoken: Vec<&str> = chunks
            .iter()
            .flat_map(|c| c.text.split_whitespace())
            .collect();
        assert_eq!(spoken, text.split_whitespace().collect::<Vec<_>>());
    }
}
