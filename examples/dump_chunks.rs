//! Exposes the chunker to `scripts/diff_chunking.py`, which checks it against
//! the Python implementation this project was ported from.
//!
//! Reads a request as JSON on stdin and writes a JSON array of chunk texts on
//! stdout. `limit` and `chunk_chars` are the character caps, as in the Python
//! version; `chunk_phonemes` is the phoneme budget the Rust chunker really
//! sizes by, and defaults to 500.
//!
//!     {"mode": "chunk"|"stream"|"ends_sentence",
//!      "limit": 100, "chunk_chars": 300, "chunk_phonemes": 500,
//!      "pieces": ["..."]}

use std::io::Read;

use kokoro_rs::phonemes;
use kokoro_rs::text::{Budget, CHUNK_PHONEMES, Chunk, ChunkStream, chunk_text, ends_sentence};

fn main() {
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .expect("read stdin");
    let request: serde_json::Value = serde_json::from_str(&input).expect("parse request");

    let limit = request["limit"].as_u64().expect("limit") as usize;
    let chunk_chars = request["chunk_chars"].as_u64().expect("chunk_chars") as usize;
    let chunk_phonemes = request["chunk_phonemes"]
        .as_u64()
        .map_or(CHUNK_PHONEMES, |n| n as usize);
    let pieces: Vec<&str> = request["pieces"]
        .as_array()
        .expect("pieces")
        .iter()
        .map(|p| p.as_str().unwrap())
        .collect();

    let mut phonemize = |text: &str| phonemes::phonemize(text, "en-us");
    let first = Budget {
        phonemes: chunk_phonemes.min(limit),
        chars: Some(limit),
    };
    let rest = Budget {
        phonemes: chunk_phonemes,
        chars: Some(chunk_chars),
    };

    let output: serde_json::Value = match request["mode"].as_str().expect("mode") {
        "chunk" => texts(chunk_text(pieces[0], first, rest, &mut phonemize).expect("chunk")).into(),
        "stream" => {
            let mut stream = ChunkStream::new(first, rest, phonemize);
            let mut chunks: Vec<Chunk> = Vec::new();
            for piece in pieces {
                chunks.extend(stream.push(piece).expect("push"));
            }
            chunks.extend(stream.finish().expect("finish"));
            texts(chunks).into()
        }
        "ends_sentence" => pieces
            .iter()
            .map(|p| ends_sentence(p))
            .collect::<Vec<_>>()
            .into(),
        mode => panic!("unknown mode {mode}"),
    };
    println!("{output}");
}

fn texts(chunks: Vec<Chunk>) -> Vec<String> {
    chunks.into_iter().map(|c| c.text).collect()
}
