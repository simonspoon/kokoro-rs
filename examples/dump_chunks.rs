//! Exposes the chunker to `scripts/diff_chunking.py`, which checks it against
//! the Python implementation this project was ported from.
//!
//! Reads a request as JSON on stdin and writes a JSON array on stdout:
//!
//!     {"mode": "chunk"|"stream"|"ends_sentence",
//!      "limit": 100, "chunk_chars": 300, "pieces": ["..."]}

use std::io::Read;

use kokoro_rs::text::{ChunkStream, chunk_text, ends_sentence};

fn main() {
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .expect("read stdin");
    let request: serde_json::Value = serde_json::from_str(&input).expect("parse request");

    let limit = request["limit"].as_u64().expect("limit") as usize;
    let chunk_chars = request["chunk_chars"].as_u64().expect("chunk_chars") as usize;
    let pieces: Vec<&str> = request["pieces"]
        .as_array()
        .expect("pieces")
        .iter()
        .map(|p| p.as_str().unwrap())
        .collect();

    let output: serde_json::Value = match request["mode"].as_str().expect("mode") {
        "chunk" => chunk_text(pieces[0], limit).into(),
        "stream" => {
            let mut stream = ChunkStream::new(limit, chunk_chars);
            let mut chunks: Vec<String> = Vec::new();
            for piece in pieces {
                chunks.extend(stream.push(piece));
            }
            chunks.extend(stream.finish());
            chunks.into()
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
