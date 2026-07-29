//! Prints the phonemes and token IDs for each argument, for comparison with
//! the Python implementation via `scripts/diff_phonemes.py`.

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let lang = std::env::var("KOKORO_LANG").unwrap_or_else(|_| "en-us".into());
    for text in args {
        let phonemes = kokoro_rs::phonemes::phonemize(&text, &lang).expect("phonemise");
        let tokens = kokoro_rs::phonemes::tokenize(&phonemes);
        println!("{phonemes}\t{tokens:?}");
    }
}
