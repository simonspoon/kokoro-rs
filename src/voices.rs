//! The voice pack: one style matrix per voice.
//!
//! `voices-v1.0.bin` is a numpy `.npz` archive of 54 arrays, each `(510, 1,
//! 256)` float32. The first axis is indexed by token count, not time: a voice's
//! style vector depends on how long the utterance is, which is how the model
//! paces a short phrase differently from a long one.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use ndarray::Array3;
use ndarray_npy::NpzReader;

/// Width of a style vector — the model's `style` input is `(1, 256)`.
pub const STYLE_DIM: usize = 256;

pub struct Voices {
    styles: BTreeMap<String, Array3<f32>>,
}

impl Voices {
    pub fn load(path: &Path) -> Result<Self> {
        let file =
            std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mut npz = NpzReader::new(file)
            .with_context(|| format!("reading {} as an npz archive", path.display()))?;

        let names = npz.names().context("listing voices")?;
        let mut styles = BTreeMap::new();
        for name in names {
            let array: Array3<f32> = npz
                .by_name(&name)
                .with_context(|| format!("reading voice {name}"))?;
            // Strip the `.npy` suffix numpy uses for member names.
            let name = name.strip_suffix(".npy").unwrap_or(&name).to_string();
            styles.insert(name, array);
        }
        if styles.is_empty() {
            bail!("{} contains no voices", path.display());
        }
        Ok(Self { styles })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.styles.keys().map(String::as_str)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.styles.contains_key(name)
    }

    /// The style vector for `name` at an utterance of `token_count` tokens.
    pub fn style(&self, name: &str, token_count: usize) -> Result<Vec<f32>> {
        let array = self
            .styles
            .get(name)
            .with_context(|| format!("unknown voice {name}"))?;
        let (rows, _, dim) = array.dim();
        if dim != STYLE_DIM {
            bail!("voice {name} has style width {dim}, expected {STYLE_DIM}");
        }
        // Utterances are already truncated to the context length, but clamp
        // rather than panic if that ever changes.
        let row = token_count.min(rows - 1);
        Ok(array.slice(ndarray::s![row, 0, ..]).to_vec())
    }
}
