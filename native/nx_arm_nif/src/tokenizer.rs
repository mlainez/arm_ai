//! Tokenizer bridge via the upstream `tokenizers` crate.
//!
//! Lets the Elixir side say `"Hello, world"` instead of
//! `[1, 15043, 29892, 3186]`. We load a `tokenizer.json` (the
//! HuggingFace canonical format) into a ResourceArc and expose
//! encode / decode NIFs.

use std::sync::Mutex;
use tokenizers::tokenizer::Tokenizer;

pub struct TokenizerResource {
    pub inner: Mutex<Tokenizer>,
}

unsafe impl Send for TokenizerResource {}
unsafe impl Sync for TokenizerResource {}

pub fn load(path: &str) -> Result<TokenizerResource, String> {
    let tok = Tokenizer::from_file(path)
        .map_err(|e| format!("Tokenizer::from_file({}) failed: {}", path, e))?;
    Ok(TokenizerResource {
        inner: Mutex::new(tok),
    })
}

pub fn encode(res: &TokenizerResource, text: &str, add_special: bool) -> Result<Vec<u32>, String> {
    let tok = res.inner.lock().map_err(|e| format!("lock: {}", e))?;
    let enc = tok
        .encode(text, add_special)
        .map_err(|e| format!("encode failed: {}", e))?;
    Ok(enc.get_ids().to_vec())
}

pub fn decode(res: &TokenizerResource, ids: &[u32], skip_special: bool) -> Result<String, String> {
    let tok = res.inner.lock().map_err(|e| format!("lock: {}", e))?;
    tok.decode(ids, skip_special)
        .map_err(|e| format!("decode failed: {}", e))
}
