use std::collections::BTreeMap;
use std::str;

// Helpers

// Given a list of integers, return a HashMap of counts of consecutive pairs
// Example: vec[1,2,3,1,2] -> HMap {(1,2): 2, (2,3): 1, (3,1): 1}
pub fn get_stats(ids: &[i32]) -> BTreeMap<(i32, i32), usize> {
    let mut counts = BTreeMap::new();
    for pair in ids.windows(2) {
        let pair = (pair[0], pair[1]);
        *counts.entry(pair).or_insert(0) += 1;
    }
    counts
}

pub fn merge(ids: &[i32], pair: (i32, i32), idx: i32) -> Vec<i32> {
    let mut newids = Vec::with_capacity(ids.len());
    let mut i = 0;
    while i < ids.len() {
        if i < ids.len() - 1 && ids[i] == pair.0 && ids[i + 1] == pair.1 {
            newids.push(idx);
            i += 2;
        } else {
            newids.push(ids[i]);
            i += 1;
        }
    }
    newids
}

pub fn build_vocab(merges: &BTreeMap<(i32, i32), i32>, special_tokens: &BTreeMap<String, i32>) -> BTreeMap<i32, Vec<u8>> {
    let mut vocab: BTreeMap<i32, Vec<u8>> = (0..256).map(|idx| (idx as i32, vec![idx as u8])).collect();
    for (&(p0, p1), &idx) in merges {
        if let Some(vocab_entry) = vocab.get(&p0) {
            let mut merged_vec = Vec::with_capacity(vocab_entry.len() + vocab[&p1].len());
            merged_vec.extend_from_slice(vocab_entry);
            merged_vec.extend_from_slice(&vocab[&p1]);
            vocab.insert(idx, merged_vec);
        }
    }

    for (special, &idx) in special_tokens {
        vocab.insert(idx, special.as_bytes().to_vec());
    }
    vocab
}

pub fn b_as_literal(bytes: &[u8]) -> String {
    let mut formatted = String::with_capacity(bytes.len() * 4 + 2); // Optimistic pre-allocation
    formatted.push('b');
    formatted.push('"');

    for &byte in bytes {
        if byte.is_ascii_graphic() || byte.is_ascii_whitespace() {
            formatted.push(byte as char);
        } else {
            formatted.push_str(&format!("\\x{:02x}", byte));
        }
    }

    formatted.push('"');
    formatted
}

fn replace_control_characters(s: &str) -> String {
    s.chars()
        .map(|ch| {
            if !ch.is_control() {
                ch.to_string()
            } else {
                format!("\\u{:04x}", ch as u32)
            }
        })
        .collect()
}

pub fn render_token(t: &[u8]) -> String {
    let s = match str::from_utf8(t) {
        Ok(valid_str) => valid_str.to_string(),
        Err(_) => String::from_utf8_lossy(t).to_string(),
    };
    replace_control_characters(&s)
}