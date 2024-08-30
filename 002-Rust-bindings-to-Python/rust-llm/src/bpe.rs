use pyo3::prelude::*;
use pyo3::prepare_freethreaded_python;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use crate::helpers::{get_stats, merge, render_token, build_vocab, b_as_literal};

#[pyclass]
pub struct BPETokenizer {
    merges: BTreeMap<(i32, i32), i32>,   // (pair, idx)
    pattern: String,                     // pattern to split the text into tokens
    special_tokens: BTreeMap<String, i32>, // special tokens (string, idx)
    vocab: BTreeMap<i32, Vec<u8>>,       // (idx, token)
}

#[pymethods]
impl BPETokenizer {
    #[new]
    pub fn new() -> Self {
        prepare_freethreaded_python();
        let merges = BTreeMap::<(i32, i32), i32>::new();
        let pattern = String::new();
        let special_tokens = BTreeMap::<String, i32>::new();
        let vocab = build_vocab(&merges, &special_tokens);

        BPETokenizer {
            merges,
            pattern,
            special_tokens,
            vocab,
        }
    }

    fn train(&mut self, text: &str, vocab_size: usize, verbose: bool) {
        assert!(vocab_size >= 256);
        let num_merges = vocab_size - 256;
    
        #[cfg(debug_assertions)]
        {
            println!("[DEBUG] Starting training with vocab_size = {}", vocab_size);
            println!("[DEBUG] Number of merges to perform: {}", num_merges);
        }
    
        // Input text preprocessing
        let text_bytes = text.as_bytes();
        let mut ids: Vec<i32> = text_bytes.iter().map(|&b| b as i32).collect();
    
        #[cfg(debug_assertions)]
        println!("[DEBUG] Initial IDs: {:?}", &ids[..100.min(ids.len())]);
    
        for i in 0..num_merges {
            let stats = get_stats(&ids);
            let pair = *stats.iter().max_by_key(|&(_, count)| count).unwrap().0;
            let idx = 256 + i as i32;
            ids = merge(&ids, pair, idx);
    
            self.merges.insert(pair, idx);
    
            // Separate immutable borrow before mutable borrow
            let pair_1_len = self.vocab[&pair.1].len();
            if let Some(vocab_entry) = self.vocab.get_mut(&pair.0) {
                let mut new_token = Vec::with_capacity(vocab_entry.len() + pair_1_len);
                new_token.extend_from_slice(vocab_entry);
                new_token.extend_from_slice(&self.vocab[&pair.1]);
                self.vocab.insert(idx, new_token);
            }
    
            if verbose {
                println!(
                    "merge {}/{}: {:?} -> {} ({:?}) had {} occurrences",
                    i + 1,
                    num_merges,
                    pair,
                    idx,
                    self.vocab[&idx],
                    stats[&pair]
                );
            }
        }
    
        #[cfg(debug_assertions)]
        {
            println!("[DEBUG] Merges: {:?}", self.merges);
            for (key, value) in &self.vocab {
                let formatted_value = b_as_literal(value);
                println!("{}: {}", key, formatted_value);
            }
        }
    }

    pub fn decode(&self, ids: Vec<i32>) -> String {
        let mut text_bytes = Vec::with_capacity(ids.len() * 2); // Pre-allocate assuming 2 bytes per id on average
        for &id in &ids {
            text_bytes.extend_from_slice(&self.vocab[&id]);
        }
        String::from_utf8_lossy(&text_bytes).into_owned()
    }

    pub fn save_model(&self, file_prefix: &str) -> std::io::Result<()> {        
        let model_file = format!("{}.model", file_prefix);
        let mut model_file = File::create(&model_file)?;
        writeln!(model_file, "minbpe v1")?;
        writeln!(model_file, "{}", self.pattern)?;
        writeln!(model_file, "{}", self.special_tokens.len())?;
    
        for (special, idx) in &self.special_tokens {
            writeln!(model_file, "{} {}", special, idx)?;
        }
    
        for (&(idx1, idx2), _) in &self.merges {
            writeln!(model_file, "{} {}", idx1, idx2)?;
        }
    
        let vocab_file = format!("{}.vocab", file_prefix);
        let mut vocab_file = File::create(&vocab_file)?;
        let inverted_merges: BTreeMap<i32, (i32, i32)> = self.merges
            .iter()
            .map(|(&(p0, p1), &idx)| (idx, (p0, p1)))
            .collect();
    
        for (&idx, token) in &self.vocab {
            let s = render_token(token);
            if let Some(&(idx0, idx1)) = inverted_merges.get(&idx) {
                let s0 = render_token(&self.vocab[&idx0]);
                let s1 = render_token(&self.vocab[&idx1]);
                writeln!(vocab_file, "[{}][{}] -> [{}] {} \n", s0, s1, s, idx)?;
            } else {
                writeln!(vocab_file, "[{}] {} \n", s, idx)?;
            }
        }
        Ok(())
    }

    pub fn load_model(&mut self, file_prefix: &str, verbose: bool) -> std::io::Result<()> {
        if verbose {
            println!("[DEBUG] Loading model with prefix: {}", file_prefix);
        }

        let model_file = File::open(file_prefix)?;
        let reader = BufReader::new(model_file);

        let mut lines = reader.lines();
        let version = lines.next().unwrap().unwrap();
        assert_eq!(version.trim(), "minbpe v1", "Invalid model file version");

        self.pattern = lines.next().unwrap().unwrap().trim().to_string();

        let num_special_tokens = lines.next().unwrap().unwrap().parse::<i32>().unwrap();
        let mut special_tokens = BTreeMap::<String, i32>::new();

        for _ in 0..num_special_tokens {
            let line = lines.next().unwrap().unwrap();
            let parts: Vec<&str> = line.split_whitespace().collect(); // getting special, special_idx
            special_tokens.insert(parts[0].to_string(), parts[1].parse::<i32>().unwrap());
        }

        let mut merges = BTreeMap::<(i32, i32), i32>::new();
        let mut idx = 256;
        for line in lines {
            let line = line.unwrap();
            let parts: Vec<&str> = line.split_whitespace().collect();
            let idx1 = parts[0].parse::<i32>().unwrap();
            let idx2 = parts[1].parse::<i32>().unwrap();
            merges.insert((idx1, idx2), idx);
            idx += 1;
        }

        self.merges = merges;
        self.special_tokens = special_tokens;
        self.vocab = build_vocab(&self.merges, &self.special_tokens);
        Ok(())
    }

    pub fn encode(&self, text: &str, verbose: bool) -> Vec<i32> {
        if verbose {
            println!("[DEBUG] Encoding text: {}", text);
        }
        let mut ids: Vec<i32> = text.bytes().map(|b| b as i32).collect();
        let merges = &self.merges;
        if verbose {
            println!("[DEBUG] Initial IDs: {:?}", ids);
            println!("[DEBUG] Merges loaded: {:?}", merges);
        }
    
        while ids.len() >= 2 {
            let stats = get_stats(&ids);
            if verbose {
                println!("[DEBUG] Current IDs: {:?}", ids);
                println!("[DEBUG] Stats: {:?}", stats);
            }
            let pair = stats
                .keys()
                .min_by_key(|&&p| merges.get(&p).unwrap_or(&i32::MAX))
                .copied()
                .unwrap();
        
            if !merges.contains_key(&pair) {
                break;
            }
        
            let new_id = merges[&pair];
            ids = merge(&ids, pair, new_id);
            if verbose {
                println!("[DEBUG] Pair: {:?}", pair);
                println!("[DEBUG] New ID for merge: {}", new_id);
            }
        }

        if verbose {
            println!("[DEBUG] Encoding complete. Final IDs: {:?}", ids);
        }
        ids
    }   
}