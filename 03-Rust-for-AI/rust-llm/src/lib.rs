use pyo3::prelude::*;

mod helpers;
mod bpe;

#[pymodule]
fn rust_llm(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<bpe::BPETokenizer>()?;
    Ok(())
}