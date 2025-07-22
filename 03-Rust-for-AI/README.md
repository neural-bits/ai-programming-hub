# Rust Library to Python Bindings

This project showcases how to implement the BPETokenizer (inspired by Andrej Karpathy's Python Implementation) in Rust, and then generate Python bindings using PyO3 and Maturin build tool.

**Here's what we'll learn**

- What is LLVM, How rust works
- What is PyO3 and Maturin
- What do libraries such as Polars, and Bytewax have in common (Rust core)
- How to port Python code to Rust
- How to build a Rust Library
- How to test library import in Python

### Resources
- 📝[Full Blog Article](https://neuralbits.substack.com/p/lets-build-andrej-karpathys-bpetokenizer)

---

## Table of Contents
  - [Dependencies](#dependencies)
  - [Install](#install)
  - [Usage](#usage)
------

### Dependencies
- [Python (version 3.11)](https://www.python.org/downloads/)
- [GNU Make](https://www.gnu.org/software/make/)
- [Conda](https://docs.anaconda.com/miniconda/)


### Install
- As we use a Makefile, to install and prepare the env, you have to run the following:
  ```shell
  make install
  ```



### Usage
First, build the Rust Library using:
```shell
make build_debug  || make build_rel
```
This will do the following:
- Build library
- Generate a `.whl` wheel for Python at `./targets/wheels/*.whl`
- Install the wheel to current Python environment.
    
Next, you can test the Library  using the `playground.ipynb` Jupyter Notebook.
- Quickest way is to run cells: `1 -> 16 -> 20`
   

