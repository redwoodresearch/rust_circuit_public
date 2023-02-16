# Rust Circuit

Rust_circuit is a library for expressing and manipulating tensor computations for neural network interpretability, written in Rust and used in Python notebooks. It includes support for causal scrubbing. 

Linux and M1 Mac are supported - Windows has not been tested and probably does not work.

This library is mainly intended for REMIX participants and former Redwood staff that want to continue using Circuits outside of Redwood. Other people are welcome to try and use it, but it's likely to be a rough time especially if you don't know Rust.

## Installation

Note: the `pip` version of `rust_circuit` depends on Redwood's internal code, so you won't be able to use that directly - you have to build the Rust code in this repository from source.

### Building from Source on Linux

- Install `rustup` from [https://rustup.rs](https://rustup.rs) . Do the default installation.
- Install `clang`: `sudo apt install clang`
- Install Miniconda from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
- Create a new Python 3.11 environment: `conda create -n circ python=3.11 -y`
- Activate the environment: `conda activate circ`
- Install PyTorch from pip:  `pip install -f https://download.pytorch.org/whl/torch_stable.html torch==1.13.1+cu116`
- Install more dependencies from pip: `pip install maturin[patchelf] attrs cattrs blake3 numpy msgpack websockets`
- Build rust_circuit (this will take a few minutes): `maturin develop --features static-z3`
- Now, in the Python interpreter you should be able to `import rust_circuit`.

### Building from Source on M1 Mac

- Install `rustup` from [https://rustup.rs](https://rustup.rs) . Do the default installation.
- Install Miniconda from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
- Create a new Python 3.11 environment: `conda create -n circ python=3.11 -y`
- Activate the environment: `conda activate circ`
- Install PyTorch through conda: `conda install pytorch -c pytorch`
- Install more dependencies from pip: `pip install maturin[patchelf] attrs cattrs blake3 numpy msgpack websockets`
- Install the `Homebrew` package manager from [https://brew.sh/](https://brew.sh/)
- Install z3 through brew: `brew install z3`
  - Link the dylib to your system lib path (`sudo ln -s /opt/homebrew/lib/libz3.dylib /usr/local/lib`)
  - Link the headers to your system header path (`sudo ln -s /opt/homebrew/include/z3*.h /usr/local/include`)
- Build rust_circuit: `maturin develop`
- Now, in the Python interpreter you should be able to `import rust_circuit`.

## VS Code Configuration

- I recommend the [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer) plugin.

## Learning Circuits

The [REMIX curriculum](https://github.com/redwoodresearch/remix_public) is open source and is the best way to learn about how to use this library.

The next best thing is to [learn the Rust language](https://doc.rust-lang.org/book/) and just read the source code directly. 

Unfortunately, many of the demo notebooks in `python/rust_circuit/demos` and unit tests in `python/tests` don't run without internal Redwood code.

## Troubleshooting

### Anyhow build error after starting up in rust analyzer

[do what this comment says/read issue more generally](https://github.com/dtolnay/anyhow/issues/250#issuecomment-1209629746)

### Can't find libpython

If you see an error about not being able to find `libpython`, you'll need to find it on your system and then add the containing folder to the `LD_LIBRARY_PATH` environment variable. On my machine, it was in `/home/ubuntu/miniconda3/envs/circ/lib/`. If you don't know where it is, try the [find_libpython](https://pypi.org/project/find-libpython/) tool.

### Debugging Rust code

You can debug into Rust by running the Python interpreter under a debugger of your choice. For example, with LLDB: `lldb -- /path/to/python -c "import rust_circuit"`

Also, you can set the environment variable RUST_BACKTRACE=1 to see more information about errors.

## Optional Info Below - Speeding up compilation

The below is only useful if you're going to be modifying and frequently compiling Circuits.

### Speeding up maturin

Maturin spends a few seconds compressing your crate when you run `maturin dev` or `maturin build`. To turn this off, clone maturin locally and run `cargo run -- build --release -b bin -o dist --features faster-tests` in maturin and then install the maturin wheel. (this should really be a runtime option not compile time flag!)

Also you can stop maturin from installing rust_circuit's deps by changing maturin source code. todo: publish maturin branch with compression and deps flags

### Speeding up linking

#### Building z3 from source

If you do this, you can dynamically link against z3 and don't have to specify `--features static-z3`.

- Option 1: `brew install z3`
  - Link the dylib to your system lib path (`sudo ln -s /opt/homebrew/lib/libz3.dylib /usr/local/lib`)
  - Link the headers to your system header path (`sudo ln -s /opt/homebrew/include/z3*.h /usr/local/include`)

- Option 2: Build `z3` from source yourself
  - Clone z3 (`git clone https://github.com/Z3Prover/z3 && cd z3`)
  - Build z3 from source (see the commands under "Execute:" [here](https://github.com/Z3Prover/z3#building-z3-using-make-and-gccclang))
  - Copy the dylib to your system lib path (`sudo cp build/libz3.dylib /usr/local/lib`)
  - Copy the headers to your system header path (`sudo cp src/api/z3*.h /usr/local/include`)

#### Installing a faster linker

Highly optimized linkers exist, and can speed up compile times by like 3 seconds. Either `lld` (2nd fastest linker) or `mold` (fastest linker) should mostly get link time down to being basically negligible.

Linkers can be configured with the env var `RUSTFLAGS="-C link-arg=-fuse-ld=/PATH/TO/MY/LINKER"`.

You might need to add `"rust-analyzer.checkOnSave.extraEnv": {"RUSTFLAGS": "-C link-arg=-fuse-ld=/PATH/TO/MY/LINKER"},` to vscode settings (Tao had to do this to make rust-analyzer work)

You can instead add this to your global cargo config at `~/.cargo/config` or `~/.cargo/config.toml` (This is what Ryan uses):

```
[target.x86_64-unknown-linux-gnu]
linker = "clang"
rustflags = ["-C", "link-arg=-fuse-ld=/PATH/TO/MY/LINKER"]
```

To use lld in particular,  `sudo apt install lld` and then configure using one
of the above approaches with `/PATH/TO/MY/LINKER` replace with `lld` (or the absolute path
to the binary).

The mold linker is maybe annoying to install on ubuntu, but I (Ryan) had no issues installing on Arch.

## Profiling (doesn't work)

To benchmark code, write tests in `benches/benches.rs`, add your new tests to `criterion_group!` inside `benches.rs`, then run `cargo bench --no-default-features`.
Currently only simp is benchmarked.

To profile code, use
`cargo bench --no-default-features  --no-run; flamegraph -o flamegraph.svg --  target/release/deps/benches-36e0a557364e8efa --nocapture`
or generally cargo bench --no-run then an executable profiler.

## Tracebacks

We filter Rust tracebacks shown to Python to get rid of lots of boilerplate, set the environment variable `PYO3_NO_TRACEBACK_FILTER` to disable.
