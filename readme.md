To develop circuit_compiler_rust, install rust, switch to nightly, then run `maturin develop` - this compiles the rust, and installs it as a python package.

Detailed instructions:
- Install `rustup` from https://rustup.rs/ . Do the default installation.
- Put it in your environment using `source ~/.cargo/env`
- Install patchelf and clang (both available on apt-get ubuntu) needed for z3
- Install `maturin`. `pip install -r ~/unity/requirements.txt`

- `cd rust_circuit`
- (Mac only) Follow the `z3` installation instructions further down before using `maturin develop`
- Run `maturin develop`
  - If `maturin develop` fails with linker errors, you can run `maturin build` then `pip install [path_to_wheel]`. Also you can add the dir containing `libpython3.9.so.1.0` or such to `LD_LIBRARY_PATH`. If you use Conda, maturin can usually find libpython by itself.
  - If `maturin develop` can't find your Z3 installation, you can use the `--features static-z3` flag to build a statically-linked copy of Z3 from source during the compilation process. If you use this flag, you must pass it to all `cargo` commands and to `maturin develop`.


If you're using vscode, we recommend the extension `rust_analyzer` and the following vscode setting:
```
    "rust-analyzer.linkedProjects": [
        "[PATH_TO_UNITY]/unity/rust_circuit/Cargo.toml"
    ],
```
which allows you to use rust_analyzer when your vscode is opened to a folder other than rust_circuit

## Publishing

We publish `rust_circuit` as a pip package, and include `rust_circuit==X.X.X` in our python `requirements_python_rust.txt`. Right before we merge a PR with Rust changes into main, we have to bump the version number in `requirements_python_rust.txt` and `Cargo.toml`, and publish the pip package. To publish, manually trigger the CircleCI pipeline with parameter `action=publish`. To publish for just your local OS and python version (which wont work for everyone on the team) and have our PyPI password, use `maturin publish`.

## Installing z3 on Mac

By default, `maturin develop` fails with a z3 related error (such as `z3.h` not found). To fix this you should install `z3` yourself and put it in the system path.

* Option 1: `brew install z3`
  * Link the dylib to your system lib path (`sudo ln -s /opt/homebrew/lib/libz3.dylib /usr/local/lib`)
  * Link the headers to your system header path (`sudo ln -s /opt/homebrew/include/z3*.h /usr/local/include`)

* Option 2: Build `z3` from source yourself
  * Clone z3 (`git clone https://github.com/Z3Prover/z3 && cd z3`)
  * Build z3 from source (see the commands under "Execute:" [here](https://github.com/Z3Prover/z3#building-z3-using-make-and-gccclang))
  * Copy the dylib to your system lib path (`sudo cp build/libz3.dylib /usr/local/lib`)
  * Copy the headers to your system header path (`sudo cp src/api/z3*.h /usr/local/include`)

* Re-run `cargo check` and you should be good to go!
* If this fails, you can also use the `--features static-z3` flag on all cargo commands and on `maturin develop`. This has slower link times but doesn't require the above to work

## About rust_circuit

Rust_circuit is a framework for expressing, manipulating, and optimally computing tensor computations, particularly geared toward computing cumulants. It's based on our python `interp/circuit` codebase. 

Some basic info: We store tensor computations/circuits in AST form in the Circuit/CircuitRc/Einsum/Add... structs. These structs often have multiple references to equivelent nodes, which we handle by hashing nodes with a cryptographic hash function (blake3 currently) and implement Eq by hash, and deduplicating on construction.

Currently, the rust_circuit codebase just optimizes circuits and computes them, and manual creation/manipulation of circuits is in the Python circuits codebase. Eventually creation/manual manipulation will be done on Rust circuits from python.

## Project Structure

We have multiple sub-crates. The subcreates, in order of dependence are

rr_util - utilities for interacting with python, including numpy-style indexes and torch tensors

circuit_base - Circuit AST types and basic operations

circuit_rewrite - rewrites and optimizations on circuits. we may want to split out optimization to a seperate crate

get_update_node - utilities for conveniently editing and manipulating circuits in jupyter notebooks

nb_operations - nb operations, often built on top of get_update_node

rust_circuit - this is the root crate, and just imports the others + has tests


## Misc advice

- You can view docs locally with `cargo doc --open`

## Profiling

To benchmark code, write tests in `benches/benches.rs`, add to Criterion group, and run `cargo bench --no-default-features`. 
Currently only simp is benchmarked

To profile code, use
`cargo bench --no-default-features  --no-run; flamegraph -o flamegraph.svg --  target/release/deps/benches-36e0a557364e8efa --nocapture`
or generally cargo bench --no-run then an executable profiler.


## Tracebacks

We filter Rust tracebacks shown to python to get rid of lots of boilerplate, set the env var `PYO3_NO_TRACEBACK_FILTER` to disable.

## Speeding up compilation

### Speeding up maturin

Maturin spends a few seconds compression your crate when you run `maturin dev` or `maturin build`. To turn this off, clone maturin locally and run `cargo run -- build --release -b bin -o dist --features faster-tests` in maturin and then install the maturin wheel. (this should really be a runtime option not compile time flag!)

Also you can stop maturin from installing rust_circuit's deps by changing maturin source code. todo: publish maturin branch with compression and deps flags

### Speeding up linking

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


## Anyhow build error after starting up in rust analyzer

[do what this comment says/read issue more generally](https://github.com/dtolnay/anyhow/issues/250#issuecomment-1209629746)
