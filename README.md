[![Github Actions Tests](https://github.com/ferric-ai/ferric/actions/workflows/ci.yml/badge.svg)](https://github.com/Ferric-AI/ferric/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/ferric.svg)](https://crates.io/crates/ferric)
[![Coverage Status](https://coveralls.io/repos/github/Ferric-AI/ferric/badge.svg)](https://coveralls.io/github/Ferric-AI/ferric)

# Ferric
A Probabilistic Programming Language in Rust with a declarative syntax.

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
ferric = "0.1"
```

## Example

```rust
use std::time::Instant;
use ferric::make_model;

make_model! {
    mod grass;
    use ferric::distributions::Bernoulli;

    let rain : bool ~ Bernoulli::new( 0.2 );

    let sprinkler : bool ~
        if rain {
            Bernoulli::new( 0.01 )
        } else {
            Bernoulli::new( 0.4 )
        };

    let grass_wet : bool ~ Bernoulli::new(
        if sprinkler && rain { 0.99 }
        else if sprinkler && !rain { 0.9 }
        else if !sprinkler && rain { 0.8 }
        else { 0.0 }
    );

    observe grass_wet;
    query rain;
    query sprinkler;
}

fn main() {
    let model = grass::Model {grass_wet: true};
    let mut num_rain = 0;
    let mut num_sprinkler = 0;
    let num_samples = 100000;
    let start = Instant::now();
    for sample in model.sample_iter().take(num_samples) {
        if sample.rain {
            num_rain += 1;
        }
        if sample.sprinkler {
            num_sprinkler += 1;
        }
    }
    let num_samples = num_samples as f64;
    println!(
        "posterior: rain = {} sprinkler = {}. Elapsed {} millisec for {} samples",
        (num_rain as f64) / num_samples,
        (num_sprinkler as f64) / num_samples,
        start.elapsed().as_millis(),
        num_samples,
    );
}
```

## License

## How it works

Ferric's `make_model!` macro declares a probabilistic model and the
relationships between random variables. Within the macro you:

- Define random variables and their generative distributions.
- Mark variables with `observe` to indicate data the model will be
    conditioned on.
- Mark variables with `query` to indicate values you want returned in
    posterior samples.

After expansion, the macro produces a module containing a `Model` type.
You construct the model by supplying values for the observed fields
(for example `let model = grass::Model { grass_wet: true };`). Use the
model's sampling API (for example `model.sample_iter()`) to draw
samples from the posterior; each sample contains the queried variables
as fields that you can inspect to estimate posterior probabilities.

Refer to the `Example` above for a canonical, end-to-end usage sample.

Licensed under either of

 * [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0)
 * [MIT license](http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

## 🛠 Development & Testing

## 1. Install Rust
If you haven't installed Rust yet, follow the official installation instructions here:
[Install Rust and Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html)
Once installed, ensure your toolchain is up to date:

```bash
rustup update stable
```

## 2. Install Coverage Tools
We use `cargo-llvm-cov` for accurate, source-based code coverage. You will need to install the tool and the required LLVM components:

```bash
# Install the LLVM tools component
rustup component add llvm-tools-preview

# Install cargo-llvm-cov
cargo install cargo-llvm-cov --locked
```

## 3. Running Tests
To run the standard test suite:

```bash
cargo test
```

## 4. Generating Coverage Reports
You can generate coverage data in multiple formats to match our CI environment or for local inspection:

* View in Browser (HTML):

```bash
cargo +nightly llvm-cov --workspace --doctests --html --open
```

* Generate LCOV (for Coveralls/IDEs):

```bash
cargo +nightly llvm-cov --workspace --doctests --lcov --output-path lcov.info
```

* Console Summary:

```bash
cargo +nightly llvm-cov --workspace --doctests
```

## 5. IDE Integration (Optional)
If you use VS Code, install the Coverage Gutters extension. After generating an `lcov.info` file, click the Watch button in the bottom status bar to see line-by-line coverage (green/red highlights) directly in your editor.

## Developer

If you're working on the Ferric codebase itself and need to expand the `make_model!` macro for debugging or inspection, install the following developer tools:

```bash
# Install the rust source component (required by some expansion tools)
rustup component add rust-src

# Install cargo-expand to make `cargo expand` available
cargo install --locked cargo-expand
```

Then you can expand the `grass` example (or any example) with:

```bash
# Expand the `grass` example and print expanded Rust to stdout
cargo expand --example grass --package ferric

# Or save the expanded output to a file for easier reading
cargo expand --example grass --package ferric > expanded_grass.rs
```

Note: `cargo expand` is a separate tool (provided by the `cargo-expand` crate) and is not included in the default Cargo installation.

## Publishing to crates.io

Follow this updated sequence to publish the crate and any internal dependencies (recommended):

- 1) Run tests & build locally:

```bash
cargo test --workspace
cargo build --release
```

- 2) Ensure versions are synced and `path` deps include `version`:

    - In `ferric-macros/Cargo.toml` set the release version (e.g. `0.1.3`).
    - In `ferric/Cargo.toml` reference the local path and published version:

```toml
ferric-macros = { path = "../ferric-macros", version = "0.1.3" }
```

- 3) Obtain crates.io authentication (one-time or CI):

```bash
cargo login <CRATES_IO_TOKEN>
# or (for CI/session)
export CARGO_REGISTRY_TOKEN=<CRATES_IO_TOKEN>
```

- 4) Publish internal/proc-macro crates first (if any). For `ferric-macros`:

```bash
cargo package --manifest-path ferric-macros/Cargo.toml
cargo publish --manifest-path ferric-macros/Cargo.toml --dry-run
cargo publish --manifest-path ferric-macros/Cargo.toml
```

    - Wait a short time (30–60s) after publishing for the crates.io index to update.

- 5) Package & dry-run the main crate, then publish:

```bash
cargo package --manifest-path ferric/Cargo.toml
cargo publish --manifest-path ferric/Cargo.toml --dry-run
cargo publish --manifest-path ferric/Cargo.toml
```

    - If `cargo package` fails with a dependency version error, it means a required dependency version is not yet available on crates.io — publish that dependency first.

- 6) Tag and push git commits:

```bash
git tag -a v0.1.3 -m "Release v0.1.3"
git push origin HEAD
git push origin --tags
```

- 7) Post-release housekeeping:

    - Move `Unreleased` changes into the released section of `CHANGELOG.md` and add a new `Unreleased` heading.
    - Bump the next development version if you follow a `X.Y.Z-dev` workflow.
    - Publish docs or update the website and announce the release.

Notes and tips:

- Always publish internal dependencies (proc macros, helper crates) before crates that depend on them.
- Keep `path` dependencies during development but include a `version` so `cargo package` validates.
- If the main crate cannot find the new dependency version immediately after publishing, wait a short time and retry.
- Verify `readme`, `license`, and `repository` fields in each `Cargo.toml` to ensure crates.io displays metadata correctly.

