# Contributing to Ferric

## 1. Install Rust

If you haven't installed Rust yet, follow the official installation instructions:
[Install Rust and Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html)

Once installed, keep your toolchain up to date:

```bash
rustup update stable
```

## 2. Install Coverage Tools

We use `cargo-llvm-cov` for accurate, source-based code coverage.

```bash
rustup component add llvm-tools-preview
cargo install cargo-llvm-cov --locked
```

## 3. Running Tests

```bash
cargo test
```

## 4. Generating Coverage Reports

View in browser (HTML):

```bash
cargo +nightly llvm-cov --workspace --doctests --html --open
```

Generate LCOV (for Coveralls/IDEs):

```bash
cargo +nightly llvm-cov --workspace --doctests --lcov --output-path lcov.info
```

Console summary:

```bash
cargo +nightly llvm-cov --workspace --doctests
```

## 5. IDE Integration

If you use VS Code, install the **Coverage Gutters** extension. After generating an `lcov.info` file, click **Watch** in the bottom status bar to see line-by-line coverage (green/red highlights) directly in your editor.

## 6. Expanding Macros

To expand the `make_model!` macro for debugging, install:

```bash
rustup component add rust-src
cargo install --locked cargo-expand
```

Then expand any example:

```bash
cargo expand --example grass --package ferric
cargo expand --example grass --package ferric > expanded_grass.rs
```

## 7. Crate Structure

Ferric uses two crates:

- **`ferric-macros`** — the `make_model!` procedural macro. Rust requires procedural
  macros to live in their own dedicated crate (`proc-macro = true`); a proc-macro crate
  can only export proc macros, so this crate cannot be merged into `ferric`.
- **`ferric`** — the runtime library: distributions, sampling utilities, and the
  re-exported `make_model!` macro.

## 8. Publishing to crates.io

Follow this sequence to publish both crates:

**1. Run tests and build locally:**

```bash
cargo test --workspace
cargo build --release
```

**2. Sync versions.** In both `ferric-macros/Cargo.toml` and `ferric/Cargo.toml`, set
the same release version (e.g. `0.1.4`). Ensure the path dependency in `ferric/Cargo.toml`
also pins the version:

```toml
ferric-macros = { path = "../ferric-macros", version = "0.1.4" }
```

**3. Authenticate with crates.io:**

```bash
cargo login <CRATES_IO_TOKEN>
# or for CI
export CARGO_REGISTRY_TOKEN=<CRATES_IO_TOKEN>
```

**4. Publish `ferric-macros` first** (proc-macro dependency must be published before the
crate that depends on it):

```bash
cargo package --manifest-path ferric-macros/Cargo.toml
cargo publish --manifest-path ferric-macros/Cargo.toml --dry-run
cargo publish --manifest-path ferric-macros/Cargo.toml
```

Wait 30–60 s for the crates.io index to update.

**5. Publish `ferric`:**

```bash
cargo package --manifest-path ferric/Cargo.toml
cargo publish --manifest-path ferric/Cargo.toml --dry-run
cargo publish --manifest-path ferric/Cargo.toml
```

**6. Tag the release:**

```bash
git tag -a v0.1.4 -m "Release v0.1.4"
git push origin HEAD
git push origin --tags
```

**7. Post-release housekeeping:**

- Move `Unreleased` changes into the released section of `CHANGELOG.md` and add a new
  `Unreleased` heading.
- Bump the next development version if you follow a `X.Y.Z-dev` workflow.

### Notes on crates.io and docs.rs

crates.io renders the `readme` field (the `README.md`). docs.rs renders the crate-level
`//!` documentation from `lib.rs`. Keep these distinct:

- **README** — installation, quick-start example, user-facing API guide.
- **`lib.rs` `//!`** — brief module overview, cross-links to key types and functions.

Duplicating large blocks between the two causes the crates.io and docs.rs pages to look
redundant when a user navigates between them.
