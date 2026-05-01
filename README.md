[![Github Actions Tests](https://github.com/ferric-ai/ferric/actions/workflows/ci.yml/badge.svg)](https://github.com/Ferric-AI/ferric/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/ferric.svg)](https://crates.io/crates/ferric)
[![Coverage Status](https://coveralls.io/repos/github/Ferric-AI/ferric/badge.svg)](https://coveralls.io/github/Ferric-AI/ferric)

# Ferric

A Probabilistic Programming Language in Rust with a declarative syntax.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
ferric = "0.1"
```

## How it works

Ferric's `make_model!` macro declares a Bayesian model and the relationships between random
variables. Inside the macro you:

- Define random variables and their distributions using `let name : Type ~ Distribution;`.
- Mark variables with `observe` to condition the model on observed data.
- Mark variables with `query` to include variables in posterior samples.

After expansion the macro produces a module containing a `Model` struct.  Construct the model
by supplying values for the observed fields, then draw from the posterior using one of the two
sampling strategies below.

## Sampling strategies

### Rejection sampling — `sample_iter`

Valid for **discrete observations only**.  Each call to `next()` draws from the prior and
discards the sample if the discrete observations don't match.  Every returned `Sample` is an
exact draw from the posterior.

```rust
use ferric::make_model;

make_model! {
    mod grass;
    use ferric::distributions::Bernoulli;

    let rain       : bool ~ Bernoulli::new(0.2);
    let sprinkler  : bool ~ if rain { Bernoulli::new(0.01) } else { Bernoulli::new(0.4) };
    let grass_wet  : bool ~ Bernoulli::new(
        if sprinkler && rain  { 0.99 }
        else if sprinkler     { 0.90 }
        else if rain          { 0.80 }
        else                  { 0.00 }
    );

    observe grass_wet;
    query rain;
    query sprinkler;
}

fn main() {
    let model = grass::Model { grass_wet: true };
    let num_samples = 100_000;
    let mut num_rain = 0;
    let mut num_sprinkler = 0;

    for sample in model.sample_iter().take(num_samples) {
        if sample.rain      { num_rain      += 1; }
        if sample.sprinkler { num_sprinkler += 1; }
    }

    println!(
        "P(rain | wet) ≈ {:.3}   P(sprinkler | wet) ≈ {:.3}",
        num_rain      as f64 / num_samples as f64,
        num_sprinkler as f64 / num_samples as f64,
    );
}
```

### Weighted sampling — `weighted_sample_iter`

Valid for **any model**, including those with continuous observations.  Each call to `next()`
draws from the prior and returns a `WeightedSample { log_weight, sample }` pair where
`log_weight` is the sum of the log-likelihoods of all observations given the draw.

Use `ferric::weighted_mean` and `ferric::weighted_std` to compute posterior summaries from the
weighted samples.

```rust
use ferric::make_model;

make_model! {
    mod signal_estimation;
    use ferric::distributions::Normal;

    // prior: true signal unknown
    let true_signal     : f64 ~ Normal::new(0.0, 2.0);
    // likelihood: noisy sensor reading
    let sensor_reading  : f64 ~ Normal::new(true_signal, 1.0);

    observe sensor_reading;
    query  true_signal;
}

fn main() {
    let model = signal_estimation::Model { sensor_reading: 2.5 };
    let num_samples = 100_000;

    let mut signal_vals  = Vec::with_capacity(num_samples);
    let mut log_weights  = Vec::with_capacity(num_samples);

    for ws in model.weighted_sample_iter().take(num_samples) {
        signal_vals.push(ws.sample.true_signal);
        log_weights.push(ws.log_weight);
    }

    let post_mean = ferric::weighted_mean(&signal_vals, &log_weights);
    let post_std  = ferric::weighted_std(&signal_vals,  &log_weights);

    println!("posterior: true_signal = {:.3} ± {:.3}", post_mean, post_std);
    // Analytical answer: mean ≈ 2.0, std ≈ 0.894
}
```

The `WeightedSample` type nests query variables under `.sample.*` and exposes the metadata
separately at `.log_weight`, so there is no naming conflict even if a query variable is
named `log_weight`.

## Available distributions

| Distribution | Domain | Parameters |
|---|---|---|
| `Bernoulli` | `bool` | `p ∈ [0, 1]` |
| `Binomial` | `u64` | `n ≥ 1`, `p ∈ (0, 1)` |
| `Geometric` | `u64` | `p ∈ (0, 1]` |
| `Poisson` | `u64` | `rate > 0` |
| `Uniform` | `f64` | `low < high` |
| `Exponential` | `f64` | `rate > 0` |
| `Normal` | `f64` | `mean`, `std_dev > 0` |
| `LogNormal` | `f64` | `mu`, `sigma > 0` |
| `Beta` | `f64` | `alpha > 0`, `beta > 0` |
| `Gamma` | `f64` | `shape > 0`, `scale > 0` |
| `StudentT` | `f64` | `df > 0` |
| `Cauchy` | `f64` | `median`, `scale > 0` |

## License

Licensed under either of

- [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0)
- [MIT license](http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above,
without any additional terms or conditions.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coverage tools, macro
expansion, and publishing instructions.
