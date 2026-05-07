[![Github Actions Tests](https://github.com/Ferric-AI/ferric/actions/workflows/ci.yml/badge.svg)](https://github.com/Ferric-AI/ferric/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/ferric.svg)](https://crates.io/crates/ferric)
[![Coverage Status](https://coveralls.io/repos/github/Ferric-AI/ferric/badge.svg)](https://coveralls.io/github/Ferric-AI/ferric)

# Ferric

A Probabilistic Programming Language in Rust with a declarative syntax.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
ferric = "0.2"
```

## How it works

Ferric's `make_model!` macro declares a Bayesian model and the relationships between random
variables. Inside the macro you:

- Define random variables and their distributions using `let name : Type ~ Distribution;`.
- Define model constants using `const name : Type;`.
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
    name grass;
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
    name signal_estimation;
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

### User-proposal importance sampling — `importance_sampler`

For continuous models where drawing latents from the prior is inefficient, Ferric also
generates a small proposer API.  Each model module includes:

- `ObservedData`: a generated struct containing the model constants and observations.
- `Proposal`: a generated struct whose fields correspond to latent stochastic variables.
- `Proposer<R>`: a trait whose `initialize` method receives `ObservedData`, and whose
  `propose` method returns one `Proposal`.

The proposal's `log_prob` must be the joint log probability of every value supplied in the
proposal.  Fields left as `None` are drawn from the model prior.  Ferric computes the sample
weight as `log p_model(proposed values) - log q(proposed values) + log p(observations | world)`.
For diagnostics, `importance_sampler_debug(proposer, n)` traces the first `n` worlds, printing
the proposal, model-prior terms for proposed values, observed likelihood terms, sampled
stochastic values, and final log weight before continuing as a normal iterator.
The rats example also exposes this through `FERRIC_DEBUG_IMPORTANCE`; for example,
`FERRIC_DEBUG_IMPORTANCE=1 cargo run -p ferric --example rats` traces the first
importance sample in each rats experiment.

```rust
use ferric::distributions::{Distribution, Normal};
use ferric::make_model;

make_model! {
    name signal_is;
    use ferric::distributions::Normal;

    let signal : f64 ~ Normal::new(0.0, 2.0);
    let reading : f64 ~ Normal::new(signal, 1.0);

    observe reading;
    query signal;
}

struct SignalProposer {
    dist: Option<Normal>,
}

impl signal_is::Proposer<rand::rngs::ThreadRng> for SignalProposer {
    fn initialize(&mut self, data: &signal_is::ObservedData) {
        self.dist = Some(Normal::new(data.reading, 1.0).unwrap());
    }

    fn propose(&mut self, rng: &mut rand::rngs::ThreadRng) -> signal_is::Proposal {
        let dist = self.dist.as_ref().unwrap();
        let signal = dist.sample(rng);
        let log_prob = <Normal as Distribution<rand::rngs::ThreadRng>>::log_prob(dist, &signal);
        let mut proposal = signal_is::Proposal::new(log_prob);
        proposal.signal = Some(signal);
        proposal
    }
}
```

## Indexed random variables

Ferric can declare arrays of random variables by adding one or more index
ranges after the variable name:

```rust
use ferric::make_model;

make_model! {
    name indexed_survival;
    use ferric::distributions::Beta;
    use ferric::distributions::Bernoulli;

    const n : u64;
    const t : u64;

    let survival : f64 ~ Beta::new(99.0, 1.0);
    let alive[person of n, time of t] : bool ~ if time == 0 {
        Bernoulli::new(1.0)
    } else if alive[person, time - 1] {
        Bernoulli::new(survival)
    } else {
        Bernoulli::new(0.0)
    };
    let age[person of n] : u64 = {
        let mut age = t;
        for time in 0..t {
            if !alive[person, time] {
                age = time;
                break;
            }
        }
        age
    };

    observe age;
    query survival;
}
```

Each index range has the form `index_name of upper_bound`. The upper bound
must be a constant or an earlier variable in the model. The index takes values
from `0` through `upper_bound - 1`; if the upper bound is zero, the indexed
variable has no values. The generated query type is a nested `Vec`, so `alive`
is `Vec<Vec<bool>>` and `age` is `Vec<u64>`.

Constants are supplied when the generated `Model` is constructed:

```rust
let model = indexed_survival::Model {
    n: 3,
    t: 10,
    age: vec![4, 7, 10],
};
```

Observed indexed stochastic variables use nested vectors with `Option<T>` at
the leaves. `Some(value)` is an observed value and `None` masks a missing
entry. Directly observed indexed variables must have constant-valued bounds,
or variable bounds that are themselves observed.

An index upper bound may also be stochastic:

```rust
make_model! {
    name random_length;
    use ferric::distributions::Bernoulli;
    use ferric::distributions::Poisson;

    const max_n : u64;

    let n : u64 ~ Poisson::new(4.0) max max_n;
    let flips[flip of n] : bool ~ Bernoulli::new(0.5);
    let num_heads : u64 = flips.iter().filter(|&&flip| flip).count() as u64;

    observe num_heads;
    query n;
}

let model = random_length::Model {
    max_n: 10,
    num_heads: 4,
};
```

When a stochastic variable is used as an index bound, Ferric requires an
explicit `max ...` annotation on that variable. The max value must be a
literal or a previously declared constant of the same type as the variable.
The annotation bounds the random variable's domain. For example,
`n ~ Poisson::new(3.0) max 100` means `n` may take values `0..=100`, and
likelihoods are normalized over that bounded domain using
`Distribution::log_cum_prob`.

Distribution and deterministic expressions inside indexed variables can refer
to the explicitly named indices. Deterministic variables can depend on indexed
variables, including arrays whose size depends on a stochastic bound. This
supports observed aggregates such as counts or summaries when the raw array
cardinality is latent. See `examples/urn_unknown_marbles.rs` for a complete
unknown-urn-size example.

### Gelfand Rats Model

The rats example in `examples/rats.rs` writes the classic Gelfand hierarchical
growth model with the full observed 30-rat weight table. It is included as a
language example for indexed continuous observations and hierarchical
dependencies. The current inference methods are not intended to solve this
model well; later samplers will make this kind of posterior practical.

### Radar Simulation Sketch

The radar example in `examples/radar.rs` shows a richer simulation-only model.
The number of aircraft is Poisson-bounded. Each aircraft has a 3D latent path:
the first timestep is drawn from a broad Gaussian, and later timesteps move
with a narrow Gaussian transition around the previous location. Real blips are
small-Gaussian measurements around their aircraft. False blips have their own
bounded count at each timestep and broad-Gaussian locations. A deterministic
query collects the set of all real and false blip locations for every timestep.
This example is meant for forward simulation; later inference methods will make
conditioning on the generated blip sets practical.

```rust
make_model! {
    name radar;
    use ferric::distributions::MultivariateNormal;
    use ferric::distributions::Poisson;
    use nalgebra::DVector;
    use std::collections::HashSet;
    use super::BlipLocation;
    use super::{blip_covariance, origin_3d, transition_covariance, wide_covariance};

    const max_aircraft : u64;
    const max_real_blips_per_aircraft : u64;
    const max_false_blips_per_timestep : u64;
    const time_steps : u64;

    let n : u64 ~ Poisson::new(2.0) max max_aircraft;
    let aircraft_location[aircraft of n, time of time_steps] : DVector<f64> ~ if time == 0 {
        MultivariateNormal::new(origin_3d(), wide_covariance())
    } else {
        MultivariateNormal::new(
            aircraft_location[aircraft, time - 1].clone(),
            transition_covariance(),
        )
    };
    let num_real_blip[aircraft of n, time of time_steps] : u64 ~
        Poisson::new(1.2) max max_real_blips_per_aircraft;
    let real_blip_location[aircraft of n, blip of max_real_blips_per_aircraft, time of time_steps] : DVector<f64> ~
        MultivariateNormal::new(
            aircraft_location[aircraft, time].clone(),
            blip_covariance(),
        );

    let num_false_blip[time of time_steps] : u64 ~
        Poisson::new(1.0) max max_false_blips_per_timestep;
    let false_blip_location[false_blip of max_false_blips_per_timestep, time of time_steps] : DVector<f64> ~
        MultivariateNormal::new(origin_3d(), wide_covariance());

    let all_blip_locations[time of time_steps] : HashSet<BlipLocation> = {
        let mut locations = HashSet::new();
        // collect real blips and false blips for this timestep
        locations
    };

    query n;
    query all_blip_locations;
}
```

## Available distributions

| Distribution | Domain | Parameters |
|---|---|---|
| `Bernoulli` | `bool` | `p ∈ [0, 1]` |
| `Dirac<T>` | `T` | deterministic `value` |
| `Empirical<T>` | `T` | finite weighted support |
| `Categorical` | `usize` | non-negative category weights |
| `Binomial` | `u64` | `n ≥ 1`, `p ∈ (0, 1)` |
| `BetaBinomial` | `u64` | `n ≥ 1`, `alpha > 0`, `beta > 0` |
| `DiscreteUniform` | `i64` | `a ≤ b` |
| `Geometric` | `u64` | `p ∈ (0, 1]` |
| `Hypergeometric` | `u64` | population, successes, draws |
| `NegativeBinomial` | `u64` | `r > 0`, `p ∈ (0, 1]` |
| `Poisson` | `u64` | `rate > 0` |
| `Uniform` | `f64` | `low < high` |
| `Exponential` | `f64` | `rate > 0` |
| `Normal` | `f64` | `mean`, `std_dev > 0` |
| `LogNormal` | `f64` | `mu`, `sigma > 0` |
| `Beta` | `f64` | `alpha > 0`, `beta > 0` |
| `Cauchy` | `f64` | `median`, `scale > 0` |
| `Chi` | `f64` | `k > 0` |
| `ChiSquared` | `f64` | `k > 0` |
| `Erlang` | `f64` | integer `k ≥ 1`, `lambda > 0` |
| `FisherF` | `f64` | `d1 > 0`, `d2 > 0` |
| `Frechet` | `f64` | `location`, `scale > 0`, `shape > 0` |
| `Gamma` | `f64` | `shape > 0`, `scale > 0` |
| `Gumbel` | `f64` | `mu`, `beta > 0` |
| `HalfNormal` | `f64` | `sigma > 0` |
| `InverseGamma` | `f64` | `alpha > 0`, `beta > 0` |
| `InverseGaussian` | `f64` | `mu > 0`, `lambda > 0` |
| `Laplace` | `f64` | `mu`, `b > 0` |
| `Logistic` | `f64` | `mu`, `s > 0` |
| `Pareto` | `f64` | `x_m > 0`, `alpha > 0` |
| `Rayleigh` | `f64` | `sigma > 0` |
| `StudentT` | `f64` | `df > 0` |
| `Triangular` | `f64` | `a ≤ c ≤ b`, `a < b` |
| `Weibull` | `f64` | `lambda > 0`, `k > 0` |
| `Dirichlet` | `Vec<f64>` | concentration vector, each `alpha_i > 0` |
| `Multinomial` | `Vec<u64>` | `n ≥ 1`, probability vector |
| `MultivariateNormal` | `nalgebra::DVector<f64>` | mean vector, SPD covariance matrix |
| `MultivariateStudentT` | `nalgebra::DVector<f64>` | mean vector, SPD scale matrix, `df > 0` |
| `MatrixNormal` | `nalgebra::DMatrix<f64>` | mean matrix, SPD row and column covariance matrices |
| `Wishart` | `nalgebra::DMatrix<f64>` | `df > p - 1`, SPD scale matrix |

## Documentation

Ferric's API documentation is published automatically by [docs.rs](https://docs.rs/ferric)
when a new crate version is published to crates.io. There is no separate official Rust
documentation host to push to manually; the canonical Rust crate docs live on docs.rs.

For project docs beyond API reference, keep them in this repository unless they grow into
a distinct website. A separate `Ferric-AI/docs` repo would add coordination overhead right
now, while docs.rs plus this README gives users the normal Rust discovery path.

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
