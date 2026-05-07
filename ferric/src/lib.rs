// Copyright 2022 The Ferric AI Project Developers

//! Ferric is a small probabilistic programming language embedded in Rust.
//!
//! You write a model with [`make_model!`], using ordinary Rust expressions for
//! deterministic dependencies and Ferric distributions for stochastic random
//! variables. The macro expands to a Rust module containing a `Model` type,
//! query sample types, and samplers.
//!
//! # Minimal Example
//!
//! ```
//! use ferric::make_model;
//!
//! make_model! {
//!     name coin;
//!     use ferric::distributions::Bernoulli;
//!
//!     const draws : u64;
//!
//!     let fair : bool ~ Bernoulli::new(0.5);
//!     let draw[trial of draws] : bool ~ if fair {
//!         Bernoulli::new(0.5)
//!     } else {
//!         Bernoulli::new(0.8)
//!     };
//!     let heads : u64 = draw.iter().filter(|&&is_head| is_head).count() as u64;
//!
//!     observe heads;
//!     query fair;
//! }
//!
//! let model = coin::Model {
//!     draws: 6,
//!     heads: 5,
//! };
//! let num_samples = 100;
//! let mut fair_count = 0;
//! for sample in model.sample_iter().take(num_samples) {
//!     if sample.fair {
//!         fair_count += 1;
//!     }
//! }
//! let prob_fair = fair_count as f64 / num_samples as f64;
//! assert!((0.0..=1.0).contains(&prob_fair));
//! ```
//!
//! # Language Overview
//!
//! A model starts with `name model_name;`, optional `use` statements, optional
//! constants, then variable declarations, observations, and queries.
//!
//! ```text
//! make_model! {
//!     name my_model;
//!     use ferric::distributions::Normal;
//!
//!     const known_value : f64;
//!
//!     let latent : f64 ~ Normal::new(0.0, 1.0);
//!     let measured : f64 ~ Normal::new(latent, known_value);
//!
//!     observe measured;
//!     query latent;
//! }
//! ```
//!
//! Constants become public fields on the generated `Model`. Observed variables
//! also become public fields and must be supplied when constructing the model:
//!
//! ```text
//! let model = my_model::Model {
//!     known_value: 0.25,
//!     measured: 1.2,
//! };
//! ```
//!
//! # Stochastic And Deterministic Variables
//!
//! Use `~` for a stochastic variable drawn from a distribution:
//!
//! ```text
//! let x : f64 ~ Normal::new(0.0, 1.0);
//! ```
//!
//! Use `=` for a deterministic variable:
//!
//! ```text
//! let shifted : f64 = x + 3.0;
//! ```
//!
//! Dependencies are Rust expressions. Earlier variables and constants may be
//! referenced by name; Ferric rewrites those references into generated model
//! evaluation calls. Distribution constructors usually return `Result`, so
//! Ferric-generated code unwraps them after your model expression is evaluated.
//!
//! # Observations And Queries
//!
//! `observe variable;` conditions on a value supplied in the generated
//! `Model`. `query variable;` includes a variable in each returned sample.
//!
//! Rejection sampling is available through `sample_iter()` and is only
//! appropriate when all observations are discrete. Self-normalised importance
//! sampling is available through `weighted_sample_iter()` when every observed
//! variable is stochastic:
//!
//! ```
//! use ferric::{make_model, weighted_mean};
//!
//! make_model! {
//!     name noisy_coin;
//!     use ferric::distributions::Bernoulli;
//!
//!     let fair : bool ~ Bernoulli::new(0.5);
//!     let reported : bool ~ if fair {
//!         Bernoulli::new(0.9)
//!     } else {
//!         Bernoulli::new(0.1)
//!     };
//!
//!     observe reported;
//!     query fair;
//! }
//!
//! let model = noisy_coin::Model { reported: true };
//! let mut values = Vec::new();
//! let mut weights = Vec::new();
//! for sample in model.weighted_sample_iter().take(100) {
//!     values.push(sample.sample.fair as u8 as f64);
//!     weights.push(sample.log_weight);
//! }
//! let posterior_mean = weighted_mean(&values, &weights);
//! let ess = ferric::effective_sample_size(&weights);
//! assert!((0.0..=1.0).contains(&posterior_mean));
//! assert!(ess > 0.0);
//! ```
//!
//! User-proposal importance sampling is available through
//! `importance_sampler(proposer)`. Each generated model module includes an
//! `ObservedData` struct, a `Proposal` struct, and a `Proposer<R>` trait. Ferric
//! calls `Proposer::initialize(&ObservedData)` once before sampling so the
//! proposer can build proposal distributions from constants and observations.
//! Each call to `Proposer::propose` returns proposed latent stochastic values
//! and their joint proposal `log_prob`; omitted proposal fields are sampled
//! from the model prior. Ferric then computes
//! `log p_model(proposed values) - log q(proposed values)` and adds the usual
//! observation log likelihoods. For diagnostics, generated models also provide
//! `importance_sampler_debug(proposer, n)`, which prints the proposal, prior
//! terms for proposed values, observed likelihood terms, sampled stochastic
//! values, and final log weight for the first `n` worlds. Use
//! [`effective_sample_size`] on the collected log weights to monitor weight
//! degeneracy. The rats example wires debugging to `FERRIC_DEBUG_IMPORTANCE`;
//! for example,
//! `FERRIC_DEBUG_IMPORTANCE=1 cargo run -p ferric --example rats` traces the
//! first importance sample in each rats experiment.
//!
//! # Indexed Random Variables
//!
//! Ferric supports one or more dimensions of indexed random variables. Each
//! dimension is written as `name of upper`, where `name` is the local index
//! variable and `upper` is a previously declared constant or variable. The
//! index takes values from `0` through `upper - 1`.
//!
//! ```text
//! const n : u64;
//! const t : u64;
//!
//! let survival : f64 ~ Beta::new(99.0, 1.0);
//! let alive[person of n, time of t] : bool ~ if time == 0 {
//!     Bernoulli::new(1.0)
//! } else if alive[person, time - 1] {
//!     Bernoulli::new(survival)
//! } else {
//!     Bernoulli::new(0.0)
//! };
//! let age[person of n] : u64 = {
//!     let mut age = t;
//!     for time in 0..t {
//!         if !alive[person, time] {
//!             age = time;
//!             break;
//!         }
//!     }
//!     age
//! };
//! observe age;
//! query survival;
//! ```
//!
//! Indexed query values are nested `Vec`s. Indexed observations are nested
//! `Vec<Option<T>>`; `Some(value)` observes that entry and `None` masks it as
//! missing.
//!
//! # Random Lengths And `max`
//!
//! An indexed variable can be bounded by a stochastic integer-valued variable:
//!
//! ```text
//! const max_n : u64;
//!
//! let n : u64 ~ ferric::distributions::Poisson::new(4.0) max max_n;
//! let flips[flip of n] : bool ~ ferric::distributions::Bernoulli::new(0.5);
//! let heads : u64 = flips.iter().filter(|&&x| x).count() as u64;
//!
//! observe heads;
//! query n;
//! ```
//!
//! The `max` annotation is a bounded-domain declaration. For example
//! `n ~ Poisson::new(3.0) max 100` means the domain of `n` is `0..=100`;
//! values above 100 are outside the model. Ferric normalizes bounded
//! likelihoods by subtracting [`distributions::Distribution::log_cum_prob`],
//! the log CDF at the bound. Generated worlds cache that value so each bounded
//! variable value computes the normalization term once per sampled world state.
//!
//! # Distributions
//!
//! Built-in scalar, vector, and matrix distributions live in [`distributions`].
//! Common choices include [`Bernoulli`](distributions::Bernoulli),
//! [`Binomial`](distributions::Binomial),
//! [`Categorical`](distributions::Categorical),
//! [`Poisson`](distributions::Poisson),
//! [`DiscreteUniform`](distributions::DiscreteUniform),
//! [`Normal`](distributions::Normal),
//! [`Gamma`](distributions::Gamma),
//! [`Beta`](distributions::Beta),
//! [`Dirichlet`](distributions::Dirichlet),
//! [`Multinomial`](distributions::Multinomial),
//! [`MultivariateNormal`](distributions::MultivariateNormal),
//! [`MatrixNormal`](distributions::MatrixNormal), and
//! [`Wishart`](distributions::Wishart).
//!
//! Each distribution page documents its parameters, support, sampling behavior,
//! and log probability.
//!
//! # Worked Examples
//!
//! The repository examples show complete models:
//!
//! - Dirichlet-multinomial conjugacy:
//!   <https://github.com/Ferric-AI/ferric/blob/main/ferric/examples/dirichlet_distribution.rs>
//! - Multivariate normal sensor inference:
//!   <https://github.com/Ferric-AI/ferric/blob/main/ferric/examples/multivariate_normal.rs>
//! - Indexed unknown-cardinality urn model:
//!   <https://github.com/Ferric-AI/ferric/blob/main/ferric/examples/urn_unknown_marbles.rs>
//! - Radar forward simulation with named indices:
//!   <https://github.com/Ferric-AI/ferric/blob/main/ferric/examples/radar.rs>
//! - Gelfand rats hierarchical growth model:
//!   <https://github.com/Ferric-AI/ferric/blob/main/ferric/examples/rats.rs>
//!
//! See the [README](https://github.com/Ferric-AI/ferric#readme) for release
//! notes, publishing notes, and a shorter tour.

// re-export make_model from the ferric-macros crate
pub use ferric_macros::make_model;

// Public modules
pub mod core;
pub mod distributions;

// re-export FeOption and its variants
pub use self::core::FeOption;
pub use FeOption::{Known, Null, Unknown};

/// Mask-aware equality for generated observation checks.
///
/// `Option<T>` treats `None` as a missing observation and `Some(value)` as an
/// exact observation. Nested `Vec`s recurse, which lets indexed random
/// variables be observed with nested arrays of optional values.
pub trait MaskedEq<T> {
    fn masked_eq(&self, value: &T) -> bool;
}

impl<T: PartialEq> MaskedEq<T> for Option<T> {
    fn masked_eq(&self, value: &T) -> bool {
        match self {
            Some(observed) => observed == value,
            None => true,
        }
    }
}

impl<O, T> MaskedEq<Vec<T>> for Vec<O>
where
    O: MaskedEq<T>,
{
    fn masked_eq(&self, value: &Vec<T>) -> bool {
        self.len() <= value.len()
            && self
                .iter()
                .zip(value.iter())
                .all(|(observed, sampled)| observed.masked_eq(sampled))
    }
}

/// Compute the self-normalised importance-weighted mean of `values`.
///
/// Given a collection of values $x_i$ and their corresponding log importance
/// weights $\tilde{w}_i$ (unnormalised, in log space), this computes the
/// self-normalised importance-sampling (SNIS) estimate of $\mathbb{E}(X)$:
///
/// $$\hat{\mu} = \frac{\sum_i w_i x_i}{\sum_i w_i},
///   \qquad w_i = e^{\tilde{w}_i - \max_j \tilde{w}_j}$$
///
/// The max-subtraction keeps the arithmetic numerically stable without
/// changing the result (it cancels in numerator and denominator).
///
/// # Panics
///
/// Panics if `values` and `log_weights` have different lengths.
///
/// # Examples
///
/// ```
/// use ferric::weighted_mean;
///
/// // Uniform weights — equivalent to a plain mean.
/// let values = vec![1.0_f64, 2.0, 3.0];
/// let log_weights = vec![0.0_f64; 3];
/// let mean = weighted_mean(&values, &log_weights);
/// assert!((mean - 2.0).abs() < 1e-10);
/// ```
pub fn weighted_mean(values: &[f64], log_weights: &[f64]) -> f64 {
    assert_eq!(
        values.len(),
        log_weights.len(),
        "values and log_weights must have the same length"
    );
    let max_lw = log_weights
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = log_weights.iter().map(|&lw| (lw - max_lw).exp()).collect();
    let total: f64 = weights.iter().sum();
    values
        .iter()
        .zip(weights.iter())
        .map(|(&v, &w)| v * w)
        .sum::<f64>()
        / total
}

/// Compute the self-normalised importance-weighted standard deviation of
/// `values`.
///
/// Uses the same SNIS weights as [`weighted_mean`] to estimate
///
/// $$\hat{\sigma} = \sqrt{\frac{\sum_i w_i (x_i - \hat{\mu})^2}{\sum_i w_i}}$$
///
/// This is the weighted population standard deviation (not the unbiased
/// sample estimate), which is appropriate for summarising an importance-
/// sampling posterior.
///
/// # Panics
///
/// Panics if `values` and `log_weights` have different lengths.
///
/// # Examples
///
/// ```
/// use ferric::weighted_std;
///
/// // Population std of [1, 2, 3] with uniform weights is sqrt(2/3).
/// let values = vec![1.0_f64, 2.0, 3.0];
/// let log_weights = vec![0.0_f64; 3];
/// let std = weighted_std(&values, &log_weights);
/// assert!((std - (2.0_f64 / 3.0).sqrt()).abs() < 1e-10);
/// ```
pub fn weighted_std(values: &[f64], log_weights: &[f64]) -> f64 {
    assert_eq!(
        values.len(),
        log_weights.len(),
        "values and log_weights must have the same length"
    );
    let mean = weighted_mean(values, log_weights);
    let max_lw = log_weights
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = log_weights.iter().map(|&lw| (lw - max_lw).exp()).collect();
    let total: f64 = weights.iter().sum();
    let variance = values
        .iter()
        .zip(weights.iter())
        .map(|(&v, &w)| w * (v - mean).powi(2))
        .sum::<f64>()
        / total;
    variance.sqrt()
}

/// Compute the effective sample size (ESS) of unnormalised log weights.
///
/// This returns
///
/// $$\mathrm{ESS} = \frac{(\sum_i w_i)^2}{\sum_i w_i^2},
///   \qquad w_i = e^{\tilde{w}_i - \max_j \tilde{w}_j}$$
///
/// The max-subtraction keeps the arithmetic numerically stable and does not
/// change the result.  Uniform weights therefore have ESS equal to the number
/// of samples, while a single dominant weight gives ESS close to 1.
///
/// Empty inputs, or inputs where every log weight is `f64::NEG_INFINITY`,
/// return 0.0.
///
/// # Examples
///
/// ```
/// use ferric::effective_sample_size;
///
/// let log_weights = vec![0.0_f64; 4];
/// assert!((effective_sample_size(&log_weights) - 4.0).abs() < 1e-10);
/// ```
pub fn effective_sample_size(log_weights: &[f64]) -> f64 {
    if log_weights.is_empty() {
        return 0.0;
    }

    if log_weights.iter().any(|log_weight| log_weight.is_nan()) {
        return f64::NAN;
    }

    let max_lw = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    if max_lw == f64::NEG_INFINITY {
        return 0.0;
    }

    if max_lw == f64::INFINITY {
        let infinite_weights = log_weights
            .iter()
            .filter(|&&log_weight| log_weight == f64::INFINITY)
            .count();
        return infinite_weights as f64;
    }

    let mut sum_weight = 0.0;
    let mut sum_weight_squared = 0.0;
    for &log_weight in log_weights {
        let weight = (log_weight - max_lw).exp();
        sum_weight += weight;
        sum_weight_squared += weight * weight;
    }

    if sum_weight_squared == 0.0 {
        0.0
    } else {
        sum_weight * sum_weight / sum_weight_squared
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masked_eq_option_and_nested_vec() {
        assert!(Some(3).masked_eq(&3));
        assert!(!Some(3).masked_eq(&4));
        let missing: Option<i32> = None;
        assert!(missing.masked_eq(&4));

        let observed = vec![Some(true), None, Some(false)];
        assert!(observed.masked_eq(&vec![true, true, false]));
        assert!(!observed.masked_eq(&vec![true, true, true]));
        assert!(!observed.masked_eq(&vec![true]));

        let nested = vec![vec![Some(1), None], vec![Some(3), Some(4)]];
        assert!(nested.masked_eq(&vec![vec![1, 2], vec![3, 4]]));
        assert!(!nested.masked_eq(&vec![vec![1, 2], vec![3, 5]]));
    }

    #[test]
    fn weighted_mean_uniform() {
        let values = vec![1.0, 2.0, 3.0];
        let log_weights = vec![0.0, 0.0, 0.0];
        let mean = weighted_mean(&values, &log_weights);
        assert!((mean - 2.0).abs() < 1e-10);
    }

    #[test]
    fn weighted_std_uniform() {
        let values = vec![1.0, 2.0, 3.0];
        let log_weights = vec![0.0, 0.0, 0.0];
        let std = weighted_std(&values, &log_weights);
        // population std of [1,2,3] = sqrt(2/3) ≈ 0.8165
        assert!((std - (2.0f64 / 3.0).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn weighted_mean_concentrated() {
        // all weight on the last element
        let values = vec![1.0, 2.0, 10.0];
        let log_weights = vec![-100.0, -100.0, 0.0];
        let mean = weighted_mean(&values, &log_weights);
        assert!((mean - 10.0).abs() < 0.01);
    }

    #[test]
    fn effective_sample_size_uniform() {
        let log_weights = vec![0.0, 0.0, 0.0, 0.0];
        assert!((effective_sample_size(&log_weights) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn effective_sample_size_concentrated() {
        let log_weights = vec![-100.0, -100.0, 0.0];
        assert!((effective_sample_size(&log_weights) - 1.0).abs() < 1e-8);
    }

    #[test]
    fn effective_sample_size_empty_or_zero_weight() {
        assert_eq!(effective_sample_size(&[]), 0.0);
        assert_eq!(
            effective_sample_size(&[f64::NEG_INFINITY, f64::NEG_INFINITY]),
            0.0
        );
    }

    #[test]
    fn effective_sample_size_nan_stays_nan() {
        assert!(effective_sample_size(&[0.0, f64::NAN]).is_nan());
    }
}
