// Copyright 2022 The Ferric AI Project Developers

//! A probabilistic programming language in Rust with a declarative syntax for
//! Bayesian models.
//!
//! # Key entry points
//!
//! - [`make_model!`] — declare a probabilistic model; expands into a module
//!   with `Model`, `Sample`, `WeightedSample`, and two iterator types.
//! - [`weighted_mean`] / [`weighted_std`] — posterior summaries from
//!   self-normalised importance-sampling (SNIS) weights.
//! - [`distributions`] — built-in probability distributions:
//!   [`Bernoulli`](distributions::Bernoulli),
//!   [`Binomial`](distributions::Binomial),
//!   [`Geometric`](distributions::Geometric),
//!   [`Poisson`](distributions::Poisson),
//!   [`Uniform`](distributions::Uniform),
//!   [`Exponential`](distributions::Exponential),
//!   [`Normal`](distributions::Normal),
//!   [`LogNormal`](distributions::LogNormal),
//!   [`Beta`](distributions::Beta),
//!   [`Gamma`](distributions::Gamma),
//!   [`StudentT`](distributions::StudentT),
//!   [`Cauchy`](distributions::Cauchy),
//!   [`MultivariateNormal`](distributions::MultivariateNormal).
//!
//! See the [README](https://github.com/ferric-ai/ferric#readme) for a
//! quick-start guide and worked examples.

// re-export make_model from the ferric-macros crate
pub use ferric_macros::make_model;

// Public modules
pub mod core;
pub mod distributions;

// re-export FeOption and its variants
pub use self::core::FeOption;
pub use FeOption::{Known, Null, Unknown};

/// Compute the self-normalised importance-weighted mean of `values`.
///
/// Given a collection of values $x_i$ and their corresponding log importance
/// weights $\tilde{w}_i$ (unnormalised, in log space), this computes the
/// self-normalised importance-sampling (SNIS) estimate of $\mathbb{E}[X]$:
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
