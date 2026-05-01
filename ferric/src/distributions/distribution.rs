// Copyright 2022 The Ferric AI Project Developers

use rand::Rng;
use rand::rngs::ThreadRng;

/// A probability distribution over a specific domain that can generate
/// random samples and evaluate (log) probabilities.
///
/// For example, [`Bernoulli`](crate::distributions::Bernoulli) implements
/// `Distribution<Domain = bool>` and generates random booleans.
pub trait Distribution<R = ThreadRng>
where
    R: Rng + ?Sized,
{
    /// The type of values produced by this distribution.
    type Domain;

    /// Draw one random sample from this distribution.
    fn sample(&self, rng: &mut R) -> Self::Domain;

    /// Compute the log probability (discrete) or log probability density
    /// (continuous) of observing `x` under this distribution.
    ///
    /// For a discrete distribution this returns $\log P(X = x)$.
    /// For a continuous distribution this returns the log-density
    /// $\log p(x)$.
    ///
    /// This value is used by the self-normalised importance sampler to
    /// weight prior samples by their likelihood of producing the observed
    /// data.
    fn log_prob(&self, x: &Self::Domain) -> f64;

    /// Returns `true` if this distribution is discrete (e.g. [`Bernoulli`],
    /// [`Poisson`]), or `false` if it is continuous (e.g. [`Normal`]).
    ///
    /// Rejection sampling is only valid for discrete observations; use
    /// [`Model::weighted_sample_iter`](crate) for models that contain
    /// continuous observed variables.
    ///
    /// [`Bernoulli`]: crate::distributions::Bernoulli
    /// [`Poisson`]: crate::distributions::Poisson
    /// [`Normal`]: crate::distributions::Normal
    fn is_discrete(&self) -> bool;
}
