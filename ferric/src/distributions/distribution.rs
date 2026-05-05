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

    /// Compute `log P(X <= x)` for discrete distributions, or the log CDF for
    /// continuous distributions.
    ///
    /// Ferric uses this when a random variable has a `max ...` bound. The
    /// bounded random variable is renormalized over values at or below the
    /// maximum, so generated likelihoods subtract this value from
    /// [`Distribution::log_prob`].
    ///
    /// The default assumes no normalization correction is needed. Distributions
    /// with unbounded support should override this before being used with
    /// `max ...`.
    fn log_cum_prob(&self, _x: &Self::Domain) -> f64 {
        0.0
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    struct PointMass;

    impl<R: Rng + ?Sized> Distribution<R> for PointMass {
        type Domain = u64;

        fn sample(&self, _rng: &mut R) -> Self::Domain {
            7
        }

        fn log_prob(&self, x: &Self::Domain) -> f64 {
            if *x == 7 { 0.0 } else { f64::NEG_INFINITY }
        }

        fn is_discrete(&self) -> bool {
            true
        }
    }

    #[test]
    fn default_log_cum_prob_has_no_normalization_correction() {
        let dist = PointMass;
        let normalizer =
            <PointMass as Distribution<rand::rngs::ThreadRng>>::log_cum_prob(&dist, &7);
        assert_eq!(normalizer, 0.0);
        assert_eq!(<PointMass as Distribution>::log_prob(&dist, &7), 0.0);
        assert_eq!(
            <PointMass as Distribution>::log_prob(&dist, &8),
            f64::NEG_INFINITY
        );
        let mut rng = rand::thread_rng();
        assert_eq!(<PointMass as Distribution>::sample(&dist, &mut rng), 7);
        assert!(<PointMass as Distribution>::is_discrete(&dist));
    }
}
