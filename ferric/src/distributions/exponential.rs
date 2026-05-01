// Copyright 2022 The Ferric AI Project Developers

use rand_distr::Distribution as Distribution2;
use rand_distr::Exp as Exp2;

use crate::distributions::Distribution;
use rand::Rng;

/// Exponential distribution over non-negative reals.
///
/// The PDF is
///
/// $$p(x \mid \lambda) = \lambda e^{-\lambda x}, \quad x \geq 0$$
///
/// where $\lambda > 0$ is the rate parameter (mean $= 1/\lambda$).
///
/// See [Exponential distribution](https://en.wikipedia.org/wiki/Exponential_distribution)
/// on Wikipedia for further details.
///
/// # Examples
///
/// ```
/// use ferric::distributions::{Distribution, Exponential};
/// use rand::thread_rng;
///
/// let dist = Exponential::new(2.0).unwrap();
/// let x: f64 = dist.sample(&mut thread_rng());
/// println!("sample = {:.4}", x);
/// ```
pub struct Exponential {
    rate: f64,
}

impl Exponential {
    /// Construct an Exponential distribution with rate parameter `rate` ($\lambda$).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `rate` is not strictly positive.
    pub fn new(rate: f64) -> Result<Exponential, String> {
        if rate <= 0.0 {
            Err(format!(
                "Exponential: illegal rate `{}` should be greater than 0",
                rate
            ))
        } else {
            Ok(Exponential { rate })
        }
    }
}

impl<R: Rng + ?Sized> Distribution<R> for Exponential {
    type Domain = f64;

    fn sample(&self, rng: &mut R) -> f64 {
        Exp2::new(self.rate).unwrap().sample(rng)
    }

    /// Returns $\ln\lambda - \lambda x$ for $x \geq 0$, or $-\infty$ otherwise.
    fn log_prob(&self, x: &f64) -> f64 {
        if *x < 0.0 {
            f64::NEG_INFINITY
        } else {
            self.rate.ln() - self.rate * x
        }
    }

    fn is_discrete(&self) -> bool {
        false
    }
}

impl std::fmt::Display for Exponential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Exponential {{ rate = {} }}", self.rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn exponential_sample() {
        let mut rng = thread_rng();
        let rate = 2.0f64;
        let dist = Exponential::new(rate).unwrap();
        println!("dist = {}", dist);
        let mut total = 0f64;
        let trials = 10000;
        for _ in 0..trials {
            total += dist.sample(&mut rng);
        }
        let empirical_mean = total / (trials as f64);
        let expected_mean = 1.0 / rate;
        let expected_std = 1.0 / rate;
        let err = 5.0 * expected_std / (trials as f64).sqrt();
        assert!((empirical_mean - expected_mean).abs() < err);
    }

    #[test]
    fn exponential_log_prob() {
        // Exp(1): log_prob(1) = ln(1) - 1*1 = -1
        let dist = Exponential::new(1.0).unwrap();
        let lp = <Exponential as Distribution<ThreadRng>>::log_prob(&dist, &1.0);
        assert!((lp - (-1.0f64)).abs() < 1e-10);
        // outside support
        let lp_out = <Exponential as Distribution<ThreadRng>>::log_prob(&dist, &-0.1);
        assert_eq!(lp_out, f64::NEG_INFINITY);
        assert!(!<Exponential as Distribution<ThreadRng>>::is_discrete(
            &dist
        ));
    }

    #[test]
    #[should_panic]
    fn exponential_zero_rate() {
        Exponential::new(0.0).unwrap();
    }

    #[test]
    #[should_panic]
    fn exponential_negative_rate() {
        Exponential::new(-1.0).unwrap();
    }
}
