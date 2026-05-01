// Copyright 2022 The Ferric AI Project Developers

use crate::distributions::Distribution;
use rand::Rng;

/// Continuous uniform distribution over the closed interval $[a, b]$.
///
/// The PDF is
///
/// $$p(x \mid a, b) = \frac{1}{b - a}, \quad x \in [a, b]$$
///
/// and zero outside the interval.
///
/// See [Continuous uniform distribution](https://en.wikipedia.org/wiki/Continuous_uniform_distribution)
/// on Wikipedia for further details.
///
/// # Examples
///
/// ```
/// use ferric::distributions::{Distribution, Uniform};
/// use rand::thread_rng;
///
/// let dist = Uniform::new(0.0, 1.0).unwrap();
/// let x: f64 = dist.sample(&mut thread_rng());
/// println!("sample = {:.4}", x);
/// ```
pub struct Uniform {
    low: f64,
    high: f64,
}

impl Uniform {
    /// Construct a Uniform distribution on the interval `[low, high]`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `high <= low`.
    pub fn new(low: f64, high: f64) -> Result<Uniform, String> {
        if high <= low {
            Err(format!(
                "Uniform: illegal interval [{}, {}]; high must be strictly greater than low",
                low, high
            ))
        } else {
            Ok(Uniform { low, high })
        }
    }
}

impl<R: Rng + ?Sized> Distribution<R> for Uniform {
    type Domain = f64;

    fn sample(&self, rng: &mut R) -> f64 {
        rng.gen_range(self.low..self.high)
    }

    /// Returns $-\ln(b - a)$ for $x \in [a, b]$, or $-\infty$ otherwise.
    fn log_prob(&self, x: &f64) -> f64 {
        if *x >= self.low && *x <= self.high {
            -(self.high - self.low).ln()
        } else {
            f64::NEG_INFINITY
        }
    }

    fn is_discrete(&self) -> bool {
        false
    }
}

impl std::fmt::Display for Uniform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Uniform {{ low = {}, high = {} }}", self.low, self.high)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn uniform_sample() {
        let mut rng = thread_rng();
        let low = 2.0f64;
        let high = 5.0f64;
        let dist = Uniform::new(low, high).unwrap();
        println!("dist = {}", dist);
        let mut total = 0f64;
        let trials = 10000;
        for _ in 0..trials {
            total += dist.sample(&mut rng);
        }
        let empirical_mean = total / (trials as f64);
        let expected_mean = (low + high) / 2.0;
        let expected_std = (high - low) / (12.0f64).sqrt();
        let err = 5.0 * expected_std / (trials as f64).sqrt();
        assert!((empirical_mean - expected_mean).abs() < err);
    }

    #[test]
    fn uniform_log_prob() {
        let dist = Uniform::new(0.0, 2.0).unwrap();
        // log p(1 | Uniform(0,2)) = -ln(2)
        let lp = <Uniform as Distribution<ThreadRng>>::log_prob(&dist, &1.0);
        assert!((lp - (-(2.0f64).ln())).abs() < 1e-10);
        // outside support -> -inf
        let lp_out = <Uniform as Distribution<ThreadRng>>::log_prob(&dist, &3.0);
        assert_eq!(lp_out, f64::NEG_INFINITY);
        assert!(!<Uniform as Distribution<ThreadRng>>::is_discrete(&dist));
    }

    #[test]
    #[should_panic]
    fn uniform_invalid_interval() {
        Uniform::new(5.0, 2.0).unwrap();
    }

    #[test]
    #[should_panic]
    fn uniform_equal_bounds() {
        Uniform::new(1.0, 1.0).unwrap();
    }
}
