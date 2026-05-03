// Copyright 2022 The Ferric AI Project Developers

use rand::Rng;
use rand_distr::Distribution as Distribution2;
use rand_distr::Normal;

use crate::distributions::Distribution;

/// Half-normal distribution over the non-negative reals.
///
/// The PDF is
///
/// $$p(x \mid \sigma) =
///     \frac{\sqrt{2}}{\sigma\sqrt{\pi}}
///     \exp\!\left(-\frac{x^2}{2\sigma^2}\right), \quad x \ge 0$$
///
/// where $\sigma > 0$ is the scale parameter.  The half-normal is the
/// absolute value of a $\mathrm{Normal}(0, \sigma^2)$ random variable.
///
/// See [Half-normal distribution](https://en.wikipedia.org/wiki/Half-normal_distribution)
/// on Wikipedia for further details.
///
/// # Examples
///
/// ```
/// use ferric::distributions::{Distribution, HalfNormal};
/// use rand::thread_rng;
///
/// let dist = HalfNormal::new(1.0).unwrap();
/// let x: f64 = dist.sample(&mut thread_rng());
/// println!("sample = {:.4}", x);
/// ```
pub struct HalfNormal {
    sigma: f64,
}

impl HalfNormal {
    /// Construct a half-normal distribution with scale `sigma` ($\sigma$).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `sigma` is not strictly positive.
    pub fn new(sigma: f64) -> Result<HalfNormal, String> {
        if sigma <= 0.0 {
            Err(format!(
                "HalfNormal: illegal scale `{}` should be greater than 0",
                sigma
            ))
        } else {
            Ok(HalfNormal { sigma })
        }
    }
}

impl<R: Rng + ?Sized> Distribution<R> for HalfNormal {
    type Domain = f64;

    /// Draw a sample as $|Z|$ where $Z \sim \mathrm{Normal}(0, \sigma^2)$.
    fn sample(&self, rng: &mut R) -> f64 {
        Normal::new(0.0, self.sigma).unwrap().sample(rng).abs()
    }

    /// Returns $\ln\sqrt{2/\pi} - \ln\sigma - x^2/(2\sigma^2)$, or
    /// $-\infty$ for $x < 0$.
    fn log_prob(&self, x: &f64) -> f64 {
        if *x < 0.0 {
            return f64::NEG_INFINITY;
        }
        (2.0 / std::f64::consts::PI).sqrt().ln()
            - self.sigma.ln()
            - x * x / (2.0 * self.sigma * self.sigma)
    }

    fn is_discrete(&self) -> bool {
        false
    }
}

impl std::fmt::Display for HalfNormal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HalfNormal {{ sigma = {} }}", self.sigma)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn half_normal_sample() {
        let mut rng = thread_rng();
        let sigma = 2.0f64;
        let dist = HalfNormal::new(sigma).unwrap();
        println!("dist = {}", dist);
        let trials = 100_000;
        let mut total = 0.0f64;
        for _ in 0..trials {
            total += dist.sample(&mut rng);
        }
        let empirical_mean = total / trials as f64;
        // Mean = sigma * sqrt(2/pi)
        let expected_mean = sigma * (2.0 / std::f64::consts::PI).sqrt();
        // Variance = sigma^2 * (1 - 2/pi)
        let variance = sigma * sigma * (1.0 - 2.0 / std::f64::consts::PI);
        let std = variance.sqrt();
        let err = 5.0 * std / (trials as f64).sqrt();
        assert!((empirical_mean - expected_mean).abs() < err);
    }

    #[test]
    fn half_normal_log_prob() {
        // HalfNormal(1) at x=0: ln(sqrt(2/pi))
        let dist = HalfNormal::new(1.0).unwrap();
        let lp = <HalfNormal as Distribution<ThreadRng>>::log_prob(&dist, &0.0);
        let expected = (2.0f64 / std::f64::consts::PI).sqrt().ln();
        assert!((lp - expected).abs() < 1e-10);

        // x < 0 → NEG_INFINITY
        let lp_neg = <HalfNormal as Distribution<ThreadRng>>::log_prob(&dist, &-1.0);
        assert_eq!(lp_neg, f64::NEG_INFINITY);

        assert!(!<HalfNormal as Distribution<ThreadRng>>::is_discrete(&dist));
    }

    #[test]
    fn half_normal_display() {
        let dist = HalfNormal::new(1.0).unwrap();
        let s = format!("{}", dist);
        assert!(s.contains("HalfNormal"), "missing type name: {}", s);
    }

    #[test]
    #[should_panic]
    fn half_normal_zero_scale() {
        HalfNormal::new(0.0).unwrap();
    }
}
