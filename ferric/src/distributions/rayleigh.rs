// Copyright 2022 The Ferric AI Project Developers

use rand::Rng;
use rand_distr::Distribution as Distribution2;
use rand_distr::Uniform;

use crate::distributions::Distribution;

/// Rayleigh distribution over the non-negative reals.
///
/// The PDF is
///
/// $$p(x \mid \sigma) = \frac{x}{\sigma^2}
///     \exp\!\left(-\frac{x^2}{2\sigma^2}\right)$$
///
/// where $\sigma > 0$ is the scale parameter.  The Rayleigh distribution
/// arises as the magnitude of a 2-D zero-mean Gaussian vector with equal
/// variance $\sigma^2$ in each component.
///
/// See [Rayleigh distribution](https://en.wikipedia.org/wiki/Rayleigh_distribution)
/// on Wikipedia for further details.
///
/// # Examples
///
/// ```
/// use ferric::distributions::{Distribution, Rayleigh};
/// use rand::thread_rng;
///
/// let dist = Rayleigh::new(1.0).unwrap();
/// let x: f64 = dist.sample(&mut thread_rng());
/// println!("sample = {:.4}", x);
/// ```
pub struct Rayleigh {
    sigma: f64,
}

impl Rayleigh {
    /// Construct a Rayleigh distribution with scale `sigma` ($\sigma$).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `sigma` is not strictly positive.
    pub fn new(sigma: f64) -> Result<Rayleigh, String> {
        if sigma <= 0.0 {
            Err(format!(
                "Rayleigh: illegal scale `{}` should be greater than 0",
                sigma
            ))
        } else {
            Ok(Rayleigh { sigma })
        }
    }
}

impl<R: Rng + ?Sized> Distribution<R> for Rayleigh {
    type Domain = f64;

    /// Draw a sample via the quantile function:
    /// $X = \sigma\sqrt{-2\ln(1-u)}$ where $u \sim U(0,1)$.
    fn sample(&self, rng: &mut R) -> f64 {
        let u: f64 = Uniform::new(0.0f64, 1.0).sample(rng);
        let u = u.clamp(1e-15, 1.0 - 1e-15);
        self.sigma * (-2.0 * (1.0 - u).ln()).sqrt()
    }

    /// Returns $\ln x - 2\ln\sigma - x^2/(2\sigma^2)$, or $-\infty$ for
    /// $x \le 0$.
    fn log_prob(&self, x: &f64) -> f64 {
        if *x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        x.ln() - 2.0 * self.sigma.ln() - x * x / (2.0 * self.sigma * self.sigma)
    }

    fn is_discrete(&self) -> bool {
        false
    }
}

impl std::fmt::Display for Rayleigh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Rayleigh {{ sigma = {} }}", self.sigma)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn rayleigh_sample() {
        let mut rng = thread_rng();
        let sigma = 2.0f64;
        let dist = Rayleigh::new(sigma).unwrap();
        println!("dist = {}", dist);
        let trials = 100_000;
        let mut total = 0.0f64;
        for _ in 0..trials {
            total += dist.sample(&mut rng);
        }
        let empirical_mean = total / trials as f64;
        // Mean = sigma * sqrt(pi/2)
        let expected_mean = sigma * (std::f64::consts::PI / 2.0).sqrt();
        // Variance = sigma^2 * (4 - pi) / 2
        let variance = sigma * sigma * (4.0 - std::f64::consts::PI) / 2.0;
        let std = variance.sqrt();
        let err = 5.0 * std / (trials as f64).sqrt();
        assert!((empirical_mean - expected_mean).abs() < err);
    }

    #[test]
    fn rayleigh_log_prob() {
        // Rayleigh(1) at x=1: ln(1) - 2*ln(1) - 1/2 = -0.5
        let dist = Rayleigh::new(1.0).unwrap();
        let lp = <Rayleigh as Distribution<ThreadRng>>::log_prob(&dist, &1.0);
        assert!((lp - (-0.5f64)).abs() < 1e-10);

        // x <= 0 → NEG_INFINITY
        let lp_zero = <Rayleigh as Distribution<ThreadRng>>::log_prob(&dist, &0.0);
        assert_eq!(lp_zero, f64::NEG_INFINITY);

        assert!(!<Rayleigh as Distribution<ThreadRng>>::is_discrete(&dist));
    }

    #[test]
    fn rayleigh_display() {
        let dist = Rayleigh::new(2.0).unwrap();
        let s = format!("{}", dist);
        assert!(s.contains("Rayleigh"), "missing type name: {}", s);
    }

    #[test]
    #[should_panic]
    fn rayleigh_zero_scale() {
        Rayleigh::new(0.0).unwrap();
    }

    #[test]
    #[should_panic]
    fn rayleigh_negative_scale() {
        Rayleigh::new(-1.0).unwrap();
    }
}
