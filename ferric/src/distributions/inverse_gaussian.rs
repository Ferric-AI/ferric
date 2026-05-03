// Copyright 2022 The Ferric AI Project Developers

use rand::Rng;
use rand_distr::Distribution as Distribution2;
use rand_distr::InverseGaussian as InverseGaussian2;

use crate::distributions::Distribution;

/// Inverse Gaussian (Wald) distribution over the positive reals.
///
/// The PDF is
///
/// $$p(x \mid \mu, \lambda) =
///     \sqrt{\frac{\lambda}{2\pi x^3}}
///     \exp\!\left(-\frac{\lambda(x-\mu)^2}{2\mu^2 x}\right)$$
///
/// where $\mu > 0$ is the mean and $\lambda > 0$ is the shape parameter.
///
/// See [Inverse Gaussian distribution](https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution)
/// on Wikipedia for further details.
///
/// # Examples
///
/// ```
/// use ferric::distributions::{Distribution, InverseGaussian};
/// use rand::thread_rng;
///
/// let dist = InverseGaussian::new(1.0, 2.0).unwrap();
/// let x: f64 = dist.sample(&mut thread_rng());
/// println!("sample = {:.4}", x);
/// ```
pub struct InverseGaussian {
    mu: f64,
    lambda: f64,
}

impl InverseGaussian {
    /// Construct an inverse Gaussian distribution with mean `mu` ($\mu$) and
    /// shape `lambda` ($\lambda$).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `mu` or `lambda` is not strictly positive.
    pub fn new(mu: f64, lambda: f64) -> Result<InverseGaussian, String> {
        if mu <= 0.0 {
            Err(format!(
                "InverseGaussian: illegal mean `{}` should be greater than 0",
                mu
            ))
        } else if lambda <= 0.0 {
            Err(format!(
                "InverseGaussian: illegal shape `{}` should be greater than 0",
                lambda
            ))
        } else {
            Ok(InverseGaussian { mu, lambda })
        }
    }
}

impl<R: Rng + ?Sized> Distribution<R> for InverseGaussian {
    type Domain = f64;

    fn sample(&self, rng: &mut R) -> f64 {
        InverseGaussian2::new(self.mu, self.lambda)
            .unwrap()
            .sample(rng)
    }

    /// Returns $\tfrac{1}{2}\ln(\lambda/(2\pi x^3))
    ///     - \lambda(x-\mu)^2/(2\mu^2 x)$,
    /// or $-\infty$ for $x \le 0$.
    fn log_prob(&self, x: &f64) -> f64 {
        if *x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        0.5 * (self.lambda / (2.0 * std::f64::consts::PI * x * x * x)).ln()
            - self.lambda * (x - self.mu).powi(2) / (2.0 * self.mu * self.mu * x)
    }

    fn is_discrete(&self) -> bool {
        false
    }
}

impl std::fmt::Display for InverseGaussian {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "InverseGaussian {{ mu = {}, lambda = {} }}",
            self.mu, self.lambda
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn inverse_gaussian_sample() {
        let mut rng = thread_rng();
        let mu = 2.0f64;
        let lambda = 3.0f64;
        let dist = InverseGaussian::new(mu, lambda).unwrap();
        println!("dist = {}", dist);
        let trials = 100_000;
        let mut total = 0.0f64;
        for _ in 0..trials {
            total += dist.sample(&mut rng);
        }
        let empirical_mean = total / trials as f64;
        // Mean = mu, Variance = mu^3 / lambda
        let variance = mu * mu * mu / lambda;
        let std = variance.sqrt();
        let err = 5.0 * std / (trials as f64).sqrt();
        assert!((empirical_mean - mu).abs() < err);
    }

    #[test]
    fn inverse_gaussian_log_prob() {
        // log_prob should be finite and negative at x=mu
        let dist = InverseGaussian::new(1.0, 1.0).unwrap();
        let lp = <InverseGaussian as Distribution<ThreadRng>>::log_prob(&dist, &1.0);
        assert!(lp.is_finite());

        // x <= 0 → NEG_INFINITY
        let lp_zero = <InverseGaussian as Distribution<ThreadRng>>::log_prob(&dist, &0.0);
        assert_eq!(lp_zero, f64::NEG_INFINITY);

        assert!(!<InverseGaussian as Distribution<ThreadRng>>::is_discrete(
            &dist
        ));
    }

    #[test]
    fn inverse_gaussian_display() {
        let dist = InverseGaussian::new(1.0, 2.0).unwrap();
        let s = format!("{}", dist);
        assert!(s.contains("InverseGaussian"), "missing type name: {}", s);
    }

    #[test]
    #[should_panic]
    fn inverse_gaussian_zero_mu() {
        InverseGaussian::new(0.0, 1.0).unwrap();
    }

    #[test]
    #[should_panic]
    fn inverse_gaussian_zero_lambda() {
        InverseGaussian::new(1.0, 0.0).unwrap();
    }
}
