// Copyright 2022 The Ferric AI Project Developers

use rand::Rng;
use rand_distr::Distribution as Distribution2;
use rand_distr::Weibull as Weibull2;

use crate::distributions::Distribution;

/// Weibull distribution over the positive reals.
///
/// The PDF is
///
/// $$p(x \mid \lambda, k) =
///     \frac{k}{\lambda}\!\left(\frac{x}{\lambda}\right)^{k-1}
///     \exp\!\left(-\!\left(\frac{x}{\lambda}\right)^k\right)$$
///
/// where $\lambda > 0$ is the scale parameter and $k > 0$ is the shape
/// parameter.
///
/// See [Weibull distribution](https://en.wikipedia.org/wiki/Weibull_distribution)
/// on Wikipedia for further details.
///
/// # Examples
///
/// ```
/// use ferric::distributions::{Distribution, Weibull};
/// use rand::thread_rng;
///
/// let dist = Weibull::new(1.0, 1.5).unwrap();
/// let x: f64 = dist.sample(&mut thread_rng());
/// println!("sample = {:.4}", x);
/// ```
pub struct Weibull {
    lambda: f64,
    k: f64,
}

impl Weibull {
    /// Construct a Weibull distribution with scale `lambda` ($\lambda$) and
    /// shape `k` ($k$).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `lambda` or `k` is not strictly positive.
    pub fn new(lambda: f64, k: f64) -> Result<Weibull, String> {
        if lambda <= 0.0 {
            Err(format!(
                "Weibull: illegal scale `{}` should be greater than 0",
                lambda
            ))
        } else if k <= 0.0 {
            Err(format!(
                "Weibull: illegal shape `{}` should be greater than 0",
                k
            ))
        } else {
            Ok(Weibull { lambda, k })
        }
    }
}

impl<R: Rng + ?Sized> Distribution<R> for Weibull {
    type Domain = f64;

    fn sample(&self, rng: &mut R) -> f64 {
        Weibull2::new(self.lambda, self.k).unwrap().sample(rng)
    }

    /// Returns $\ln k - \ln\lambda + (k-1)\ln x - (x/\lambda)^k$, or
    /// $-\infty$ for $x \le 0$.
    fn log_prob(&self, x: &f64) -> f64 {
        if *x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let z = x / self.lambda;
        self.k.ln() - self.k * self.lambda.ln() + (self.k - 1.0) * x.ln() - z.powf(self.k)
    }

    fn is_discrete(&self) -> bool {
        false
    }
}

impl std::fmt::Display for Weibull {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Weibull {{ lambda = {}, k = {} }}", self.lambda, self.k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn weibull_sample() {
        let mut rng = thread_rng();
        let lambda = 2.0f64;
        let k = 1.5f64;
        let dist = Weibull::new(lambda, k).unwrap();
        println!("dist = {}", dist);
        let trials = 100_000;
        let mut total = 0.0f64;
        for _ in 0..trials {
            total += dist.sample(&mut rng);
        }
        let empirical_mean = total / trials as f64;
        // Mean = lambda * Gamma(1 + 1/k)
        assert_eq!(gamma_fn(1.0), 1.0);
        let expected_mean = lambda * gamma_fn(1.0 + 1.0 / k);
        let variance =
            lambda * lambda * (gamma_fn(1.0 + 2.0 / k) - gamma_fn(1.0 + 1.0 / k).powi(2));
        let std = variance.sqrt();
        let err = 5.0 * std / (trials as f64).sqrt();
        assert!((empirical_mean - expected_mean).abs() < err);
    }

    fn gamma_fn(x: f64) -> f64 {
        // Stirling approximation good enough for test values
        if (x - 1.0).abs() < 1e-12 {
            return 1.0;
        }
        // Use lgamma from libm via std
        (libm::lgamma(x)).exp()
    }

    #[test]
    fn weibull_log_prob() {
        // Weibull(1, 1) is Exponential(1): log_prob(1) = -1
        let dist = Weibull::new(1.0, 1.0).unwrap();
        let lp = <Weibull as Distribution<ThreadRng>>::log_prob(&dist, &1.0);
        assert!((lp - (-1.0f64)).abs() < 1e-10);

        // Non-positive x → NEG_INFINITY
        let lp_zero = <Weibull as Distribution<ThreadRng>>::log_prob(&dist, &0.0);
        assert_eq!(lp_zero, f64::NEG_INFINITY);

        assert!(!<Weibull as Distribution<ThreadRng>>::is_discrete(&dist));
    }

    #[test]
    fn weibull_display() {
        let dist = Weibull::new(2.0, 1.5).unwrap();
        let s = format!("{}", dist);
        assert!(s.contains("Weibull"), "missing type name: {}", s);
    }

    #[test]
    #[should_panic]
    fn weibull_zero_scale() {
        Weibull::new(0.0, 1.0).unwrap();
    }

    #[test]
    #[should_panic]
    fn weibull_zero_shape() {
        Weibull::new(1.0, 0.0).unwrap();
    }

    #[test]
    #[should_panic]
    fn weibull_negative_scale() {
        Weibull::new(-1.0, 1.0).unwrap();
    }
}
