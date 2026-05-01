// Copyright 2022 The Ferric AI Project Developers

use rand_distr::Distribution as Distribution2;
use rand_distr::LogNormal as LogNormal2;

use crate::distributions::Distribution;
use rand::Rng;

/// Log-normal distribution over positive reals.
///
/// If $X \sim \mathrm{LogNormal}(\mu, \sigma)$ then $\ln X \sim \mathcal{N}(\mu, \sigma^2)$.
/// The PDF is
///
/// $$p(x \mid \mu, \sigma) =
///     \frac{1}{x\,\sigma\sqrt{2\pi}}
///     \exp\!\left(-\frac{(\ln x - \mu)^{2}}{2\sigma^{2}}\right),
///     \quad x > 0$$
///
/// where $\mu \in \mathbb{R}$ is the mean of the underlying normal and
/// $\sigma > 0$ is the standard deviation of the underlying normal.
///
/// See [Log-normal distribution](https://en.wikipedia.org/wiki/Log-normal_distribution)
/// on Wikipedia for further details.
///
/// # Examples
///
/// ```
/// use ferric::distributions::{Distribution, LogNormal};
/// use rand::thread_rng;
///
/// let dist = LogNormal::new(0.0, 1.0).unwrap();
/// let x: f64 = dist.sample(&mut thread_rng());
/// println!("sample = {:.4}", x);
/// ```
pub struct LogNormal {
    mu: f64,
    sigma: f64,
}

impl LogNormal {
    /// Construct a LogNormal distribution with underlying-normal mean `mu` ($\mu$)
    /// and standard deviation `sigma` ($\sigma$).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `sigma` is not strictly positive.
    pub fn new(mu: f64, sigma: f64) -> Result<LogNormal, String> {
        if sigma <= 0.0 {
            Err(format!(
                "LogNormal: illegal sigma `{}` should be greater than 0",
                sigma
            ))
        } else {
            Ok(LogNormal { mu, sigma })
        }
    }
}

impl<R: Rng + ?Sized> Distribution<R> for LogNormal {
    type Domain = f64;

    fn sample(&self, rng: &mut R) -> f64 {
        LogNormal2::new(self.mu, self.sigma).unwrap().sample(rng)
    }

    /// Returns
    /// $-\tfrac{1}{2}\!\left(\tfrac{\ln x - \mu}{\sigma}\right)^{\!2}
    ///   - \ln x - \ln\sigma - \tfrac{1}{2}\ln(2\pi)$
    /// for $x > 0$, or $-\infty$ otherwise.
    fn log_prob(&self, x: &f64) -> f64 {
        if *x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let z = (x.ln() - self.mu) / self.sigma;
        -0.5 * z * z - x.ln() - self.sigma.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
    }

    fn is_discrete(&self) -> bool {
        false
    }
}

impl std::fmt::Display for LogNormal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LogNormal {{ mu = {}, sigma = {} }}",
            self.mu, self.sigma
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn log_normal_sample() {
        let mut rng = thread_rng();
        let mu = 0.0f64;
        let sigma = 0.5f64;
        let dist = LogNormal::new(mu, sigma).unwrap();
        println!("dist = {}", dist);
        let mut total = 0f64;
        let trials = 10000;
        for _ in 0..trials {
            total += dist.sample(&mut rng);
        }
        let empirical_mean = total / (trials as f64);
        // E[X] = exp(mu + sigma^2/2)
        let expected_mean = (mu + sigma * sigma / 2.0).exp();
        // Var[X] = (exp(sigma^2)-1)*exp(2*mu+sigma^2)
        let expected_var = (sigma * sigma).exp_m1() * (2.0 * mu + sigma * sigma).exp();
        let err = 5.0 * expected_var.sqrt() / (trials as f64).sqrt();
        assert!((empirical_mean - expected_mean).abs() < err);
    }

    #[test]
    fn log_normal_log_prob() {
        // LogNormal(0, 1): log_prob(1) = -0.5*ln(2π)
        let dist = LogNormal::new(0.0, 1.0).unwrap();
        let lp = <LogNormal as Distribution<ThreadRng>>::log_prob(&dist, &1.0);
        let expected = -0.5 * (2.0 * std::f64::consts::PI).ln();
        assert!((lp - expected).abs() < 1e-10);
        // outside support
        let lp_out = <LogNormal as Distribution<ThreadRng>>::log_prob(&dist, &-1.0);
        assert_eq!(lp_out, f64::NEG_INFINITY);
        let lp_zero = <LogNormal as Distribution<ThreadRng>>::log_prob(&dist, &0.0);
        assert_eq!(lp_zero, f64::NEG_INFINITY);
        assert!(!<LogNormal as Distribution<ThreadRng>>::is_discrete(&dist));
    }

    #[test]
    #[should_panic]
    fn log_normal_zero_sigma() {
        LogNormal::new(0.0, 0.0).unwrap();
    }

    #[test]
    #[should_panic]
    fn log_normal_negative_sigma() {
        LogNormal::new(0.0, -1.0).unwrap();
    }
}
