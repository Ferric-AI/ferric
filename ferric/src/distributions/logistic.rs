// Copyright 2022 The Ferric AI Project Developers

use rand::Rng;
use rand_distr::Distribution as Distribution2;
use rand_distr::Uniform;

use crate::distributions::Distribution;

/// Logistic distribution over the reals.
///
/// The PDF is
///
/// $$p(x \mid \mu, s) = \frac{e^{-(x-\mu)/s}}{s\,(1 + e^{-(x-\mu)/s})^2}$$
///
/// where $\mu \in \mathbb{R}$ is the location parameter and $s > 0$ is the
/// scale parameter.  The distribution is equivalent to the difference of two
/// independent standard Gumbel random variables.
///
/// See [Logistic distribution](https://en.wikipedia.org/wiki/Logistic_distribution)
/// on Wikipedia for further details.
///
/// # Examples
///
/// ```
/// use ferric::distributions::{Distribution, Logistic};
/// use rand::thread_rng;
///
/// let dist = Logistic::new(0.0, 1.0).unwrap();
/// let x: f64 = dist.sample(&mut thread_rng());
/// println!("sample = {:.4}", x);
/// ```
pub struct Logistic {
    mu: f64,
    s: f64,
}

impl Logistic {
    /// Construct a Logistic distribution with location `mu` ($\mu$) and
    /// scale `s` ($s$).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `s` is not strictly positive.
    pub fn new(mu: f64, s: f64) -> Result<Logistic, String> {
        if s <= 0.0 {
            Err(format!(
                "Logistic: illegal scale `{}` should be greater than 0",
                s
            ))
        } else {
            Ok(Logistic { mu, s })
        }
    }
}

impl<R: Rng + ?Sized> Distribution<R> for Logistic {
    type Domain = f64;

    /// Draw a sample via the quantile function:
    /// $X = \mu + s \ln\!\left(\frac{u}{1-u}\right)$ where $u \sim U(0,1)$.
    fn sample(&self, rng: &mut R) -> f64 {
        // Use open unit interval to avoid ln(0)
        let u = Uniform::new(0.0f64, 1.0).sample(rng);
        // Clamp away from 0 and 1 for numerical safety
        let u = u.clamp(1e-15, 1.0 - 1e-15);
        self.mu + self.s * (u / (1.0 - u)).ln()
    }

    /// Returns $-(x-\mu)/s - \ln s - 2\ln(1 + e^{-(x-\mu)/s})$.
    fn log_prob(&self, x: &f64) -> f64 {
        let z = (x - self.mu) / self.s;
        // Use numerically stable softplus: ln(1 + e^{-|z|}) + max(z, 0)
        let softplus_neg_z = (-z.abs()).exp().ln_1p() + z.max(0.0);
        -z - self.s.ln() - 2.0 * softplus_neg_z
    }

    fn is_discrete(&self) -> bool {
        false
    }
}

impl std::fmt::Display for Logistic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Logistic {{ mu = {}, s = {} }}", self.mu, self.s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn logistic_sample() {
        let mut rng = thread_rng();
        let mu = 3.0f64;
        let s = 2.0f64;
        let dist = Logistic::new(mu, s).unwrap();
        println!("dist = {}", dist);
        let trials = 100_000;
        let mut total = 0.0f64;
        for _ in 0..trials {
            total += dist.sample(&mut rng);
        }
        let empirical_mean = total / trials as f64;
        // Mean = mu, Std = s * pi / sqrt(3)
        let std = s * std::f64::consts::PI / 3.0f64.sqrt();
        let err = 5.0 * std / (trials as f64).sqrt();
        assert!((empirical_mean - mu).abs() < err);
    }

    #[test]
    fn logistic_log_prob() {
        // Logistic(0, 1) at x=0: log_prob = -ln(4)
        let dist = Logistic::new(0.0, 1.0).unwrap();
        let lp = <Logistic as Distribution<ThreadRng>>::log_prob(&dist, &0.0);
        let expected = -(4.0f64).ln();
        assert!((lp - expected).abs() < 1e-10);

        // At large |x|, log_prob should not overflow
        let lp_large = <Logistic as Distribution<ThreadRng>>::log_prob(&dist, &100.0);
        assert!(lp_large.is_finite());

        assert!(!<Logistic as Distribution<ThreadRng>>::is_discrete(&dist));
    }

    #[test]
    fn logistic_display() {
        let dist = Logistic::new(1.0, 2.0).unwrap();
        let s = format!("{}", dist);
        assert!(s.contains("Logistic"), "missing type name: {}", s);
    }

    #[test]
    #[should_panic]
    fn logistic_zero_scale() {
        Logistic::new(0.0, 0.0).unwrap();
    }

    #[test]
    #[should_panic]
    fn logistic_negative_scale() {
        Logistic::new(0.0, -1.0).unwrap();
    }
}
