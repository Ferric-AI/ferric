// Copyright 2022 The Ferric AI Project Developers

use rand_distr::Beta as Beta2;
use rand_distr::Distribution as Distribution2;

use crate::distributions::Distribution;
use rand::Rng;

/// Beta distribution over the open interval $(0, 1)$.
///
/// The PDF is
///
/// $$p(x \mid \alpha, \beta) =
///     \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)},
///     \quad x \in (0, 1)$$
///
/// where $B(\alpha, \beta) = \Gamma(\alpha)\Gamma(\beta)/\Gamma(\alpha+\beta)$
/// is the beta function and $\alpha, \beta > 0$ are the shape parameters.
///
/// See [Beta distribution](https://en.wikipedia.org/wiki/Beta_distribution)
/// on Wikipedia for further details.
///
/// # Examples
///
/// ```
/// use ferric::distributions::{Beta, Distribution};
/// use rand::thread_rng;
///
/// let dist = Beta::new(2.0, 5.0).unwrap();
/// let x: f64 = dist.sample(&mut thread_rng());
/// println!("sample = {:.4}", x);
/// ```
pub struct Beta {
    alpha: f64,
    beta: f64,
}

impl Beta {
    /// Construct a Beta distribution with shape parameters `alpha` ($\alpha$)
    /// and `beta` ($\beta$).
    ///
    /// # Errors
    ///
    /// Returns `Err` if either `alpha` or `beta` is not strictly positive.
    pub fn new(alpha: f64, beta: f64) -> Result<Beta, String> {
        if alpha <= 0.0 {
            Err(format!(
                "Beta: illegal alpha `{}` should be greater than 0",
                alpha
            ))
        } else if beta <= 0.0 {
            Err(format!(
                "Beta: illegal beta `{}` should be greater than 0",
                beta
            ))
        } else {
            Ok(Beta { alpha, beta })
        }
    }
}

impl<R: Rng + ?Sized> Distribution<R> for Beta {
    type Domain = f64;

    fn sample(&self, rng: &mut R) -> f64 {
        Beta2::new(self.alpha, self.beta).unwrap().sample(rng)
    }

    /// Returns $(\alpha-1)\ln x + (\beta-1)\ln(1-x) - \ln B(\alpha,\beta)$
    /// where $\ln B(\alpha,\beta) = \ln\Gamma(\alpha) + \ln\Gamma(\beta) - \ln\Gamma(\alpha+\beta)$.
    fn log_prob(&self, x: &f64) -> f64 {
        if *x <= 0.0 || *x >= 1.0 {
            return f64::NEG_INFINITY;
        }
        let ln_beta = libm::lgamma(self.alpha) + libm::lgamma(self.beta)
            - libm::lgamma(self.alpha + self.beta);
        (self.alpha - 1.0) * x.ln() + (self.beta - 1.0) * (1.0 - x).ln() - ln_beta
    }

    fn is_discrete(&self) -> bool {
        false
    }
}

impl std::fmt::Display for Beta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Beta {{ alpha = {}, beta = {} }}", self.alpha, self.beta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn beta_sample() {
        let mut rng = thread_rng();
        let alpha = 2.0f64;
        let beta = 5.0f64;
        let dist = Beta::new(alpha, beta).unwrap();
        println!("dist = {}", dist);
        let mut total = 0f64;
        let trials = 10000;
        for _ in 0..trials {
            total += dist.sample(&mut rng);
        }
        let empirical_mean = total / (trials as f64);
        let expected_mean = alpha / (alpha + beta);
        let expected_var = alpha * beta / ((alpha + beta).powi(2) * (alpha + beta + 1.0));
        let err = 5.0 * expected_var.sqrt() / (trials as f64).sqrt();
        assert!((empirical_mean - expected_mean).abs() < err);
    }

    #[test]
    fn beta_log_prob() {
        // Beta(1,1) = Uniform(0,1): log_prob(0.5) = 0.0
        let dist = Beta::new(1.0, 1.0).unwrap();
        let lp = <Beta as Distribution<ThreadRng>>::log_prob(&dist, &0.5);
        assert!((lp - 0.0).abs() < 1e-10);
        // outside support
        let lp_out = <Beta as Distribution<ThreadRng>>::log_prob(&dist, &1.5);
        assert_eq!(lp_out, f64::NEG_INFINITY);
        let lp_low = <Beta as Distribution<ThreadRng>>::log_prob(&dist, &-0.1);
        assert_eq!(lp_low, f64::NEG_INFINITY);
        assert!(!<Beta as Distribution<ThreadRng>>::is_discrete(&dist));
    }

    #[test]
    #[should_panic]
    fn beta_zero_alpha() {
        Beta::new(0.0, 1.0).unwrap();
    }

    #[test]
    #[should_panic]
    fn beta_negative_beta() {
        Beta::new(1.0, -1.0).unwrap();
    }
}
