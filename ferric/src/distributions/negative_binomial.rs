// Copyright 2022 The Ferric AI Project Developers

use rand::Rng;
use rand_distr::Distribution as Distribution2;
use rand_distr::Gamma as Gamma2;
use rand_distr::Poisson;

use crate::distributions::Distribution;

/// Negative binomial distribution over the non-negative integers.
///
/// The PMF is
///
/// $$P(X = k \mid r, p) =
///     \binom{k+r-1}{k} p^r (1-p)^k$$
///
/// where $r > 0$ is the number of successes and $p \in (0, 1]$ is the
/// success probability.  $X$ counts the number of failures before the $r$-th
/// success.
///
/// See [Negative binomial distribution](https://en.wikipedia.org/wiki/Negative_binomial_distribution)
/// on Wikipedia for further details.
///
/// # Examples
///
/// ```
/// use ferric::distributions::{Distribution, NegativeBinomial};
/// use rand::thread_rng;
///
/// let dist = NegativeBinomial::new(5.0, 0.4).unwrap();
/// let x: u64 = dist.sample(&mut thread_rng());
/// println!("sample = {}", x);
/// ```
pub struct NegativeBinomial {
    r: f64,
    p: f64,
}

impl NegativeBinomial {
    /// Construct a negative binomial distribution with `r` successes and
    /// success probability `p`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `r` is not strictly positive or `p` is not in
    /// $(0, 1]$.
    pub fn new(r: f64, p: f64) -> Result<NegativeBinomial, String> {
        if r <= 0.0 {
            Err(format!(
                "NegativeBinomial: illegal r `{}` should be greater than 0",
                r
            ))
        } else if !(p > 0.0 && p <= 1.0) {
            Err(format!(
                "NegativeBinomial: illegal p `{}` must be in (0, 1]",
                p
            ))
        } else {
            Ok(NegativeBinomial { r, p })
        }
    }
}

impl<R: Rng + ?Sized> Distribution<R> for NegativeBinomial {
    type Domain = u64;

    /// Draw a sample via the Poisson-Gamma mixture:
    /// $\lambda \sim \mathrm{Gamma}(r, (1-p)/p)$ then $X \sim \mathrm{Poisson}(\lambda)$.
    fn sample(&self, rng: &mut R) -> u64 {
        if self.p == 1.0 {
            return 0;
        }
        let scale = (1.0 - self.p) / self.p;
        let lambda = Gamma2::new(self.r, scale).unwrap().sample(rng);
        Poisson::new(lambda).unwrap().sample(rng) as u64
    }

    /// Returns $\ln\Gamma(k+r) - \ln\Gamma(r) - \ln(k!) + r\ln p + k\ln(1-p)$.
    fn log_prob(&self, x: &u64) -> f64 {
        if self.p == 1.0 {
            return if *x == 0 { 0.0 } else { f64::NEG_INFINITY };
        }
        let k = *x as f64;
        libm::lgamma(k + self.r) - libm::lgamma(self.r) - libm::lgamma(k + 1.0)
            + self.r * self.p.ln()
            + k * (1.0 - self.p).ln()
    }

    fn is_discrete(&self) -> bool {
        true
    }
}

impl std::fmt::Display for NegativeBinomial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NegativeBinomial {{ r = {}, p = {} }}", self.r, self.p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn negative_binomial_sample() {
        let mut rng = thread_rng();
        let r = 5.0f64;
        let p = 0.4f64;
        let dist = NegativeBinomial::new(r, p).unwrap();
        println!("dist = {}", dist);
        let trials = 100_000;
        let mut total = 0.0f64;
        for _ in 0..trials {
            total += dist.sample(&mut rng) as f64;
        }
        let empirical_mean = total / trials as f64;
        // Mean = r * (1 - p) / p
        let expected_mean = r * (1.0 - p) / p;
        let variance = r * (1.0 - p) / (p * p);
        let std = variance.sqrt();
        let err = 5.0 * std / (trials as f64).sqrt();
        assert!((empirical_mean - expected_mean).abs() < err);
    }

    #[test]
    fn negative_binomial_log_prob() {
        // NegBin(1, 0.5) at k=0: ln(1) + 0 + 1*ln(0.5) + 0 = -ln(2)
        let dist = NegativeBinomial::new(1.0, 0.5).unwrap();
        let lp = <NegativeBinomial as Distribution<ThreadRng>>::log_prob(&dist, &0);
        assert!((lp - (-2.0f64.ln())).abs() < 1e-10);
        assert!(<NegativeBinomial as Distribution<ThreadRng>>::is_discrete(
            &dist
        ));
    }

    #[test]
    fn negative_binomial_certain_success() {
        let dist = NegativeBinomial::new(2.0, 1.0).unwrap();
        assert_eq!(dist.sample(&mut thread_rng()), 0);
        assert_eq!(
            <NegativeBinomial as Distribution<ThreadRng>>::log_prob(&dist, &0),
            0.0
        );
        assert_eq!(
            <NegativeBinomial as Distribution<ThreadRng>>::log_prob(&dist, &1),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn negative_binomial_display() {
        let dist = NegativeBinomial::new(5.0, 0.4).unwrap();
        let s = format!("{}", dist);
        assert!(s.contains("NegativeBinomial"), "missing type name: {}", s);
    }

    #[test]
    #[should_panic]
    fn negative_binomial_zero_r() {
        NegativeBinomial::new(0.0, 0.5).unwrap();
    }

    #[test]
    #[should_panic]
    fn negative_binomial_invalid_p() {
        NegativeBinomial::new(1.0, 0.0).unwrap();
    }
}
