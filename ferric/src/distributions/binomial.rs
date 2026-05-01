// Copyright 2022 The Ferric AI Project Developers

use rand_distr::Binomial as Binomial2;
use rand_distr::Distribution as Distribution2;

use crate::distributions::Distribution;
use rand::Rng;

/// Binomial distribution over $\{0, 1, \ldots, n\}$.
///
/// The PMF is
///
/// $$P(X = k \mid n, p) = \binom{n}{k} p^{k} (1-p)^{n-k},
///   \quad k \in \{0, 1, \ldots, n\}$$
///
/// where $n \geq 1$ is the number of trials and $p \in (0, 1)$ is the
/// probability of success on each trial.
///
/// See [Binomial distribution](https://en.wikipedia.org/wiki/Binomial_distribution)
/// on Wikipedia for further details.
///
/// # Examples
///
/// ```
/// use ferric::distributions::{Binomial, Distribution};
/// use rand::thread_rng;
///
/// let dist = Binomial::new(10, 0.3).unwrap();
/// let k: u64 = dist.sample(&mut thread_rng());
/// println!("count = {}", k);
/// ```
pub struct Binomial {
    n: u64,
    p: f64,
}

impl Binomial {
    /// Construct a Binomial distribution with `n` trials and success probability `p`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `n` is zero or if `p` is not in the open interval $(0, 1)$.
    pub fn new(n: u64, p: f64) -> Result<Binomial, String> {
        if n == 0 {
            Err("Binomial: n must be at least 1".to_string())
        } else if !(p > 0.0 && p < 1.0) {
            Err(format!(
                "Binomial: illegal p `{}` should be in the open interval (0, 1)",
                p
            ))
        } else {
            Ok(Binomial { n, p })
        }
    }
}

impl<R: Rng + ?Sized> Distribution<R> for Binomial {
    type Domain = u64;

    fn sample(&self, rng: &mut R) -> u64 {
        Binomial2::new(self.n, self.p).unwrap().sample(rng)
    }

    /// Returns
    /// $\ln\Gamma(n+1) - \ln\Gamma(k+1) - \ln\Gamma(n-k+1)
    ///  + k\ln p + (n-k)\ln(1-p)$,
    /// or $-\infty$ if $k > n$.
    fn log_prob(&self, k: &u64) -> f64 {
        if *k > self.n {
            return f64::NEG_INFINITY;
        }
        let k_f = *k as f64;
        let n_f = self.n as f64;
        let log_binom =
            libm::lgamma(n_f + 1.0) - libm::lgamma(k_f + 1.0) - libm::lgamma(n_f - k_f + 1.0);
        log_binom + k_f * self.p.ln() + (n_f - k_f) * (1.0 - self.p).ln()
    }

    fn is_discrete(&self) -> bool {
        true
    }
}

impl std::fmt::Display for Binomial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Binomial {{ n = {}, p = {} }}", self.n, self.p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn binomial_sample() {
        let mut rng = thread_rng();
        let n = 20u64;
        let p = 0.4f64;
        let dist = Binomial::new(n, p).unwrap();
        println!("dist = {}", dist);
        let mut total = 0u64;
        let trials = 10000;
        for _ in 0..trials {
            total += dist.sample(&mut rng);
        }
        let empirical_mean = (total as f64) / (trials as f64);
        let expected_mean = (n as f64) * p;
        let expected_std = ((n as f64) * p * (1.0 - p)).sqrt();
        let err = 5.0 * expected_std / (trials as f64).sqrt();
        assert!((empirical_mean - expected_mean).abs() < err);
    }

    #[test]
    fn binomial_log_prob() {
        // Binomial(10, 0.5): log_prob(5) = ln(C(10,5) * 0.5^10)
        let dist = Binomial::new(10, 0.5).unwrap();
        let lp = <Binomial as Distribution<ThreadRng>>::log_prob(&dist, &5);
        // C(10,5) = 252; 252 * (0.5)^10 = 252/1024
        let expected = (252.0f64 / 1024.0).ln();
        assert!((lp - expected).abs() < 1e-10);
        // k > n -> -inf
        let lp_out = <Binomial as Distribution<ThreadRng>>::log_prob(&dist, &11);
        assert_eq!(lp_out, f64::NEG_INFINITY);
        assert!(<Binomial as Distribution<ThreadRng>>::is_discrete(&dist));
    }

    #[test]
    #[should_panic]
    fn binomial_zero_n() {
        Binomial::new(0, 0.5).unwrap();
    }

    #[test]
    #[should_panic]
    fn binomial_p_too_low() {
        Binomial::new(10, 0.0).unwrap();
    }

    #[test]
    #[should_panic]
    fn binomial_p_too_high() {
        Binomial::new(10, 1.0).unwrap();
    }
}
