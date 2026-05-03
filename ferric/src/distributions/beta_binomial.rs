// Copyright 2022 The Ferric AI Project Developers

use rand::Rng;
use rand_distr::Beta as Beta2;
use rand_distr::Binomial;
use rand_distr::Distribution as Distribution2;

use crate::distributions::Distribution;

/// Beta-binomial distribution over $\{0, 1, \ldots, n\}$.
///
/// The PMF is
///
/// $$P(X = k \mid n, \alpha, \beta) =
///     \binom{n}{k}
///     \frac{B(k+\alpha,\,n-k+\beta)}{B(\alpha,\,\beta)}$$
///
/// where $n \ge 1$ is the number of trials, $\alpha > 0$ and $\beta > 0$ are
/// the Beta prior parameters, and $B$ is the beta function.  This is a
/// binomial distribution whose success probability is drawn from a
/// $\mathrm{Beta}(\alpha, \beta)$ prior.
///
/// See [Beta-binomial distribution](https://en.wikipedia.org/wiki/Beta-binomial_distribution)
/// on Wikipedia for further details.
///
/// # Examples
///
/// ```
/// use ferric::distributions::{Distribution, BetaBinomial};
/// use rand::thread_rng;
///
/// let dist = BetaBinomial::new(10, 2.0, 5.0).unwrap();
/// let x: u64 = dist.sample(&mut thread_rng());
/// println!("sample = {}", x);
/// ```
pub struct BetaBinomial {
    n: u64,
    alpha: f64,
    beta: f64,
}

impl BetaBinomial {
    /// Construct a beta-binomial distribution with `n` trials and Beta prior
    /// parameters `alpha` and `beta`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `n` is zero or `alpha`/`beta` are not strictly
    /// positive.
    pub fn new(n: u64, alpha: f64, beta: f64) -> Result<BetaBinomial, String> {
        if n == 0 {
            return Err("BetaBinomial: n must be at least 1".into());
        }
        if alpha <= 0.0 {
            return Err(format!(
                "BetaBinomial: illegal alpha `{}` should be greater than 0",
                alpha
            ));
        }
        if beta <= 0.0 {
            return Err(format!(
                "BetaBinomial: illegal beta `{}` should be greater than 0",
                beta
            ));
        }
        Ok(BetaBinomial { n, alpha, beta })
    }
}

impl<R: Rng + ?Sized> Distribution<R> for BetaBinomial {
    type Domain = u64;

    /// Draw a sample by first drawing $p \sim \mathrm{Beta}(\alpha, \beta)$,
    /// then $X \sim \mathrm{Binomial}(n, p)$.
    fn sample(&self, rng: &mut R) -> u64 {
        let p = Beta2::new(self.alpha, self.beta).unwrap().sample(rng);
        Binomial::new(self.n, p).unwrap().sample(rng)
    }

    /// Returns the log-PMF using the log-beta function.
    fn log_prob(&self, x: &u64) -> f64 {
        let k = *x as f64;
        let n = self.n as f64;
        if k > n {
            return f64::NEG_INFINITY;
        }
        log_binom(n, k) + log_beta(k + self.alpha, n - k + self.beta)
            - log_beta(self.alpha, self.beta)
    }

    fn is_discrete(&self) -> bool {
        true
    }
}

fn log_binom(n: f64, k: f64) -> f64 {
    libm::lgamma(n + 1.0) - libm::lgamma(k + 1.0) - libm::lgamma(n - k + 1.0)
}

fn log_beta(a: f64, b: f64) -> f64 {
    libm::lgamma(a) + libm::lgamma(b) - libm::lgamma(a + b)
}

impl std::fmt::Display for BetaBinomial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BetaBinomial {{ n = {}, alpha = {}, beta = {} }}",
            self.n, self.alpha, self.beta
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn beta_binomial_sample() {
        let mut rng = thread_rng();
        let n = 10u64;
        let alpha = 2.0f64;
        let beta = 5.0f64;
        let dist = BetaBinomial::new(n, alpha, beta).unwrap();
        println!("dist = {}", dist);
        let trials = 100_000;
        let mut total = 0.0f64;
        for _ in 0..trials {
            total += dist.sample(&mut rng) as f64;
        }
        let empirical_mean = total / trials as f64;
        // Mean = n * alpha / (alpha + beta)
        let expected_mean = n as f64 * alpha / (alpha + beta);
        let variance = n as f64 * alpha * beta * (alpha + beta + n as f64)
            / ((alpha + beta).powi(2) * (alpha + beta + 1.0));
        let std = variance.sqrt();
        let err = 5.0 * std / (trials as f64).sqrt();
        assert!((empirical_mean - expected_mean).abs() < err);
    }

    #[test]
    fn beta_binomial_log_prob() {
        // BetaBinomial(1, 1, 1) = Uniform{0, 1}: P(X=0) = P(X=1) = 0.5
        let dist = BetaBinomial::new(1, 1.0, 1.0).unwrap();
        let lp0 = <BetaBinomial as Distribution<ThreadRng>>::log_prob(&dist, &0);
        let lp1 = <BetaBinomial as Distribution<ThreadRng>>::log_prob(&dist, &1);
        assert!((lp0 - (-2.0f64.ln())).abs() < 1e-9);
        assert!((lp1 - (-2.0f64.ln())).abs() < 1e-9);
        assert!(<BetaBinomial as Distribution<ThreadRng>>::is_discrete(
            &dist
        ));
    }

    #[test]
    fn beta_binomial_display() {
        let dist = BetaBinomial::new(10, 2.0, 5.0).unwrap();
        let s = format!("{}", dist);
        assert!(s.contains("BetaBinomial"), "missing type name: {}", s);
    }

    #[test]
    fn beta_binomial_zero_n() {
        assert!(BetaBinomial::new(0, 1.0, 1.0).is_err());
    }

    #[test]
    fn beta_binomial_invalid_alpha() {
        assert!(BetaBinomial::new(10, 0.0, 1.0).is_err());
    }

    #[test]
    fn beta_binomial_invalid_beta() {
        assert!(BetaBinomial::new(10, 1.0, 0.0).is_err());
    }

    #[test]
    fn beta_binomial_log_prob_out_of_range() {
        let dist = BetaBinomial::new(10, 1.0, 1.0).unwrap();
        assert_eq!(
            <BetaBinomial as Distribution<ThreadRng>>::log_prob(&dist, &11),
            f64::NEG_INFINITY
        );
    }
}
