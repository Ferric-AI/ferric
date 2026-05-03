// Copyright 2022 The Ferric AI Project Developers

use rand::Rng;
use rand_distr::Distribution as Distribution2;
use rand_distr::Hypergeometric as Hypergeometric2;

use crate::distributions::Distribution;

/// Hypergeometric distribution.
///
/// The PMF is
///
/// $$P(X = k \mid N, K, n) =
///     \frac{\binom{K}{k}\binom{N-K}{n-k}}{\binom{N}{n}}$$
///
/// where $N$ is the total population size, $K$ is the number of items with
/// the feature of interest in the population, and $n$ is the number of draws
/// (without replacement).
///
/// See [Hypergeometric distribution](https://en.wikipedia.org/wiki/Hypergeometric_distribution)
/// on Wikipedia for further details.
///
/// # Examples
///
/// ```
/// use ferric::distributions::{Distribution, Hypergeometric};
/// use rand::thread_rng;
///
/// let dist = Hypergeometric::new(50, 10, 5).unwrap();
/// let x: u64 = dist.sample(&mut thread_rng());
/// println!("sample = {}", x);
/// ```
pub struct Hypergeometric {
    n_total: u64,
    k_success: u64,
    n_draws: u64,
}

impl Hypergeometric {
    /// Construct a hypergeometric distribution.
    ///
    /// # Parameters
    ///
    /// - `n_total`: total population size $N$
    /// - `k_success`: items with the feature $K$ ($K \le N$)
    /// - `n_draws`: number of draws $n$ ($n \le N$)
    ///
    /// # Errors
    ///
    /// Returns `Err` if parameters are invalid.
    pub fn new(n_total: u64, k_success: u64, n_draws: u64) -> Result<Hypergeometric, String> {
        if k_success > n_total {
            return Err(format!(
                "Hypergeometric: k_success `{}` must be ≤ n_total `{}`",
                k_success, n_total
            ));
        }
        if n_draws > n_total {
            return Err(format!(
                "Hypergeometric: n_draws `{}` must be ≤ n_total `{}`",
                n_draws, n_total
            ));
        }
        Ok(Hypergeometric {
            n_total,
            k_success,
            n_draws,
        })
    }
}

impl<R: Rng + ?Sized> Distribution<R> for Hypergeometric {
    type Domain = u64;

    fn sample(&self, rng: &mut R) -> u64 {
        Hypergeometric2::new(self.n_total, self.k_success, self.n_draws)
            .unwrap()
            .sample(rng)
    }

    /// Returns the log-PMF.  Uses log-factorials via `lgamma` to avoid
    /// overflow.
    fn log_prob(&self, x: &u64) -> f64 {
        let k = *x as f64;
        let n = self.n_total as f64;
        let big_k = self.k_success as f64;
        let n_draws = self.n_draws as f64;

        // Support: max(0, n_draws + K - N) ≤ k ≤ min(K, n_draws)
        let k_min = (n_draws + big_k - n).max(0.0).ceil() as u64;
        let k_max = big_k.min(n_draws) as u64;
        if *x < k_min || *x > k_max {
            return f64::NEG_INFINITY;
        }

        log_binom(big_k, k) + log_binom(n - big_k, n_draws - k) - log_binom(n, n_draws)
    }

    fn is_discrete(&self) -> bool {
        true
    }
}

fn log_binom(n: f64, k: f64) -> f64 {
    libm::lgamma(n + 1.0) - libm::lgamma(k + 1.0) - libm::lgamma(n - k + 1.0)
}

impl std::fmt::Display for Hypergeometric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Hypergeometric {{ n_total = {}, k_success = {}, n_draws = {} }}",
            self.n_total, self.k_success, self.n_draws
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn hypergeometric_sample() {
        let mut rng = thread_rng();
        let n = 50u64;
        let k = 20u64;
        let draws = 10u64;
        let dist = Hypergeometric::new(n, k, draws).unwrap();
        println!("dist = {}", dist);
        let trials = 100_000;
        let mut total = 0.0f64;
        for _ in 0..trials {
            total += dist.sample(&mut rng) as f64;
        }
        let empirical_mean = total / trials as f64;
        // Mean = n_draws * K / N
        let expected_mean = draws as f64 * k as f64 / n as f64;
        let variance = draws as f64
            * (k as f64 / n as f64)
            * ((n - k) as f64 / n as f64)
            * ((n - draws) as f64 / (n - 1) as f64);
        let std = variance.sqrt();
        let err = 5.0 * std / (trials as f64).sqrt();
        assert!((empirical_mean - expected_mean).abs() < err);
    }

    #[test]
    fn hypergeometric_log_prob() {
        // Hyp(10, 5, 5): P(X=5) = C(5,5)*C(5,0)/C(10,5) = 1/252
        let dist = Hypergeometric::new(10, 5, 5).unwrap();
        let lp = <Hypergeometric as Distribution<ThreadRng>>::log_prob(&dist, &5);
        let expected = (1.0f64 / 252.0).ln();
        assert!((lp - expected).abs() < 1e-9);

        // out of support → NEG_INFINITY
        let lp_oob = <Hypergeometric as Distribution<ThreadRng>>::log_prob(&dist, &6);
        assert_eq!(lp_oob, f64::NEG_INFINITY);

        assert!(<Hypergeometric as Distribution<ThreadRng>>::is_discrete(
            &dist
        ));
    }

    #[test]
    fn hypergeometric_display() {
        let dist = Hypergeometric::new(50, 10, 5).unwrap();
        let s = format!("{}", dist);
        assert!(s.contains("Hypergeometric"), "missing type name: {}", s);
    }

    #[test]
    fn hypergeometric_invalid_k() {
        assert!(Hypergeometric::new(10, 11, 5).is_err());
    }

    #[test]
    fn hypergeometric_invalid_draws() {
        assert!(Hypergeometric::new(10, 5, 11).is_err());
    }
}
