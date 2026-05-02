// Copyright 2022 The Ferric AI Project Developers

use rand_distr::Distribution as Distribution2;
use rand_distr::WeightedIndex;

use crate::distributions::Distribution;
use rand::Rng;

/// Multinomial distribution over non-negative integer count vectors.
///
/// Models $n$ independent draws from a $K$-category categorical distribution
/// with probability vector $(p_1, \ldots, p_K)$.  The result is the vector
/// of counts $(k_1, \ldots, k_K)$ where $k_i$ is the number of draws that
/// fell into category $i$.  The PMF is
///
/// $$P(\mathbf{k} \mid n, \mathbf{p}) =
///     \frac{n!}{k_1!\cdots k_K!}\prod_{i=1}^K p_i^{k_i},
///     \quad \textstyle\sum_i k_i = n$$
///
/// where $n \geq 1$ is the number of trials and $p_i > 0$ with
/// $\sum_i p_i = 1$.
///
/// See [Multinomial distribution](https://en.wikipedia.org/wiki/Multinomial_distribution)
/// on Wikipedia for further details.
///
/// # Examples
///
/// ```
/// use ferric::distributions::{Distribution, Multinomial};
/// use rand::thread_rng;
///
/// let dist = Multinomial::new(10, vec![0.2, 0.5, 0.3]).unwrap();
/// let counts: Vec<u64> = dist.sample(&mut thread_rng());
/// println!("counts = {:?}", counts);
/// ```
pub struct Multinomial {
    n: u64,
    probs: Vec<f64>,
}

impl Multinomial {
    /// Construct a Multinomial distribution with `n` trials and probability
    /// vector `probs`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `n` is zero, `probs` has fewer than 2 categories,
    /// any probability is not strictly positive, or the probabilities do not
    /// sum to 1 within $10^{-9}$.
    pub fn new(n: u64, probs: Vec<f64>) -> Result<Multinomial, String> {
        if n == 0 {
            return Err("Multinomial: n must be at least 1".to_string());
        }
        if probs.len() < 2 {
            return Err("Multinomial: probs must have at least 2 categories".to_string());
        }
        for &p in &probs {
            if p <= 0.0 {
                return Err(format!(
                    "Multinomial: all probabilities must be > 0, got {}",
                    p
                ));
            }
        }
        let sum: f64 = probs.iter().sum();
        if (sum - 1.0).abs() > 1e-9 {
            return Err(format!(
                "Multinomial: probabilities must sum to 1, got {}",
                sum
            ));
        }
        Ok(Multinomial { n, probs })
    }
}

impl<R: Rng + ?Sized> Distribution<R> for Multinomial {
    type Domain = Vec<u64>;

    fn sample(&self, rng: &mut R) -> Vec<u64> {
        let mut counts = vec![0u64; self.probs.len()];
        let weighted = WeightedIndex::new(&self.probs).unwrap();
        for _ in 0..self.n {
            counts[weighted.sample(rng)] += 1;
        }
        counts
    }

    /// Returns
    /// $\ln\Gamma(n+1) - \sum_i \ln\Gamma(k_i+1) + \sum_i k_i \ln p_i$,
    /// or $-\infty$ if `k` has the wrong length or $\sum_i k_i \neq n$.
    fn log_prob(&self, k: &Vec<u64>) -> f64 {
        if k.len() != self.probs.len() {
            return f64::NEG_INFINITY;
        }
        if k.iter().sum::<u64>() != self.n {
            return f64::NEG_INFINITY;
        }
        let n_f = self.n as f64;
        let log_multinomial = libm::lgamma(n_f + 1.0)
            - k.iter()
                .map(|&ki| libm::lgamma(ki as f64 + 1.0))
                .sum::<f64>();
        let log_kernel: f64 = k
            .iter()
            .zip(self.probs.iter())
            .map(|(&ki, &pi)| ki as f64 * pi.ln())
            .sum();
        log_multinomial + log_kernel
    }

    fn is_discrete(&self) -> bool {
        true
    }
}

impl std::fmt::Display for Multinomial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Multinomial {{ n = {}, probs = {:?} }}",
            self.n, self.probs
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn multinomial_sample() {
        let mut rng = thread_rng();
        let n = 100u64;
        let probs = vec![0.2f64, 0.5, 0.3];
        let dist = Multinomial::new(n, probs.clone()).unwrap();
        println!("dist = {}", dist);
        let trials = 1000;
        let mut sums = vec![0u64; 3];
        for _ in 0..trials {
            let counts = dist.sample(&mut rng);
            assert_eq!(counts.iter().sum::<u64>(), n);
            for (s, &c) in sums.iter_mut().zip(counts.iter()) {
                *s += c;
            }
        }
        // Check empirical category proportions are close to probs
        let total = (trials * n) as f64;
        for (i, &p) in probs.iter().enumerate() {
            let empirical = sums[i] as f64 / total;
            assert!(
                (empirical - p).abs() < 0.02,
                "category {} empirical {} != expected {}",
                i,
                empirical,
                p
            );
        }
    }

    #[test]
    fn multinomial_log_prob() {
        // Multinomial(10, [0.5, 0.5]): P([5,5]) = C(10,5) * 0.5^10 = 252/1024
        let dist = Multinomial::new(10, vec![0.5, 0.5]).unwrap();
        let lp = <Multinomial as Distribution<ThreadRng>>::log_prob(&dist, &vec![5, 5]);
        let expected = (252.0f64 / 1024.0).ln();
        assert!((lp - expected).abs() < 1e-10);

        // wrong length
        let lp_short = <Multinomial as Distribution<ThreadRng>>::log_prob(&dist, &vec![10]);
        assert_eq!(lp_short, f64::NEG_INFINITY);

        // counts don't sum to n
        let lp_sum = <Multinomial as Distribution<ThreadRng>>::log_prob(&dist, &vec![4, 5]);
        assert_eq!(lp_sum, f64::NEG_INFINITY);

        assert!(<Multinomial as Distribution<ThreadRng>>::is_discrete(&dist));
    }

    #[test]
    #[should_panic]
    fn multinomial_zero_n() {
        Multinomial::new(0, vec![0.5, 0.5]).unwrap();
    }

    #[test]
    #[should_panic]
    fn multinomial_too_few_categories() {
        Multinomial::new(10, vec![1.0]).unwrap();
    }

    #[test]
    #[should_panic]
    fn multinomial_zero_prob() {
        Multinomial::new(10, vec![0.0, 1.0]).unwrap();
    }

    #[test]
    #[should_panic]
    fn multinomial_probs_not_sum_to_one() {
        Multinomial::new(10, vec![0.4, 0.4]).unwrap();
    }
}
