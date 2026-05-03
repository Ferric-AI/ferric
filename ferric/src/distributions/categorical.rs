// Copyright 2022 The Ferric AI Project Developers

use rand::Rng;

use crate::distributions::Distribution;

/// Categorical distribution over $\{0, 1, \ldots, k-1\}$.
///
/// The PMF is
///
/// $$P(X = i \mid \mathbf{p}) = p_i$$
///
/// where $\mathbf{p} = (p_0, \ldots, p_{k-1})$ is a probability vector
/// ($p_i \ge 0$, $\sum_i p_i = 1$).
///
/// See [Categorical distribution](https://en.wikipedia.org/wiki/Categorical_distribution)
/// on Wikipedia for further details.
///
/// # Examples
///
/// ```
/// use ferric::distributions::{Distribution, Categorical};
/// use rand::thread_rng;
///
/// let dist = Categorical::new(vec![0.1, 0.3, 0.6]).unwrap();
/// let x: usize = dist.sample(&mut thread_rng());
/// println!("sample = {}", x);
/// ```
pub struct Categorical {
    probs: Vec<f64>,
    cumulative: Vec<f64>,
}

impl Categorical {
    /// Construct a categorical distribution from a probability vector `probs`.
    ///
    /// The weights are automatically normalised, so any non-negative vector
    /// with a positive sum is accepted.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `probs` is empty, any weight is negative, or the
    /// total weight is zero.
    pub fn new(probs: Vec<f64>) -> Result<Categorical, String> {
        if probs.is_empty() {
            return Err("Categorical: probs must not be empty".into());
        }
        if probs.iter().any(|&p| p < 0.0) {
            return Err("Categorical: all probabilities must be non-negative".into());
        }
        let total: f64 = probs.iter().sum();
        if total <= 0.0 {
            return Err("Categorical: probabilities must sum to a positive value".into());
        }
        let normalised: Vec<f64> = probs.iter().map(|&p| p / total).collect();
        let mut cumulative = Vec::with_capacity(normalised.len());
        let mut acc = 0.0f64;
        for &p in &normalised {
            acc += p;
            cumulative.push(acc);
        }
        Ok(Categorical {
            probs: normalised,
            cumulative,
        })
    }
}

impl<R: Rng + ?Sized> Distribution<R> for Categorical {
    type Domain = usize;

    fn sample(&self, rng: &mut R) -> usize {
        let u: f64 = rng.r#gen();
        self.cumulative
            .iter()
            .position(|&c| u < c)
            .unwrap_or(self.probs.len() - 1)
    }

    /// Returns $\ln p_i$, or $-\infty$ if $i$ is out of range.
    fn log_prob(&self, x: &usize) -> f64 {
        self.probs
            .get(*x)
            .map(|&p| if p > 0.0 { p.ln() } else { f64::NEG_INFINITY })
            .unwrap_or(f64::NEG_INFINITY)
    }

    fn is_discrete(&self) -> bool {
        true
    }
}

impl std::fmt::Display for Categorical {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Categorical {{ probs = {:?} }}", self.probs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn categorical_sample() {
        let mut rng = thread_rng();
        let probs = vec![0.1, 0.3, 0.6];
        let dist = Categorical::new(probs.clone()).unwrap();
        println!("dist = {}", dist);
        let trials = 100_000;
        let mut counts = vec![0u64; 3];
        for _ in 0..trials {
            counts[dist.sample(&mut rng)] += 1;
        }
        for (i, &expected_p) in probs.iter().enumerate() {
            let empirical_p = counts[i] as f64 / trials as f64;
            let err = 5.0 * (expected_p * (1.0 - expected_p) / trials as f64).sqrt();
            assert!(
                (empirical_p - expected_p).abs() < err,
                "category {}: empirical {} expected {}",
                i,
                empirical_p,
                expected_p
            );
        }
    }

    #[test]
    fn categorical_log_prob() {
        let dist = Categorical::new(vec![0.25, 0.75]).unwrap();
        let lp0 = <Categorical as Distribution<ThreadRng>>::log_prob(&dist, &0);
        assert!((lp0 - (-4.0f64.ln())).abs() < 1e-10);
        let lp1 = <Categorical as Distribution<ThreadRng>>::log_prob(&dist, &1);
        assert!((lp1 - (3.0f64 / 4.0).ln()).abs() < 1e-10);
        // out of range
        let lp_oob = <Categorical as Distribution<ThreadRng>>::log_prob(&dist, &5);
        assert_eq!(lp_oob, f64::NEG_INFINITY);
        assert!(<Categorical as Distribution<ThreadRng>>::is_discrete(&dist));
    }

    #[test]
    fn categorical_unnormalized_weights() {
        // unnormalized weights should be accepted
        let dist = Categorical::new(vec![1.0, 3.0]).unwrap();
        let lp0 = <Categorical as Distribution<ThreadRng>>::log_prob(&dist, &0);
        assert!((lp0 - (-4.0f64.ln())).abs() < 1e-10);
    }

    #[test]
    fn categorical_zero_weight_category() {
        let dist = Categorical::new(vec![0.0, 1.0]).unwrap();
        assert_eq!(
            <Categorical as Distribution<ThreadRng>>::log_prob(&dist, &0),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn categorical_display() {
        let dist = Categorical::new(vec![0.5, 0.5]).unwrap();
        let s = format!("{}", dist);
        assert!(s.contains("Categorical"), "missing type name: {}", s);
    }

    #[test]
    fn categorical_empty() {
        assert!(Categorical::new(vec![]).is_err());
    }

    #[test]
    fn categorical_negative_prob() {
        assert!(Categorical::new(vec![-1.0, 2.0]).is_err());
    }

    #[test]
    fn categorical_zero_total() {
        assert!(Categorical::new(vec![0.0, 0.0]).is_err());
    }
}
