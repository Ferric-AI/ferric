// Copyright 2022 The Ferric AI Project Developers

use rand_distr::Distribution as Distribution2;
use rand_distr::Geometric as Geometric2;

use crate::distributions::Distribution;
use rand::Rng;

/// Geometric distribution over $\{0, 1, 2, \ldots\}$.
///
/// Models the number of failures before the first success in a sequence of
/// independent Bernoulli trials.  The PMF is
///
/// $$P(X = k \mid p) = p\,(1-p)^{k}, \quad k \in \{0, 1, 2, \ldots\}$$
///
/// where $p \in (0, 1]$ is the probability of success on each trial.
///
/// See [Geometric distribution](https://en.wikipedia.org/wiki/Geometric_distribution)
/// on Wikipedia for further details.
///
/// # Examples
///
/// ```
/// use ferric::distributions::{Distribution, Geometric};
/// use rand::thread_rng;
///
/// let dist = Geometric::new(0.3).unwrap();
/// let k: u64 = dist.sample(&mut thread_rng());
/// println!("failures before first success = {}", k);
/// ```
pub struct Geometric {
    p: f64,
}

impl Geometric {
    /// Construct a Geometric distribution with success probability `p`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `p` is not in the half-open interval $(0, 1]$.
    pub fn new(p: f64) -> Result<Geometric, String> {
        if !(p > 0.0 && p <= 1.0) {
            Err(format!(
                "Geometric: illegal p `{}` should be in the interval (0, 1]",
                p
            ))
        } else {
            Ok(Geometric { p })
        }
    }
}

impl<R: Rng + ?Sized> Distribution<R> for Geometric {
    type Domain = u64;

    fn sample(&self, rng: &mut R) -> u64 {
        Geometric2::new(self.p).unwrap().sample(rng)
    }

    /// Returns $\ln p + k\,\ln(1-p)$, handling $p = 1$ exactly.
    fn log_prob(&self, k: &u64) -> f64 {
        if self.p == 1.0 {
            // P(X=0) = 1, P(X>0) = 0
            if *k == 0 { 0.0 } else { f64::NEG_INFINITY }
        } else {
            self.p.ln() + (*k as f64) * (1.0 - self.p).ln()
        }
    }

    fn is_discrete(&self) -> bool {
        true
    }
}

impl std::fmt::Display for Geometric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Geometric {{ p = {} }}", self.p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn geometric_sample() {
        let mut rng = thread_rng();
        let p = 0.3f64;
        let dist = Geometric::new(p).unwrap();
        println!("dist = {}", dist);
        let mut total = 0u64;
        let trials = 10000;
        for _ in 0..trials {
            total += dist.sample(&mut rng);
        }
        let empirical_mean = (total as f64) / (trials as f64);
        let expected_mean = (1.0 - p) / p;
        let expected_std = ((1.0 - p) / (p * p)).sqrt();
        let err = 5.0 * expected_std / (trials as f64).sqrt();
        assert!((empirical_mean - expected_mean).abs() < err);
    }

    #[test]
    fn geometric_log_prob() {
        // Geometric(0.5): log_prob(0) = ln(0.5), log_prob(1) = 2*ln(0.5)
        let dist = Geometric::new(0.5).unwrap();
        let lp0 = <Geometric as Distribution<ThreadRng>>::log_prob(&dist, &0);
        assert!((lp0 - 0.5f64.ln()).abs() < 1e-10);
        let lp1 = <Geometric as Distribution<ThreadRng>>::log_prob(&dist, &1);
        assert!((lp1 - 2.0 * 0.5f64.ln()).abs() < 1e-10);
        assert!(<Geometric as Distribution<ThreadRng>>::is_discrete(&dist));
    }

    #[test]
    fn geometric_p_one() {
        // p=1: always succeeds on first try -> P(X=0)=1, P(X>0)=0
        let dist = Geometric::new(1.0).unwrap();
        let lp0 = <Geometric as Distribution<ThreadRng>>::log_prob(&dist, &0);
        assert_eq!(lp0, 0.0);
        let lp1 = <Geometric as Distribution<ThreadRng>>::log_prob(&dist, &1);
        assert_eq!(lp1, f64::NEG_INFINITY);
    }

    #[test]
    #[should_panic]
    fn geometric_zero_p() {
        Geometric::new(0.0).unwrap();
    }

    #[test]
    #[should_panic]
    fn geometric_p_too_high() {
        Geometric::new(1.1).unwrap();
    }
}
