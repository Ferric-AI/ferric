// Copyright 2022 The Ferric AI Project Developers

use crate::distributions::Distribution;
use rand::Rng;

/// Bernoulli distribution over `{false, true}`.
///
/// The PMF is
///
/// $$P(X = x \mid p) = p^{x}(1-p)^{1-x}, \quad x \in \{0, 1\}$$
///
/// where $x = 1$ corresponds to `true` and $x = 0$ to `false`, and $p \in [0, 1]$
/// is the probability of success.
///
/// See [Bernoulli distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution)
/// on Wikipedia for further details.
///
/// # Examples
///
/// ```
/// use ferric::distributions::{Bernoulli, Distribution};
/// use rand::thread_rng;
///
/// let dist = Bernoulli::new(0.3).unwrap();
/// let sample: bool = dist.sample(&mut thread_rng());
/// println!("outcome = {}", sample);
/// ```
pub struct Bernoulli {
    /// Probability of success.
    p: f64,
}

impl Bernoulli {
    /// Construct a Bernoulli distribution with success probability `p`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `p` is not in `[0, 1]`.
    pub fn new(p: f64) -> Result<Bernoulli, String> {
        if !(0f64..=1f64).contains(&p) {
            Err(format! {"Bernoulli: illegal probability `{}` should be between 0 and 1", p})
        } else {
            Ok(Bernoulli { p })
        }
    }
}

impl<R: Rng + ?Sized> Distribution<R> for Bernoulli {
    type Domain = bool;

    fn sample(&self, rng: &mut R) -> bool {
        let val: f64 = rng.r#gen();
        val < self.p
    }

    /// Returns $\log p$ when `x` is `true`, or $\log(1-p)$ when `x` is `false`.
    fn log_prob(&self, x: &bool) -> f64 {
        if *x { self.p.ln() } else { (1.0 - self.p).ln() }
    }

    fn is_discrete(&self) -> bool {
        true
    }
}

impl std::fmt::Display for Bernoulli {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Bernoulli {{ p = {} }}", self.p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::Distribution;
    use rand::thread_rng;

    #[test]
    fn bernoulli_sample() {
        let mut rng = thread_rng();
        let dist = Bernoulli::new(0.1).unwrap();
        println!("dist = {}", dist);
        let mut succ = 0;
        let trials = 10000;
        for _ in 0..trials {
            if dist.sample(&mut rng) {
                succ += 1;
            }
        }
        let mean = (succ as f64) / (trials as f64);
        assert!((mean - 0.1).abs() < 0.01);
        // both of the following extreme distributios should be allowed
        let _dist2 = Bernoulli::new(0.0).unwrap();
        let _dist3 = Bernoulli::new(1.0).unwrap();
    }

    #[test]
    #[should_panic]
    fn bernoulli_too_low() {
        let _dist = Bernoulli::new(-0.01).unwrap();
    }

    #[test]
    #[should_panic]
    fn bernoulli_too_high() {
        let _dist = Bernoulli::new(1.01).unwrap();
    }

    #[test]
    fn bernoulli_log_prob_and_display() {
        let dist = Bernoulli::new(0.25).unwrap();
        // true -> ln(p), false -> ln(1-p)
        let lp_true = Distribution::<rand::rngs::ThreadRng>::log_prob(&dist, &true);
        let lp_false = Distribution::<rand::rngs::ThreadRng>::log_prob(&dist, &false);
        assert_eq!(lp_true, 0.25f64.ln());
        assert_eq!(lp_false, (1.0f64 - 0.25).ln());
        // extremes: p = 0 -> ln(0) == -inf for true
        let d0 = Bernoulli::new(0.0).unwrap();
        assert!(Distribution::<rand::rngs::ThreadRng>::log_prob(&d0, &true).is_infinite());
        assert_eq!(
            Distribution::<rand::rngs::ThreadRng>::log_prob(&d0, &false),
            (1.0f64 - 0.0).ln()
        );
        // p = 1 -> ln(1-p) == -inf for false
        let d1 = Bernoulli::new(1.0).unwrap();
        assert_eq!(
            Distribution::<rand::rngs::ThreadRng>::log_prob(&d1, &true),
            (1.0f64).ln()
        );
        assert!(Distribution::<rand::rngs::ThreadRng>::log_prob(&d1, &false).is_infinite());
        // display and is_discrete
        assert!(format!("{}", dist).contains("Bernoulli"));
        assert!(Distribution::<rand::rngs::ThreadRng>::is_discrete(&dist));
    }
}
