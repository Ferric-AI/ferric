// Copyright 2022 The Ferric AI Project Developers

use rand::Rng;
use rand::distributions::Uniform as RandUniform;
use rand_distr::Distribution as Distribution2;

use crate::distributions::Distribution;

/// Discrete uniform distribution over $\{a, a+1, \ldots, b\}$.
///
/// The PMF is
///
/// $$P(X = k \mid a, b) = \frac{1}{b - a + 1}$$
///
/// for integer $k \in [a, b]$.
///
/// See [Discrete uniform distribution](https://en.wikipedia.org/wiki/Discrete_uniform_distribution)
/// on Wikipedia for further details.
///
/// # Examples
///
/// ```
/// use ferric::distributions::{Distribution, DiscreteUniform};
/// use rand::thread_rng;
///
/// let dist = DiscreteUniform::new(1, 6).unwrap();
/// let x: i64 = dist.sample(&mut thread_rng());
/// println!("sample = {}", x);
/// ```
pub struct DiscreteUniform {
    a: i64,
    b: i64,
}

impl DiscreteUniform {
    /// Construct a discrete uniform distribution on $\{a, \ldots, b\}$.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `a > b`.
    pub fn new(a: i64, b: i64) -> Result<DiscreteUniform, String> {
        if a > b {
            Err(format!(
                "DiscreteUniform: lower bound `{}` must be ≤ upper bound `{}`",
                a, b
            ))
        } else {
            Ok(DiscreteUniform { a, b })
        }
    }
}

impl<R: Rng + ?Sized> Distribution<R> for DiscreteUniform {
    type Domain = i64;

    fn sample(&self, rng: &mut R) -> i64 {
        RandUniform::new_inclusive(self.a, self.b).sample(rng)
    }

    /// Returns $-\ln(b - a + 1)$, or $-\infty$ outside $[a, b]$.
    fn log_prob(&self, x: &i64) -> f64 {
        if *x < self.a || *x > self.b {
            return f64::NEG_INFINITY;
        }
        -((self.b - self.a + 1) as f64).ln()
    }

    fn is_discrete(&self) -> bool {
        true
    }
}

impl std::fmt::Display for DiscreteUniform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DiscreteUniform {{ a = {}, b = {} }}", self.a, self.b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn discrete_uniform_sample() {
        let mut rng = thread_rng();
        let dist = DiscreteUniform::new(1, 6).unwrap();
        println!("dist = {}", dist);
        let trials = 100_000;
        let mut total = 0.0f64;
        for _ in 0..trials {
            total += dist.sample(&mut rng) as f64;
        }
        let empirical_mean = total / trials as f64;
        // Mean = (a + b) / 2 = 3.5
        let expected_mean = 3.5f64;
        // Std = sqrt((b - a + 1)^2 - 1) / sqrt(12)
        let std = ((6.0f64 - 1.0 + 1.0).powi(2) - 1.0).sqrt() / 12.0f64.sqrt();
        let err = 5.0 * std / (trials as f64).sqrt();
        assert!((empirical_mean - expected_mean).abs() < err);
    }

    #[test]
    fn discrete_uniform_log_prob() {
        // DiscreteUniform(1, 4): 4 outcomes → log_prob = -ln(4)
        let dist = DiscreteUniform::new(1, 4).unwrap();
        let lp = <DiscreteUniform as Distribution<ThreadRng>>::log_prob(&dist, &2);
        assert!((lp - (-(4.0f64.ln()))).abs() < 1e-10);

        // out of range → NEG_INFINITY
        let lp_out = <DiscreteUniform as Distribution<ThreadRng>>::log_prob(&dist, &0);
        assert_eq!(lp_out, f64::NEG_INFINITY);

        assert!(<DiscreteUniform as Distribution<ThreadRng>>::is_discrete(
            &dist
        ));
    }

    #[test]
    fn discrete_uniform_display() {
        let dist = DiscreteUniform::new(1, 6).unwrap();
        let s = format!("{}", dist);
        assert!(s.contains("DiscreteUniform"), "missing type name: {}", s);
    }

    #[test]
    fn discrete_uniform_single() {
        // a == b is valid: only one outcome
        let dist = DiscreteUniform::new(3, 3).unwrap();
        let mut rng = thread_rng();
        assert_eq!(dist.sample(&mut rng), 3);
    }

    #[test]
    fn discrete_uniform_invalid() {
        assert!(DiscreteUniform::new(5, 3).is_err());
    }
}
