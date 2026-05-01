// Copyright 2022 The Ferric AI Project Developers

use rand_distr::Distribution as Distribution2;
use rand_distr::Poisson as Poisson2;

use crate::distributions::Distribution;
use rand::Rng;

/// Poisson distribution over non-negative integers.
///
/// The PMF is
///
/// $$P(X = k \mid \lambda) = \frac{\lambda^{k} e^{-\lambda}}{k!},
/// \quad k \in \{0, 1, 2, \ldots\}$$
///
/// where $\lambda > 0$ is the rate (expected number of events).
///
/// See [Poisson distribution](https://en.wikipedia.org/wiki/Poisson_distribution)
/// on Wikipedia for further details.
///
/// # Examples
///
/// ```
/// use ferric::distributions::{Distribution, Poisson};
/// use rand::thread_rng;
///
/// let dist = Poisson::new(2.5).unwrap();
/// let count: u64 = dist.sample(&mut thread_rng());
/// println!("count = {}", count);
/// ```
pub struct Poisson {
    /// Expected number of events (rate).
    rate: f64,
}

impl Poisson {
    /// Construct a Poisson distribution with the given `rate` ($\lambda$).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `rate` is not strictly positive.
    pub fn new(rate: f64) -> Result<Poisson, String> {
        if rate <= 0f64 {
            Err(format! {"Poisson: illegal rate `{}` should be greater than 0", rate})
        } else {
            Ok(Poisson { rate })
        }
    }
}

impl<R: Rng + ?Sized> Distribution<R> for Poisson {
    type Domain = u64;

    fn sample(&self, rng: &mut R) -> u64 {
        Poisson2::new(self.rate).unwrap().sample(rng) as u64
    }

    /// Returns $k \ln\lambda - \lambda - \ln(k!)$ where $k$ is `*x`.
    fn log_prob(&self, x: &u64) -> f64 {
        let k = *x as f64;
        let log_factorial: f64 = (1..=*x).map(|i| (i as f64).ln()).sum();
        k * self.rate.ln() - self.rate - log_factorial
    }

    fn is_discrete(&self) -> bool {
        true
    }
}

impl std::fmt::Display for Poisson {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Poisson {{ rate = {} }}", self.rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::Distribution;
    use rand::thread_rng;

    #[test]
    fn poisson_sample() {
        let mut rng = thread_rng();
        let rate = 2.7f64;
        let dist = Poisson::new(rate).unwrap();
        println!("dist = {}", dist);
        let mut total = 0u64;
        let trials = 10000;
        for _ in 0..trials {
            total += dist.sample(&mut rng);
        }
        let mean = (total as f64) / (trials as f64);
        let err = 5.0 * (rate / (trials as f64)).sqrt();
        println!("empirical mean is {} 5sigma error is {}", mean, err);
        assert!((mean - 2.7).abs() < err);
    }

    #[test]
    #[should_panic]
    fn poisson_too_low() {
        let _dist = Poisson::new(-0.01).unwrap();
    }

    #[test]
    fn poisson_log_prob_and_display() {
        let rate = 3.0f64;
        let dist = Poisson::new(rate).unwrap();
        // test log_prob for k = 0, 1, 5
        let lp0 = Distribution::<rand::rngs::ThreadRng>::log_prob(&dist, &0u64);
        // manual: 0 * ln(rate) - rate - ln(0!) == -rate
        assert_eq!(lp0, -rate);
        let lp1 = Distribution::<rand::rngs::ThreadRng>::log_prob(&dist, &1u64);
        let expected_lp1 = 1.0 * rate.ln() - rate - 0.0; // ln(1!) == 0.0
        assert!((lp1 - expected_lp1).abs() < 1e-12);
        let lp5 = Distribution::<rand::rngs::ThreadRng>::log_prob(&dist, &5u64);
        // compare to computed factorial log
        let k = 5u64;
        let log_fact: f64 = (1..=k).map(|i| (i as f64).ln()).sum();
        let expected_lp5 = (k as f64) * rate.ln() - rate - log_fact;
        assert!((lp5 - expected_lp5).abs() < 1e-12);
        // display and is_discrete
        assert!(format!("{}", dist).contains("Poisson"));
        assert!(Distribution::<rand::rngs::ThreadRng>::is_discrete(&dist));
        // zero rate should be rejected
        let bad = Poisson::new(0.0);
        assert!(bad.is_err());
    }
}
