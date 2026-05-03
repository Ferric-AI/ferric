// Copyright 2022 The Ferric AI Project Developers

use rand::Rng;
use rand_distr::Distribution as Distribution2;
use rand_distr::Gamma as Gamma2;

use crate::distributions::Distribution;

/// Erlang distribution over non-negative real numbers.
///
/// This is a gamma distribution with integer shape `k` and rate `lambda`.
pub struct Erlang {
    k: u64,
    lambda: f64,
}

impl Erlang {
    /// Construct an Erlang distribution with integer shape `k` and rate
    /// `lambda`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `k == 0` or `lambda <= 0`.
    pub fn new(k: u64, lambda: f64) -> Result<Erlang, String> {
        if k == 0 {
            return Err("Erlang: k must be at least 1".into());
        }
        if lambda <= 0.0 {
            return Err(format!(
                "Erlang: illegal lambda `{}` should be greater than 0",
                lambda
            ));
        }
        Ok(Erlang { k, lambda })
    }
}

impl<R: Rng + ?Sized> Distribution<R> for Erlang {
    type Domain = f64;

    fn sample(&self, rng: &mut R) -> f64 {
        Gamma2::new(self.k as f64, 1.0 / self.lambda)
            .unwrap()
            .sample(rng)
    }

    fn log_prob(&self, x: &f64) -> f64 {
        if *x < 0.0 {
            return f64::NEG_INFINITY;
        }
        let k = self.k as f64;
        k * self.lambda.ln() + (k - 1.0) * x.ln() - self.lambda * x - libm::lgamma(k)
    }

    fn is_discrete(&self) -> bool {
        false
    }
}

impl std::fmt::Display for Erlang {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Erlang {{ k = {}, lambda = {} }}", self.k, self.lambda)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn erlang_log_prob() {
        let dist = Erlang::new(1, 2.0).unwrap();
        let lp = <Erlang as Distribution<ThreadRng>>::log_prob(&dist, &0.5);
        assert!((lp - (2.0_f64.ln() - 1.0)).abs() < 1e-10);
        assert_eq!(
            <Erlang as Distribution<ThreadRng>>::log_prob(&dist, &-1.0),
            f64::NEG_INFINITY
        );
        assert!(!<Erlang as Distribution<ThreadRng>>::is_discrete(&dist));
    }

    #[test]
    fn erlang_sample_and_display() {
        let dist = Erlang::new(3, 2.0).unwrap();
        assert!(dist.sample(&mut thread_rng()) >= 0.0);
        assert!(format!("{}", dist).contains("Erlang"));
    }

    #[test]
    fn erlang_invalid() {
        assert!(Erlang::new(0, 1.0).is_err());
        assert!(Erlang::new(1, 0.0).is_err());
    }
}
