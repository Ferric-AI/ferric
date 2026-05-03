// Copyright 2022 The Ferric AI Project Developers

use rand::Rng;

use crate::distributions::{ChiSquared, Distribution};

/// Chi distribution over non-negative real numbers.
///
/// If $Y \sim \chi^2_k$, then $X = \sqrt{Y}$ has a chi distribution with
/// $k > 0$ degrees of freedom.
pub struct Chi {
    k: f64,
    chi_squared: ChiSquared,
}

impl Chi {
    /// Construct a chi distribution with `k` degrees of freedom.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `k <= 0`.
    pub fn new(k: f64) -> Result<Chi, String> {
        Ok(Chi {
            k,
            chi_squared: ChiSquared::new(k)?,
        })
    }
}

impl<R: Rng + ?Sized> Distribution<R> for Chi {
    type Domain = f64;

    fn sample(&self, rng: &mut R) -> f64 {
        self.chi_squared.sample(rng).sqrt()
    }

    fn log_prob(&self, x: &f64) -> f64 {
        if *x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        (1.0 - self.k / 2.0) * 2.0_f64.ln() - libm::lgamma(self.k / 2.0) + (self.k - 1.0) * x.ln()
            - 0.5 * x * x
    }

    fn is_discrete(&self) -> bool {
        false
    }
}

impl std::fmt::Display for Chi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Chi {{ k = {} }}", self.k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn chi_log_prob() {
        let dist = Chi::new(2.0).unwrap();
        let lp = <Chi as Distribution<ThreadRng>>::log_prob(&dist, &1.0);
        assert!((lp - (-0.5)).abs() < 1e-10);
        assert_eq!(
            <Chi as Distribution<ThreadRng>>::log_prob(&dist, &0.0),
            f64::NEG_INFINITY
        );
        assert!(!<Chi as Distribution<ThreadRng>>::is_discrete(&dist));
    }

    #[test]
    fn chi_sample_and_display() {
        let dist = Chi::new(3.0).unwrap();
        let x = dist.sample(&mut thread_rng());
        assert!(x >= 0.0);
        assert!(format!("{}", dist).contains("Chi"));
    }

    #[test]
    fn chi_invalid_k() {
        assert!(Chi::new(0.0).is_err());
    }
}
