// Copyright 2022 The Ferric AI Project Developers

use rand::Rng;

use crate::distributions::Distribution;

/// Deterministic point-mass distribution at a single value.
///
/// `Dirac::new(value)` always samples `value` and assigns log probability
/// `0` to that value and `-inf` to every other value.
pub struct Dirac<T> {
    value: T,
}

impl<T> Dirac<T> {
    /// Construct a point mass at `value`.
    pub fn new(value: T) -> Result<Dirac<T>, String> {
        Ok(Dirac { value })
    }
}

impl<R, T> Distribution<R> for Dirac<T>
where
    R: Rng + ?Sized,
    T: Clone + PartialEq,
{
    type Domain = T;

    fn sample(&self, _rng: &mut R) -> T {
        self.value.clone()
    }

    fn log_prob(&self, x: &T) -> f64 {
        if *x == self.value {
            0.0
        } else {
            f64::NEG_INFINITY
        }
    }

    fn is_discrete(&self) -> bool {
        true
    }
}

impl<T: std::fmt::Debug> std::fmt::Display for Dirac<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Dirac {{ value = {:?} }}", self.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn dirac_samples_and_scores() {
        let dist = Dirac::new(42).unwrap();
        assert_eq!(dist.sample(&mut thread_rng()), 42);
        assert_eq!(
            <Dirac<i32> as Distribution<ThreadRng>>::log_prob(&dist, &42),
            0.0
        );
        assert_eq!(
            <Dirac<i32> as Distribution<ThreadRng>>::log_prob(&dist, &7),
            f64::NEG_INFINITY
        );
        assert!(<Dirac<i32> as Distribution<ThreadRng>>::is_discrete(&dist));
        assert!(format!("{}", dist).contains("Dirac"));
    }
}
