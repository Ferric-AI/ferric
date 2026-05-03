// Copyright 2022 The Ferric AI Project Developers

use rand::Rng;
use rand_distr::Distribution as Distribution2;
use rand_distr::WeightedIndex;

use crate::distributions::Distribution;

/// Weighted empirical distribution over a finite set of values.
///
/// Duplicate values are allowed; their weights are added when evaluating
/// `log_prob`.
pub struct Empirical<T> {
    values: Vec<T>,
    probs: Vec<f64>,
}

impl<T> Empirical<T> {
    /// Construct an empirical distribution from `(value, weight)` pairs.
    ///
    /// The weights are normalised automatically.
    ///
    /// # Errors
    ///
    /// Returns `Err` if no pairs are supplied, a weight is negative, or all
    /// weights are zero.
    pub fn new(weighted_values: Vec<(T, f64)>) -> Result<Empirical<T>, String> {
        if weighted_values.is_empty() {
            return Err("Empirical: weighted_values must not be empty".into());
        }
        if weighted_values.iter().any(|(_, w)| *w < 0.0) {
            return Err("Empirical: weights must be non-negative".into());
        }
        let total: f64 = weighted_values.iter().map(|(_, w)| *w).sum();
        if total <= 0.0 {
            return Err("Empirical: weights must sum to a positive value".into());
        }
        let (values, weights): (Vec<T>, Vec<f64>) = weighted_values.into_iter().unzip();
        Ok(Empirical {
            values,
            probs: weights.into_iter().map(|w| w / total).collect(),
        })
    }
}

impl<R, T> Distribution<R> for Empirical<T>
where
    R: Rng + ?Sized,
    T: Clone + PartialEq,
{
    type Domain = T;

    fn sample(&self, rng: &mut R) -> T {
        let index = WeightedIndex::new(&self.probs).unwrap().sample(rng);
        self.values[index].clone()
    }

    fn log_prob(&self, x: &T) -> f64 {
        let mut p = 0.0;
        for (value, prob) in self.values.iter().zip(self.probs.iter()) {
            if value == x {
                p += *prob;
            }
        }
        if p > 0.0 { p.ln() } else { f64::NEG_INFINITY }
    }

    fn is_discrete(&self) -> bool {
        true
    }
}

impl<T: std::fmt::Debug> std::fmt::Display for Empirical<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Empirical {{ values = {:?} }}", self.values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn empirical_samples_and_scores() {
        let dist = Empirical::new(vec![("a", 1.0), ("b", 3.0), ("a", 1.0)]).unwrap();
        let x = dist.sample(&mut thread_rng());
        assert!(["a", "b"].contains(&x));
        let lpa = <Empirical<&str> as Distribution<ThreadRng>>::log_prob(&dist, &"a");
        let lpb = <Empirical<&str> as Distribution<ThreadRng>>::log_prob(&dist, &"b");
        assert!((lpa - 0.4_f64.ln()).abs() < 1e-10);
        assert!((lpb - 0.6_f64.ln()).abs() < 1e-10);
        assert_eq!(
            <Empirical<&str> as Distribution<ThreadRng>>::log_prob(&dist, &"c"),
            f64::NEG_INFINITY
        );
        assert!(<Empirical<&str> as Distribution<ThreadRng>>::is_discrete(
            &dist
        ));
        assert!(format!("{}", dist).contains("Empirical"));
    }

    #[test]
    fn empirical_invalid() {
        assert!(Empirical::<i32>::new(vec![]).is_err());
        assert!(Empirical::new(vec![(1, -1.0)]).is_err());
        assert!(Empirical::new(vec![(1, 0.0)]).is_err());
    }

    #[test]
    fn empirical_integer_support() {
        let dist = Empirical::new(vec![(1, 2.0), (2, 1.0)]).unwrap();
        let x = dist.sample(&mut thread_rng());
        assert!([1, 2].contains(&x));
        assert!(
            (<Empirical<i32> as Distribution<ThreadRng>>::log_prob(&dist, &1)
                - (2.0_f64 / 3.0).ln())
            .abs()
                < 1e-10
        );
        assert!(format!("{}", dist).contains("Empirical"));
    }
}
