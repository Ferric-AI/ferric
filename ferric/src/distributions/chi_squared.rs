// Copyright 2022 The Ferric AI Project Developers

use rand::Rng;
use rand_distr::ChiSquared as ChiSquared2;
use rand_distr::Distribution as Distribution2;

use crate::distributions::Distribution;

/// Chi-squared distribution over the positive reals.
///
/// The PDF is
///
/// $$p(x \mid k) =
///     \frac{x^{k/2-1} e^{-x/2}}{2^{k/2}\,\Gamma(k/2)}$$
///
/// where $k > 0$ is the degrees-of-freedom parameter.  This is a special
/// case of the Gamma distribution: $\chi^2_k = \mathrm{Gamma}(k/2,\,2)$.
///
/// See [Chi-squared distribution](https://en.wikipedia.org/wiki/Chi-squared_distribution)
/// on Wikipedia for further details.
///
/// # Examples
///
/// ```
/// use ferric::distributions::{Distribution, ChiSquared};
/// use rand::thread_rng;
///
/// let dist = ChiSquared::new(3.0).unwrap();
/// let x: f64 = dist.sample(&mut thread_rng());
/// println!("sample = {:.4}", x);
/// ```
pub struct ChiSquared {
    k: f64,
}

impl ChiSquared {
    /// Construct a chi-squared distribution with `k` degrees of freedom.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `k` is not strictly positive.
    pub fn new(k: f64) -> Result<ChiSquared, String> {
        if k <= 0.0 {
            Err(format!(
                "ChiSquared: illegal degrees of freedom `{}` should be greater than 0",
                k
            ))
        } else {
            Ok(ChiSquared { k })
        }
    }
}

impl<R: Rng + ?Sized> Distribution<R> for ChiSquared {
    type Domain = f64;

    fn sample(&self, rng: &mut R) -> f64 {
        ChiSquared2::new(self.k).unwrap().sample(rng)
    }

    /// Returns $(k/2-1)\ln x - x/2 - (k/2)\ln 2 - \ln\Gamma(k/2)$, or
    /// $-\infty$ for $x \le 0$.
    fn log_prob(&self, x: &f64) -> f64 {
        if *x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let half_k = self.k / 2.0;
        (half_k - 1.0) * x.ln() - x / 2.0 - half_k * 2.0f64.ln() - libm::lgamma(half_k)
    }

    fn is_discrete(&self) -> bool {
        false
    }
}

impl std::fmt::Display for ChiSquared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ChiSquared {{ k = {} }}", self.k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn chi_squared_sample() {
        let mut rng = thread_rng();
        let k = 4.0f64;
        let dist = ChiSquared::new(k).unwrap();
        println!("dist = {}", dist);
        let trials = 100_000;
        let mut total = 0.0f64;
        for _ in 0..trials {
            total += dist.sample(&mut rng);
        }
        let empirical_mean = total / trials as f64;
        // Mean = k, Std = sqrt(2k)
        let std = (2.0 * k).sqrt();
        let err = 5.0 * std / (trials as f64).sqrt();
        assert!((empirical_mean - k).abs() < err);
    }

    #[test]
    fn chi_squared_log_prob() {
        // ChiSquared(2) = Exponential(1/2), at x=2: log_prob = -1 - ln(2)
        let dist = ChiSquared::new(2.0).unwrap();
        let lp = <ChiSquared as Distribution<ThreadRng>>::log_prob(&dist, &2.0);
        let expected = -1.0 - 2.0f64.ln();
        assert!((lp - expected).abs() < 1e-10);

        // x <= 0 → NEG_INFINITY
        let lp_zero = <ChiSquared as Distribution<ThreadRng>>::log_prob(&dist, &0.0);
        assert_eq!(lp_zero, f64::NEG_INFINITY);

        assert!(!<ChiSquared as Distribution<ThreadRng>>::is_discrete(&dist));
    }

    #[test]
    fn chi_squared_display() {
        let dist = ChiSquared::new(3.0).unwrap();
        let s = format!("{}", dist);
        assert!(s.contains("ChiSquared"), "missing type name: {}", s);
    }

    #[test]
    #[should_panic]
    fn chi_squared_zero_k() {
        ChiSquared::new(0.0).unwrap();
    }
}
