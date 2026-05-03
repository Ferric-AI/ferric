// Copyright 2022 The Ferric AI Project Developers

use rand::Rng;
use rand_distr::Distribution as Distribution2;
use rand_distr::FisherF as FisherF2;

use crate::distributions::Distribution;

/// Fisher's F-distribution over the positive reals.
///
/// The PDF is
///
/// $$p(x \mid d_1, d_2) =
///     \frac{1}{B(d_1/2,\,d_2/2)}
///     \left(\frac{d_1}{d_2}\right)^{d_1/2}
///     x^{d_1/2-1}
///     \left(1+\frac{d_1}{d_2}x\right)^{-(d_1+d_2)/2}$$
///
/// where $d_1, d_2 > 0$ are the numerator and denominator degrees of freedom
/// and $B$ is the beta function.
///
/// See [F-distribution](https://en.wikipedia.org/wiki/F-distribution) on
/// Wikipedia for further details.
///
/// # Examples
///
/// ```
/// use ferric::distributions::{Distribution, FisherF};
/// use rand::thread_rng;
///
/// let dist = FisherF::new(5.0, 10.0).unwrap();
/// let x: f64 = dist.sample(&mut thread_rng());
/// println!("sample = {:.4}", x);
/// ```
pub struct FisherF {
    d1: f64,
    d2: f64,
}

impl FisherF {
    /// Construct an F-distribution with `d1` numerator and `d2` denominator
    /// degrees of freedom.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `d1` or `d2` is not strictly positive.
    pub fn new(d1: f64, d2: f64) -> Result<FisherF, String> {
        if d1 <= 0.0 {
            Err(format!(
                "FisherF: illegal d1 `{}` should be greater than 0",
                d1
            ))
        } else if d2 <= 0.0 {
            Err(format!(
                "FisherF: illegal d2 `{}` should be greater than 0",
                d2
            ))
        } else {
            Ok(FisherF { d1, d2 })
        }
    }
}

impl<R: Rng + ?Sized> Distribution<R> for FisherF {
    type Domain = f64;

    fn sample(&self, rng: &mut R) -> f64 {
        FisherF2::new(self.d1, self.d2).unwrap().sample(rng)
    }

    /// Returns the log-PDF evaluated at `x`.  Returns $-\infty$ for
    /// $x \le 0$.
    fn log_prob(&self, x: &f64) -> f64 {
        if *x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let d1 = self.d1;
        let d2 = self.d2;
        let log_beta =
            libm::lgamma(d1 / 2.0) + libm::lgamma(d2 / 2.0) - libm::lgamma((d1 + d2) / 2.0);
        (d1 / 2.0) * (d1 / d2).ln() + (d1 / 2.0 - 1.0) * x.ln()
            - ((d1 + d2) / 2.0) * (1.0 + d1 / d2 * x).ln()
            - log_beta
    }

    fn is_discrete(&self) -> bool {
        false
    }
}

impl std::fmt::Display for FisherF {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FisherF {{ d1 = {}, d2 = {} }}", self.d1, self.d2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn fisher_f_sample() {
        let mut rng = thread_rng();
        let d1 = 10.0f64;
        let d2 = 20.0f64;
        let dist = FisherF::new(d1, d2).unwrap();
        println!("dist = {}", dist);
        let trials = 100_000;
        let mut total = 0.0f64;
        for _ in 0..trials {
            total += dist.sample(&mut rng);
        }
        let empirical_mean = total / trials as f64;
        // Mean = d2 / (d2 - 2) for d2 > 2
        let expected_mean = d2 / (d2 - 2.0);
        let variance = 2.0 * d2 * d2 * (d1 + d2 - 2.0) / (d1 * (d2 - 2.0).powi(2) * (d2 - 4.0));
        let std = variance.sqrt();
        let err = 5.0 * std / (trials as f64).sqrt();
        assert!((empirical_mean - expected_mean).abs() < err);
    }

    #[test]
    fn fisher_f_log_prob() {
        // F(2, 2) at x=1: log-pdf should be finite
        let dist = FisherF::new(2.0, 2.0).unwrap();
        let lp = <FisherF as Distribution<ThreadRng>>::log_prob(&dist, &1.0);
        assert!(lp.is_finite());

        // x <= 0 → NEG_INFINITY
        let lp_zero = <FisherF as Distribution<ThreadRng>>::log_prob(&dist, &0.0);
        assert_eq!(lp_zero, f64::NEG_INFINITY);

        assert!(!<FisherF as Distribution<ThreadRng>>::is_discrete(&dist));
    }

    #[test]
    fn fisher_f_display() {
        let dist = FisherF::new(5.0, 10.0).unwrap();
        let s = format!("{}", dist);
        assert!(s.contains("FisherF"), "missing type name: {}", s);
    }

    #[test]
    #[should_panic]
    fn fisher_f_zero_d1() {
        FisherF::new(0.0, 1.0).unwrap();
    }

    #[test]
    #[should_panic]
    fn fisher_f_zero_d2() {
        FisherF::new(1.0, 0.0).unwrap();
    }
}
