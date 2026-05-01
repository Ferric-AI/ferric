// Copyright 2022 The Ferric AI Project Developers

use rand_distr::Distribution as Distribution2;
use rand_distr::StudentT as StudentT2;

use crate::distributions::Distribution;
use rand::Rng;

/// Student's $t$-distribution centred at zero.
///
/// The PDF is
///
/// $$p(x \mid \nu) =
///     \frac{\Gamma\!\left(\tfrac{\nu+1}{2}\right)}
///          {\sqrt{\nu\pi}\;\Gamma\!\left(\tfrac{\nu}{2}\right)}
///     \left(1 + \frac{x^{2}}{\nu}\right)^{-(\nu+1)/2}$$
///
/// where $\nu > 0$ is the degrees-of-freedom parameter.  For $\nu = 1$
/// this reduces to the standard Cauchy distribution; as $\nu \to \infty$
/// it converges to $\mathcal{N}(0, 1)$.
///
/// See [Student's t-distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution)
/// on Wikipedia for further details.
///
/// # Examples
///
/// ```
/// use ferric::distributions::{Distribution, StudentT};
/// use rand::thread_rng;
///
/// let dist = StudentT::new(3.0).unwrap();
/// let x: f64 = dist.sample(&mut thread_rng());
/// println!("sample = {:.4}", x);
/// ```
pub struct StudentT {
    df: f64,
}

impl StudentT {
    /// Construct a Student's $t$-distribution with `df` ($\nu$) degrees of freedom.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `df` is not strictly positive.
    pub fn new(df: f64) -> Result<StudentT, String> {
        if df <= 0.0 {
            Err(format!(
                "StudentT: illegal df `{}` should be greater than 0",
                df
            ))
        } else {
            Ok(StudentT { df })
        }
    }
}

impl<R: Rng + ?Sized> Distribution<R> for StudentT {
    type Domain = f64;

    fn sample(&self, rng: &mut R) -> f64 {
        StudentT2::new(self.df).unwrap().sample(rng)
    }

    /// Returns
    /// $\ln\Gamma\!\left(\tfrac{\nu+1}{2}\right)
    ///   - \ln\Gamma\!\left(\tfrac{\nu}{2}\right)
    ///   - \tfrac{1}{2}\ln(\nu\pi)
    ///   - \tfrac{\nu+1}{2}\ln\!\left(1 + x^2/\nu\right)$.
    fn log_prob(&self, x: &f64) -> f64 {
        let nu = self.df;
        libm::lgamma((nu + 1.0) / 2.0)
            - libm::lgamma(nu / 2.0)
            - 0.5 * (nu * std::f64::consts::PI).ln()
            - (nu + 1.0) / 2.0 * (1.0 + x * x / nu).ln()
    }

    fn is_discrete(&self) -> bool {
        false
    }
}

impl std::fmt::Display for StudentT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StudentT {{ df = {} }}", self.df)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn student_t_sample() {
        let mut rng = thread_rng();
        // df > 1 required for finite mean (= 0)
        let dist = StudentT::new(5.0).unwrap();
        println!("dist = {}", dist);
        let mut total = 0f64;
        let trials = 100000;
        for _ in 0..trials {
            total += dist.sample(&mut rng);
        }
        let empirical_mean = total / (trials as f64);
        // Expected mean = 0; std = sqrt(df/(df-2)) for df > 2
        let expected_std = (5.0f64 / 3.0).sqrt();
        let err = 5.0 * expected_std / (trials as f64).sqrt();
        assert!(empirical_mean.abs() < err);
    }

    #[test]
    fn student_t_log_prob() {
        // StudentT(1) = Cauchy(0,1): log_prob(0) = -ln(π)
        let dist = StudentT::new(1.0).unwrap();
        let lp = <StudentT as Distribution<ThreadRng>>::log_prob(&dist, &0.0);
        assert!((lp - (-(std::f64::consts::PI).ln())).abs() < 1e-10);
        assert!(!<StudentT as Distribution<ThreadRng>>::is_discrete(&dist));
    }

    #[test]
    #[should_panic]
    fn student_t_zero_df() {
        StudentT::new(0.0).unwrap();
    }

    #[test]
    #[should_panic]
    fn student_t_negative_df() {
        StudentT::new(-2.0).unwrap();
    }
}
