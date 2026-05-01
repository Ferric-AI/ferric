// Copyright 2022 The Ferric AI Project Developers

use rand_distr::Distribution as Distribution2;
use rand_distr::Normal as Normal2;

use crate::distributions::Distribution;
use rand::Rng;

/// Normal (Gaussian) distribution over the real line.
///
/// The PDF is
///
/// $$p(x \mid \mu, \sigma) =
///     \frac{1}{\sigma\sqrt{2\pi}}
///     \exp\!\left(-\frac{(x - \mu)^{2}}{2\sigma^{2}}\right)$$
///
/// where $\mu$ is the mean and $\sigma > 0$ is the standard deviation.
///
/// See [Normal distribution](https://en.wikipedia.org/wiki/Normal_distribution)
/// on Wikipedia for further details.
///
/// # Examples
///
/// ```
/// use ferric::distributions::{Distribution, Normal};
/// use rand::thread_rng;
///
/// let dist = Normal::new(0.0, 1.0).unwrap();
/// let x: f64 = dist.sample(&mut thread_rng());
/// println!("sample = {:.4}", x);
/// ```
pub struct Normal {
    mean: f64,
    std_dev: f64,
}

impl Normal {
    /// Construct a Normal distribution with the given `mean` ($\mu$) and
    /// `std_dev` ($\sigma$).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `std_dev` is not strictly positive.
    pub fn new(mean: f64, std_dev: f64) -> Result<Normal, String> {
        if std_dev <= 0f64 {
            Err(format! {"Normal: illegal std_dev `{}` should be greater than 0", std_dev})
        } else {
            Ok(Normal { mean, std_dev })
        }
    }
}

impl<R: Rng + ?Sized> Distribution<R> for Normal {
    type Domain = f64;

    fn sample(&self, rng: &mut R) -> f64 {
        Normal2::new(self.mean, self.std_dev).unwrap().sample(rng)
    }

    /// Returns the log-density
    /// $-\tfrac{1}{2}\!\left(\tfrac{x-\mu}{\sigma}\right)^{\!2}
    ///   - \ln\sigma - \tfrac{1}{2}\ln(2\pi)$.
    fn log_prob(&self, x: &f64) -> f64 {
        let z = (x - self.mean) / self.std_dev;
        -0.5 * z * z - self.std_dev.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
    }

    fn is_discrete(&self) -> bool {
        false
    }
}

impl std::fmt::Display for Normal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Normal {{ mean = {}, std_dev = {} }}",
            self.mean, self.std_dev
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn normal_sample() {
        let mut rng = thread_rng();
        let mean = 3.0f64;
        let std_dev = 2.0f64;
        let dist = Normal::new(mean, std_dev).unwrap();
        println!("dist = {}", dist);
        let mut total = 0f64;
        let trials = 10000;
        for _ in 0..trials {
            total += dist.sample(&mut rng);
        }
        let empirical_mean = total / (trials as f64);
        let err = 5.0 * std_dev / (trials as f64).sqrt();
        println!(
            "empirical mean is {} 5sigma error is {}",
            empirical_mean, err
        );
        assert!((empirical_mean - mean).abs() < err);
    }

    #[test]
    fn normal_log_prob() {
        let dist = Normal::new(0.0, 1.0).unwrap();
        // log p(0 | N(0,1)) = -0.5 * ln(2π)
        let lp = <Normal as Distribution<ThreadRng>>::log_prob(&dist, &0.0);
        assert!((lp - (-0.5 * (2.0 * std::f64::consts::PI).ln())).abs() < 1e-10);
        assert!(!<Normal as Distribution<ThreadRng>>::is_discrete(&dist));
    }

    #[test]
    #[should_panic]
    fn normal_zero_std() {
        let _dist = Normal::new(0.0, 0.0).unwrap();
    }

    #[test]
    #[should_panic]
    fn normal_negative_std() {
        let _dist = Normal::new(0.0, -1.0).unwrap();
    }
}
