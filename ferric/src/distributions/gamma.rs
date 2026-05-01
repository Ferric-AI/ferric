// Copyright 2022 The Ferric AI Project Developers

use rand_distr::Distribution as Distribution2;
use rand_distr::Gamma as Gamma2;

use crate::distributions::Distribution;
use rand::Rng;

/// Gamma distribution over positive reals.
///
/// The PDF is
///
/// $$p(x \mid k, \theta) =
///     \frac{x^{k-1} e^{-x/\theta}}{\Gamma(k)\,\theta^{k}},
///     \quad x > 0$$
///
/// where $k > 0$ is the shape parameter and $\theta > 0$ is the scale
/// parameter (mean $= k\theta$).
///
/// See [Gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution)
/// on Wikipedia for further details.
///
/// # Examples
///
/// ```
/// use ferric::distributions::{Distribution, Gamma};
/// use rand::thread_rng;
///
/// let dist = Gamma::new(2.0, 1.0).unwrap();
/// let x: f64 = dist.sample(&mut thread_rng());
/// println!("sample = {:.4}", x);
/// ```
pub struct Gamma {
    shape: f64,
    scale: f64,
}

impl Gamma {
    /// Construct a Gamma distribution with `shape` ($k$) and `scale` ($\theta$).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `shape` or `scale` is not strictly positive.
    pub fn new(shape: f64, scale: f64) -> Result<Gamma, String> {
        if shape <= 0.0 {
            Err(format!(
                "Gamma: illegal shape `{}` should be greater than 0",
                shape
            ))
        } else if scale <= 0.0 {
            Err(format!(
                "Gamma: illegal scale `{}` should be greater than 0",
                scale
            ))
        } else {
            Ok(Gamma { shape, scale })
        }
    }
}

impl<R: Rng + ?Sized> Distribution<R> for Gamma {
    type Domain = f64;

    fn sample(&self, rng: &mut R) -> f64 {
        Gamma2::new(self.shape, self.scale).unwrap().sample(rng)
    }

    /// Returns $(k-1)\ln x - x/\theta - k\ln\theta - \ln\Gamma(k)$.
    fn log_prob(&self, x: &f64) -> f64 {
        if *x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        (self.shape - 1.0) * x.ln()
            - x / self.scale
            - self.shape * self.scale.ln()
            - libm::lgamma(self.shape)
    }

    fn is_discrete(&self) -> bool {
        false
    }
}

impl std::fmt::Display for Gamma {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Gamma {{ shape = {}, scale = {} }}",
            self.shape, self.scale
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn gamma_sample() {
        let mut rng = thread_rng();
        let shape = 3.0f64;
        let scale = 2.0f64;
        let dist = Gamma::new(shape, scale).unwrap();
        println!("dist = {}", dist);
        let mut total = 0f64;
        let trials = 10000;
        for _ in 0..trials {
            total += dist.sample(&mut rng);
        }
        let empirical_mean = total / (trials as f64);
        let expected_mean = shape * scale;
        let expected_std = (shape * scale * scale).sqrt();
        let err = 5.0 * expected_std / (trials as f64).sqrt();
        assert!((empirical_mean - expected_mean).abs() < err);
    }

    #[test]
    fn gamma_log_prob() {
        // Gamma(1, 1) = Exp(1): log_prob(1) = -1
        let dist = Gamma::new(1.0, 1.0).unwrap();
        let lp = <Gamma as Distribution<ThreadRng>>::log_prob(&dist, &1.0);
        assert!((lp - (-1.0f64)).abs() < 1e-10);
        // outside support
        let lp_out = <Gamma as Distribution<ThreadRng>>::log_prob(&dist, &-1.0);
        assert_eq!(lp_out, f64::NEG_INFINITY);
        assert!(!<Gamma as Distribution<ThreadRng>>::is_discrete(&dist));
    }

    #[test]
    #[should_panic]
    fn gamma_zero_shape() {
        Gamma::new(0.0, 1.0).unwrap();
    }

    #[test]
    #[should_panic]
    fn gamma_zero_scale() {
        Gamma::new(1.0, 0.0).unwrap();
    }
}
