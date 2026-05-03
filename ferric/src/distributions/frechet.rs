// Copyright 2022 The Ferric AI Project Developers

use rand::Rng;
use rand_distr::Distribution as Distribution2;
use rand_distr::Frechet as Frechet2;

use crate::distributions::Distribution;

/// Fréchet (type-II extreme value) distribution.
///
/// The PDF is
///
/// $$p(x \mid m, s, \alpha) =
///     \frac{\alpha}{s}
///     \left(\frac{x-m}{s}\right)^{-\alpha-1}
///     \exp\!\left(-\left(\frac{x-m}{s}\right)^{-\alpha}\right)$$
///
/// where $m \in \mathbb{R}$ is the location parameter, $s > 0$ is the scale
/// parameter, $\alpha > 0$ is the shape parameter, and $x > m$.
///
/// See [Fréchet distribution](https://en.wikipedia.org/wiki/Fr%C3%A9chet_distribution)
/// on Wikipedia for further details.
///
/// # Examples
///
/// ```
/// use ferric::distributions::{Distribution, Frechet};
/// use rand::thread_rng;
///
/// let dist = Frechet::new(0.0, 1.0, 2.0).unwrap();
/// let x: f64 = dist.sample(&mut thread_rng());
/// println!("sample = {:.4}", x);
/// ```
pub struct Frechet {
    location: f64,
    scale: f64,
    shape: f64,
}

impl Frechet {
    /// Construct a Fréchet distribution with `location` ($m$), `scale` ($s$),
    /// and `shape` ($\alpha$).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `scale` or `shape` is not strictly positive.
    pub fn new(location: f64, scale: f64, shape: f64) -> Result<Frechet, String> {
        if scale <= 0.0 {
            Err(format!(
                "Frechet: illegal scale `{}` should be greater than 0",
                scale
            ))
        } else if shape <= 0.0 {
            Err(format!(
                "Frechet: illegal shape `{}` should be greater than 0",
                shape
            ))
        } else {
            Ok(Frechet {
                location,
                scale,
                shape,
            })
        }
    }
}

impl<R: Rng + ?Sized> Distribution<R> for Frechet {
    type Domain = f64;

    fn sample(&self, rng: &mut R) -> f64 {
        Frechet2::new(self.location, self.scale, self.shape)
            .unwrap()
            .sample(rng)
    }

    /// Returns $\ln\alpha - \ln s - (\alpha+1)\ln z - z^{-\alpha}$ where
    /// $z = (x-m)/s$, or $-\infty$ for $x \le m$.
    fn log_prob(&self, x: &f64) -> f64 {
        if *x <= self.location {
            return f64::NEG_INFINITY;
        }
        let z = (x - self.location) / self.scale;
        self.shape.ln() - self.scale.ln() - (self.shape + 1.0) * z.ln() - z.powf(-self.shape)
    }

    fn is_discrete(&self) -> bool {
        false
    }
}

impl std::fmt::Display for Frechet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Frechet {{ location = {}, scale = {}, shape = {} }}",
            self.location, self.scale, self.shape
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn frechet_sample() {
        let mut rng = thread_rng();
        // Use alpha > 2 so that variance exists; location=0, scale=1
        let dist = Frechet::new(0.0, 1.0, 3.0).unwrap();
        println!("dist = {}", dist);
        let trials = 100_000;
        let mut total = 0.0f64;
        for _ in 0..trials {
            total += dist.sample(&mut rng);
        }
        let empirical_mean = total / trials as f64;
        // Mean = Gamma(1 - 1/alpha) for alpha > 1
        let expected_mean = libm::tgamma(1.0 - 1.0 / 3.0);
        let variance = libm::tgamma(1.0 - 2.0 / 3.0) - expected_mean.powi(2);
        let std = variance.sqrt();
        let err = 5.0 * std / (trials as f64).sqrt();
        assert!((empirical_mean - expected_mean).abs() < err);
    }

    #[test]
    fn frechet_log_prob() {
        // log_prob finite for x > location
        let dist = Frechet::new(0.0, 1.0, 2.0).unwrap();
        let lp = <Frechet as Distribution<ThreadRng>>::log_prob(&dist, &1.0);
        assert!(lp.is_finite());

        // x <= location → NEG_INFINITY
        let lp_low = <Frechet as Distribution<ThreadRng>>::log_prob(&dist, &0.0);
        assert_eq!(lp_low, f64::NEG_INFINITY);

        assert!(!<Frechet as Distribution<ThreadRng>>::is_discrete(&dist));
    }

    #[test]
    fn frechet_display() {
        let dist = Frechet::new(0.0, 1.0, 2.0).unwrap();
        let s = format!("{}", dist);
        assert!(s.contains("Frechet"), "missing type name: {}", s);
    }

    #[test]
    #[should_panic]
    fn frechet_zero_scale() {
        Frechet::new(0.0, 0.0, 1.0).unwrap();
    }

    #[test]
    #[should_panic]
    fn frechet_zero_shape() {
        Frechet::new(0.0, 1.0, 0.0).unwrap();
    }
}
