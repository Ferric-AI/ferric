// Copyright 2022 The Ferric AI Project Developers

use rand_distr::Cauchy as Cauchy2;
use rand_distr::Distribution as Distribution2;

use crate::distributions::Distribution;
use rand::Rng;

/// Cauchy (Lorentz) distribution over the real line.
///
/// The PDF is
///
/// $$p(x \mid x_0, \gamma) =
///     \frac{1}{\pi\gamma\!\left[1 + \left(\tfrac{x-x_0}{\gamma}\right)^{2}\right]}$$
///
/// where $x_0 \in \mathbb{R}$ is the location (median) and $\gamma > 0$ is
/// the scale parameter.  The Cauchy distribution has no finite mean or variance.
///
/// See [Cauchy distribution](https://en.wikipedia.org/wiki/Cauchy_distribution)
/// on Wikipedia for further details.
///
/// # Examples
///
/// ```
/// use ferric::distributions::{Cauchy, Distribution};
/// use rand::thread_rng;
///
/// let dist = Cauchy::new(0.0, 1.0).unwrap();
/// let x: f64 = dist.sample(&mut thread_rng());
/// println!("sample = {:.4}", x);
/// ```
pub struct Cauchy {
    median: f64,
    scale: f64,
}

impl Cauchy {
    /// Construct a Cauchy distribution with `median` ($x_0$) and `scale` ($\gamma$).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `scale` is not strictly positive.
    pub fn new(median: f64, scale: f64) -> Result<Cauchy, String> {
        if scale <= 0.0 {
            Err(format!(
                "Cauchy: illegal scale `{}` should be greater than 0",
                scale
            ))
        } else {
            Ok(Cauchy { median, scale })
        }
    }
}

impl<R: Rng + ?Sized> Distribution<R> for Cauchy {
    type Domain = f64;

    fn sample(&self, rng: &mut R) -> f64 {
        Cauchy2::new(self.median, self.scale).unwrap().sample(rng)
    }

    /// Returns $-\ln\pi - \ln\gamma - \ln\!\left[1 + \left(\tfrac{x-x_0}{\gamma}\right)^{2}\right]$.
    fn log_prob(&self, x: &f64) -> f64 {
        let z = (x - self.median) / self.scale;
        -(std::f64::consts::PI).ln() - self.scale.ln() - (1.0 + z * z).ln()
    }

    fn is_discrete(&self) -> bool {
        false
    }
}

impl std::fmt::Display for Cauchy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Cauchy {{ median = {}, scale = {} }}",
            self.median, self.scale
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn cauchy_sample() {
        let mut rng = thread_rng();
        let dist = Cauchy::new(0.0, 1.0).unwrap();
        println!("dist = {}", dist);
        // Cauchy has no finite mean, but its median is the location parameter (0).
        // Roughly half the samples should be positive.
        let trials = 10000;
        let mut num_positive = 0usize;
        for _ in 0..trials {
            if dist.sample(&mut rng) > 0.0 {
                num_positive += 1;
            }
        }
        let frac_positive = num_positive as f64 / trials as f64;
        assert!((frac_positive - 0.5).abs() < 0.05);
    }

    #[test]
    fn cauchy_log_prob() {
        // Cauchy(0, 1): log_prob(0) = -ln(π)
        let dist = Cauchy::new(0.0, 1.0).unwrap();
        let lp = <Cauchy as Distribution<ThreadRng>>::log_prob(&dist, &0.0);
        assert!((lp - (-(std::f64::consts::PI).ln())).abs() < 1e-10);
        assert!(!<Cauchy as Distribution<ThreadRng>>::is_discrete(&dist));
    }

    #[test]
    #[should_panic]
    fn cauchy_zero_scale() {
        Cauchy::new(0.0, 0.0).unwrap();
    }

    #[test]
    #[should_panic]
    fn cauchy_negative_scale() {
        Cauchy::new(0.0, -1.0).unwrap();
    }
}
