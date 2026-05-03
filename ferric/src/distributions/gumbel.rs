// Copyright 2022 The Ferric AI Project Developers

use rand::Rng;
use rand_distr::Distribution as Distribution2;
use rand_distr::Gumbel as Gumbel2;

use crate::distributions::Distribution;

/// Gumbel (type-I extreme value) distribution over the reals.
///
/// The PDF is
///
/// $$p(x \mid \mu, \beta) =
///     \frac{1}{\beta}\exp\!\left(-\frac{x-\mu}{\beta}
///     - e^{-(x-\mu)/\beta}\right)$$
///
/// where $\mu \in \mathbb{R}$ is the location parameter and $\beta > 0$ is
/// the scale parameter.
///
/// See [Gumbel distribution](https://en.wikipedia.org/wiki/Gumbel_distribution)
/// on Wikipedia for further details.
///
/// # Examples
///
/// ```
/// use ferric::distributions::{Distribution, Gumbel};
/// use rand::thread_rng;
///
/// let dist = Gumbel::new(0.0, 1.0).unwrap();
/// let x: f64 = dist.sample(&mut thread_rng());
/// println!("sample = {:.4}", x);
/// ```
pub struct Gumbel {
    mu: f64,
    beta: f64,
}

impl Gumbel {
    /// Construct a Gumbel distribution with location `mu` ($\mu$) and
    /// scale `beta` ($\beta$).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `beta` is not strictly positive.
    pub fn new(mu: f64, beta: f64) -> Result<Gumbel, String> {
        if beta <= 0.0 {
            Err(format!(
                "Gumbel: illegal scale `{}` should be greater than 0",
                beta
            ))
        } else {
            Ok(Gumbel { mu, beta })
        }
    }
}

impl<R: Rng + ?Sized> Distribution<R> for Gumbel {
    type Domain = f64;

    fn sample(&self, rng: &mut R) -> f64 {
        Gumbel2::new(self.mu, self.beta).unwrap().sample(rng)
    }

    /// Returns $-(x-\mu)/\beta - e^{-(x-\mu)/\beta} - \ln\beta$.
    fn log_prob(&self, x: &f64) -> f64 {
        let z = (x - self.mu) / self.beta;
        -z - (-z).exp() - self.beta.ln()
    }

    fn is_discrete(&self) -> bool {
        false
    }
}

impl std::fmt::Display for Gumbel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Gumbel {{ mu = {}, beta = {} }}", self.mu, self.beta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn gumbel_sample() {
        let mut rng = thread_rng();
        let mu = 1.0f64;
        let beta = 2.0f64;
        let dist = Gumbel::new(mu, beta).unwrap();
        println!("dist = {}", dist);
        let trials = 100_000;
        let mut total = 0.0f64;
        for _ in 0..trials {
            total += dist.sample(&mut rng);
        }
        let empirical_mean = total / trials as f64;
        // Mean = mu + beta * Euler–Mascheroni constant ≈ mu + beta * 0.5772
        let expected_mean = mu + beta * 0.577_215_664_9;
        // Std = beta * pi / sqrt(6)
        let std = beta * std::f64::consts::PI / 6.0f64.sqrt();
        let err = 5.0 * std / (trials as f64).sqrt();
        assert!((empirical_mean - expected_mean).abs() < err);
    }

    #[test]
    fn gumbel_log_prob() {
        // Gumbel(0, 1) at x=0: z=0, log_prob = 0 - 1 - 0 = -1
        let dist = Gumbel::new(0.0, 1.0).unwrap();
        let lp = <Gumbel as Distribution<ThreadRng>>::log_prob(&dist, &0.0);
        assert!((lp - (-1.0f64)).abs() < 1e-10);
        assert!(!<Gumbel as Distribution<ThreadRng>>::is_discrete(&dist));
    }

    #[test]
    fn gumbel_display() {
        let dist = Gumbel::new(1.0, 2.0).unwrap();
        let s = format!("{}", dist);
        assert!(s.contains("Gumbel"), "missing type name: {}", s);
    }

    #[test]
    #[should_panic]
    fn gumbel_zero_scale() {
        Gumbel::new(0.0, 0.0).unwrap();
    }

    #[test]
    #[should_panic]
    fn gumbel_negative_scale() {
        Gumbel::new(0.0, -1.0).unwrap();
    }
}
