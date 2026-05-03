// Copyright 2022 The Ferric AI Project Developers

use rand::Rng;
use rand_distr::Distribution as Distribution2;
use rand_distr::Exp;

use crate::distributions::Distribution;

/// Laplace (double-exponential) distribution over the reals.
///
/// The PDF is
///
/// $$p(x \mid \mu, b) = \frac{1}{2b}\exp\!\left(-\frac{|x-\mu|}{b}\right)$$
///
/// where $\mu \in \mathbb{R}$ is the location parameter and $b > 0$ is the
/// scale parameter.  The distribution is equivalent to the difference of two
/// independent $\mathrm{Exponential}(1/b)$ random variables shifted by $\mu$.
///
/// See [Laplace distribution](https://en.wikipedia.org/wiki/Laplace_distribution)
/// on Wikipedia for further details.
///
/// # Examples
///
/// ```
/// use ferric::distributions::{Distribution, Laplace};
/// use rand::thread_rng;
///
/// let dist = Laplace::new(0.0, 1.0).unwrap();
/// let x: f64 = dist.sample(&mut thread_rng());
/// println!("sample = {:.4}", x);
/// ```
pub struct Laplace {
    mu: f64,
    b: f64,
}

impl Laplace {
    /// Construct a Laplace distribution with location `mu` ($\mu$) and scale
    /// `b` ($b$).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `b` is not strictly positive.
    pub fn new(mu: f64, b: f64) -> Result<Laplace, String> {
        if b <= 0.0 {
            Err(format!(
                "Laplace: illegal scale `{}` should be greater than 0",
                b
            ))
        } else {
            Ok(Laplace { mu, b })
        }
    }
}

impl<R: Rng + ?Sized> Distribution<R> for Laplace {
    type Domain = f64;

    /// Draw a sample via the difference of two $\mathrm{Exp}(1)$ variates:
    /// $X = \mu + b(E_1 - E_2)$ where $E_1, E_2 \sim \mathrm{Exp}(1)$.
    fn sample(&self, rng: &mut R) -> f64 {
        let exp = Exp::new(1.0).unwrap();
        let e1 = exp.sample(rng);
        let e2 = exp.sample(rng);
        self.mu + self.b * (e1 - e2)
    }

    /// Returns $-\ln(2b) - |x - \mu| / b$.
    fn log_prob(&self, x: &f64) -> f64 {
        -(2.0 * self.b).ln() - (x - self.mu).abs() / self.b
    }

    fn is_discrete(&self) -> bool {
        false
    }
}

impl std::fmt::Display for Laplace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Laplace {{ mu = {}, b = {} }}", self.mu, self.b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn laplace_sample() {
        let mut rng = thread_rng();
        let mu = 2.0f64;
        let b = 1.5f64;
        let dist = Laplace::new(mu, b).unwrap();
        println!("dist = {}", dist);
        let trials = 100_000;
        let mut total = 0.0f64;
        for _ in 0..trials {
            total += dist.sample(&mut rng);
        }
        let empirical_mean = total / trials as f64;
        // Mean = mu, Std = b*sqrt(2)
        let err = 5.0 * b * 2.0f64.sqrt() / (trials as f64).sqrt();
        assert!((empirical_mean - mu).abs() < err);
    }

    #[test]
    fn laplace_log_prob() {
        // Laplace(0, 1) at x=0: log_prob = -ln(2)
        let dist = Laplace::new(0.0, 1.0).unwrap();
        let lp = <Laplace as Distribution<ThreadRng>>::log_prob(&dist, &0.0);
        assert!((lp - (-2.0f64.ln())).abs() < 1e-10);

        // Laplace(0, 1) at x=1: log_prob = -ln(2) - 1
        let lp2 = <Laplace as Distribution<ThreadRng>>::log_prob(&dist, &1.0);
        assert!((lp2 - (-2.0f64.ln() - 1.0)).abs() < 1e-10);

        // Symmetry: log_prob(x) = log_prob(-x)
        let lp3 = <Laplace as Distribution<ThreadRng>>::log_prob(&dist, &-1.0);
        assert!((lp3 - lp2).abs() < 1e-10);

        assert!(!<Laplace as Distribution<ThreadRng>>::is_discrete(&dist));
    }

    #[test]
    fn laplace_display() {
        let dist = Laplace::new(1.0, 2.0).unwrap();
        let s = format!("{}", dist);
        assert!(s.contains("Laplace"), "missing type name: {}", s);
    }

    #[test]
    #[should_panic]
    fn laplace_zero_scale() {
        Laplace::new(0.0, 0.0).unwrap();
    }

    #[test]
    #[should_panic]
    fn laplace_negative_scale() {
        Laplace::new(0.0, -1.0).unwrap();
    }
}
