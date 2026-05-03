// Copyright 2022 The Ferric AI Project Developers

use rand::Rng;
use rand_distr::Distribution as Distribution2;
use rand_distr::Pareto as Pareto2;

use crate::distributions::Distribution;

/// Pareto distribution over $[x_m, \infty)$.
///
/// The PDF is
///
/// $$p(x \mid x_m, \alpha) =
///     \frac{\alpha\, x_m^\alpha}{x^{\alpha+1}}$$
///
/// where $x_m > 0$ is the scale (minimum) parameter and $\alpha > 0$ is the
/// shape parameter.
///
/// See [Pareto distribution](https://en.wikipedia.org/wiki/Pareto_distribution)
/// on Wikipedia for further details.
///
/// # Examples
///
/// ```
/// use ferric::distributions::{Distribution, Pareto};
/// use rand::thread_rng;
///
/// let dist = Pareto::new(1.0, 2.0).unwrap();
/// let x: f64 = dist.sample(&mut thread_rng());
/// println!("sample = {:.4}", x);
/// ```
pub struct Pareto {
    x_m: f64,
    alpha: f64,
}

impl Pareto {
    /// Construct a Pareto distribution with scale `x_m` ($x_m$) and shape
    /// `alpha` ($\alpha$).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `x_m` or `alpha` is not strictly positive.
    pub fn new(x_m: f64, alpha: f64) -> Result<Pareto, String> {
        if x_m <= 0.0 {
            Err(format!(
                "Pareto: illegal scale `{}` should be greater than 0",
                x_m
            ))
        } else if alpha <= 0.0 {
            Err(format!(
                "Pareto: illegal shape `{}` should be greater than 0",
                alpha
            ))
        } else {
            Ok(Pareto { x_m, alpha })
        }
    }
}

impl<R: Rng + ?Sized> Distribution<R> for Pareto {
    type Domain = f64;

    fn sample(&self, rng: &mut R) -> f64 {
        Pareto2::new(self.x_m, self.alpha).unwrap().sample(rng)
    }

    /// Returns $\ln\alpha + \alpha\ln x_m - (\alpha+1)\ln x$, or
    /// $-\infty$ for $x < x_m$.
    fn log_prob(&self, x: &f64) -> f64 {
        if *x < self.x_m {
            return f64::NEG_INFINITY;
        }
        self.alpha.ln() + self.alpha * self.x_m.ln() - (self.alpha + 1.0) * x.ln()
    }

    fn is_discrete(&self) -> bool {
        false
    }
}

impl std::fmt::Display for Pareto {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Pareto {{ x_m = {}, alpha = {} }}", self.x_m, self.alpha)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn pareto_sample() {
        let mut rng = thread_rng();
        let x_m = 1.0f64;
        let alpha = 3.0f64;
        let dist = Pareto::new(x_m, alpha).unwrap();
        println!("dist = {}", dist);
        let trials = 100_000;
        let mut total = 0.0f64;
        for _ in 0..trials {
            total += dist.sample(&mut rng);
        }
        let empirical_mean = total / trials as f64;
        // Mean = alpha * x_m / (alpha - 1) for alpha > 1
        let expected_mean = alpha * x_m / (alpha - 1.0);
        let variance = x_m * x_m * alpha / ((alpha - 1.0).powi(2) * (alpha - 2.0));
        let std = variance.sqrt();
        let err = 5.0 * std / (trials as f64).sqrt();
        assert!((empirical_mean - expected_mean).abs() < err);
    }

    #[test]
    fn pareto_log_prob() {
        // Pareto(1, 1) at x=1: ln(1) + 1*ln(1) - 2*ln(1) = 0
        let dist = Pareto::new(1.0, 1.0).unwrap();
        let lp = <Pareto as Distribution<ThreadRng>>::log_prob(&dist, &1.0);
        assert!((lp - 0.0f64).abs() < 1e-10);

        // x < x_m → NEG_INFINITY
        let lp_low = <Pareto as Distribution<ThreadRng>>::log_prob(&dist, &0.5);
        assert_eq!(lp_low, f64::NEG_INFINITY);

        assert!(!<Pareto as Distribution<ThreadRng>>::is_discrete(&dist));
    }

    #[test]
    fn pareto_display() {
        let dist = Pareto::new(1.0, 2.0).unwrap();
        let s = format!("{}", dist);
        assert!(s.contains("Pareto"), "missing type name: {}", s);
    }

    #[test]
    #[should_panic]
    fn pareto_zero_scale() {
        Pareto::new(0.0, 1.0).unwrap();
    }

    #[test]
    #[should_panic]
    fn pareto_zero_shape() {
        Pareto::new(1.0, 0.0).unwrap();
    }
}
