// Copyright 2022 The Ferric AI Project Developers

use rand::Rng;
use rand_distr::Distribution as Distribution2;
use rand_distr::Gamma as Gamma2;

use crate::distributions::Distribution;

/// Inverse-gamma distribution over the positive reals.
///
/// The PDF is
///
/// $$p(x \mid \alpha, \beta) =
///     \frac{\beta^\alpha}{\Gamma(\alpha)}\,
///     x^{-\alpha-1}\exp\!\left(-\frac{\beta}{x}\right)$$
///
/// where $\alpha > 0$ is the shape parameter and $\beta > 0$ is the scale
/// parameter.  If $X \sim \mathrm{Gamma}(\alpha, 1/\beta)$ then
/// $1/X \sim \mathrm{InverseGamma}(\alpha, \beta)$.
///
/// See [Inverse-gamma distribution](https://en.wikipedia.org/wiki/Inverse-gamma_distribution)
/// on Wikipedia for further details.
///
/// # Examples
///
/// ```
/// use ferric::distributions::{Distribution, InverseGamma};
/// use rand::thread_rng;
///
/// let dist = InverseGamma::new(3.0, 2.0).unwrap();
/// let x: f64 = dist.sample(&mut thread_rng());
/// println!("sample = {:.4}", x);
/// ```
pub struct InverseGamma {
    alpha: f64,
    beta: f64,
}

impl InverseGamma {
    /// Construct an inverse-gamma distribution with shape `alpha` ($\alpha$)
    /// and scale `beta` ($\beta$).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `alpha` or `beta` is not strictly positive.
    pub fn new(alpha: f64, beta: f64) -> Result<InverseGamma, String> {
        if alpha <= 0.0 {
            Err(format!(
                "InverseGamma: illegal shape `{}` should be greater than 0",
                alpha
            ))
        } else if beta <= 0.0 {
            Err(format!(
                "InverseGamma: illegal scale `{}` should be greater than 0",
                beta
            ))
        } else {
            Ok(InverseGamma { alpha, beta })
        }
    }
}

impl<R: Rng + ?Sized> Distribution<R> for InverseGamma {
    type Domain = f64;

    /// Draw a sample as $1/\mathrm{Gamma}(\alpha,\,1/\beta)$.
    fn sample(&self, rng: &mut R) -> f64 {
        // Gamma2 uses shape/rate parameterisation: Gamma2::new(shape, scale=1/rate)
        let g = Gamma2::new(self.alpha, 1.0 / self.beta)
            .unwrap()
            .sample(rng);
        1.0 / g
    }

    /// Returns $\alpha\ln\beta - \ln\Gamma(\alpha) - (\alpha+1)\ln x - \beta/x$,
    /// or $-\infty$ for $x \le 0$.
    fn log_prob(&self, x: &f64) -> f64 {
        if *x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        self.alpha * self.beta.ln()
            - libm::lgamma(self.alpha)
            - (self.alpha + 1.0) * x.ln()
            - self.beta / x
    }

    fn is_discrete(&self) -> bool {
        false
    }
}

impl std::fmt::Display for InverseGamma {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "InverseGamma {{ alpha = {}, beta = {} }}",
            self.alpha, self.beta
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn inverse_gamma_sample() {
        let mut rng = thread_rng();
        let alpha = 3.0f64;
        let beta = 2.0f64;
        let dist = InverseGamma::new(alpha, beta).unwrap();
        println!("dist = {}", dist);
        let trials = 100_000;
        let mut total = 0.0f64;
        for _ in 0..trials {
            total += dist.sample(&mut rng);
        }
        let empirical_mean = total / trials as f64;
        // Mean = beta / (alpha - 1) for alpha > 1
        let expected_mean = beta / (alpha - 1.0);
        // Std = sqrt(beta^2 / ((alpha-1)^2 * (alpha-2)))
        let variance = beta * beta / ((alpha - 1.0).powi(2) * (alpha - 2.0));
        let std = variance.sqrt();
        let err = 5.0 * std / (trials as f64).sqrt();
        assert!((empirical_mean - expected_mean).abs() < err);
    }

    #[test]
    fn inverse_gamma_log_prob() {
        // InverseGamma(1, 1) at x=1: 1*ln(1) - lgamma(1) - 2*ln(1) - 1 = -1
        let dist = InverseGamma::new(1.0, 1.0).unwrap();
        let lp = <InverseGamma as Distribution<ThreadRng>>::log_prob(&dist, &1.0);
        assert!((lp - (-1.0f64)).abs() < 1e-10);

        // x <= 0 → NEG_INFINITY
        let lp_zero = <InverseGamma as Distribution<ThreadRng>>::log_prob(&dist, &0.0);
        assert_eq!(lp_zero, f64::NEG_INFINITY);

        assert!(!<InverseGamma as Distribution<ThreadRng>>::is_discrete(
            &dist
        ));
    }

    #[test]
    fn inverse_gamma_display() {
        let dist = InverseGamma::new(3.0, 2.0).unwrap();
        let s = format!("{}", dist);
        assert!(s.contains("InverseGamma"), "missing type name: {}", s);
    }

    #[test]
    #[should_panic]
    fn inverse_gamma_zero_alpha() {
        InverseGamma::new(0.0, 1.0).unwrap();
    }

    #[test]
    #[should_panic]
    fn inverse_gamma_zero_beta() {
        InverseGamma::new(1.0, 0.0).unwrap();
    }
}
