// Copyright 2022 The Ferric AI Project Developers

use rand_distr::Dirichlet as Dirichlet2;
use rand_distr::Distribution as Distribution2;

use crate::distributions::Distribution;
use rand::Rng;

/// Dirichlet distribution over the probability simplex $\Delta^{K-1}$.
///
/// A sample $\mathbf{x} = (x_1, \ldots, x_K)$ satisfies $x_i > 0$ and
/// $\sum_{i=1}^K x_i = 1$.  The PDF is
///
/// $$p(\mathbf{x} \mid \boldsymbol\alpha) =
///     \frac{\Gamma(\alpha_0)}{\prod_{i=1}^K \Gamma(\alpha_i)}
///     \prod_{i=1}^K x_i^{\alpha_i - 1},
///     \quad \alpha_0 = \textstyle\sum_i \alpha_i$$
///
/// where $\alpha_i > 0$ are the concentration parameters.  When all
/// $\alpha_i = 1$ the distribution is uniform over the simplex.
///
/// The Dirichlet is conjugate to the Multinomial and is commonly used as
/// a prior over categorical probabilities.
///
/// See [Dirichlet distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution)
/// on Wikipedia for further details.
///
/// # Examples
///
/// ```
/// use ferric::distributions::{Dirichlet, Distribution};
/// use rand::thread_rng;
///
/// let dist = Dirichlet::new(vec![1.0, 1.0, 1.0]).unwrap();
/// let theta: Vec<f64> = dist.sample(&mut thread_rng());
/// println!("theta = {:?}", theta);
/// ```
pub struct Dirichlet {
    alphas: Vec<f64>,
}

impl Dirichlet {
    /// Construct a Dirichlet distribution with concentration vector `alphas`
    /// ($\boldsymbol\alpha$).
    ///
    /// # Errors
    ///
    /// Returns `Err` if fewer than two concentrations are given or any
    /// concentration is not strictly positive.
    pub fn new(alphas: Vec<f64>) -> Result<Dirichlet, String> {
        if alphas.len() < 2 {
            return Err(
                "Dirichlet: concentration vector must contain at least 2 elements".to_string(),
            );
        }
        for &a in &alphas {
            if a <= 0.0 {
                return Err(format!(
                    "Dirichlet: all concentrations must be > 0, got {}",
                    a
                ));
            }
        }
        Ok(Dirichlet { alphas })
    }
}

impl<R: Rng + ?Sized> Distribution<R> for Dirichlet {
    type Domain = Vec<f64>;

    fn sample(&self, rng: &mut R) -> Vec<f64> {
        Dirichlet2::new(&self.alphas).unwrap().sample(rng)
    }

    /// Returns
    /// $\ln\Gamma(\alpha_0) - \sum_i \ln\Gamma(\alpha_i)
    ///  + \sum_i (\alpha_i - 1)\ln x_i$
    /// where $\alpha_0 = \sum_i \alpha_i$.
    ///
    /// Returns $-\infty$ if `x` has the wrong length, any component is
    /// outside $(0, 1)$, or the components do not sum to $1 \pm 10^{-9}$.
    fn log_prob(&self, x: &Vec<f64>) -> f64 {
        if x.len() != self.alphas.len() {
            return f64::NEG_INFINITY;
        }
        for &xi in x {
            if xi <= 0.0 || xi >= 1.0 {
                return f64::NEG_INFINITY;
            }
        }
        if (x.iter().sum::<f64>() - 1.0).abs() > 1e-9 {
            return f64::NEG_INFINITY;
        }
        let sum_alpha: f64 = self.alphas.iter().sum();
        let log_z: f64 =
            libm::lgamma(sum_alpha) - self.alphas.iter().map(|&a| libm::lgamma(a)).sum::<f64>();
        let log_kernel: f64 = self
            .alphas
            .iter()
            .zip(x.iter())
            .map(|(&a, &xi)| (a - 1.0) * xi.ln())
            .sum();
        log_z + log_kernel
    }

    fn is_discrete(&self) -> bool {
        false
    }
}

impl std::fmt::Display for Dirichlet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Dirichlet {{ alphas = {:?} }}", self.alphas)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn dirichlet_sample() {
        let mut rng = thread_rng();
        let alphas = vec![2.0f64, 3.0, 5.0];
        let dist = Dirichlet::new(alphas.clone()).unwrap();
        println!("dist = {}", dist);
        let trials = 10000;
        let mut sums = vec![0.0f64; 3];
        for _ in 0..trials {
            let x = dist.sample(&mut rng);
            // each sample must lie in the simplex
            assert!(x.iter().all(|&xi| xi > 0.0 && xi < 1.0));
            assert!((x.iter().sum::<f64>() - 1.0).abs() < 1e-10);
            for (s, xi) in sums.iter_mut().zip(x.iter()) {
                *s += xi;
            }
        }
        // Check empirical mean = alpha_i / sum(alpha)
        let sum_alpha: f64 = alphas.iter().sum();
        for (i, &alpha_i) in alphas.iter().enumerate() {
            let empirical_mean = sums[i] / (trials as f64);
            let expected_mean = alpha_i / sum_alpha;
            assert!(
                (empirical_mean - expected_mean).abs() < 0.02,
                "component {} empirical mean {} != expected {}",
                i,
                empirical_mean,
                expected_mean
            );
        }
    }

    #[test]
    fn dirichlet_log_prob() {
        // Dirichlet(1,1,1): log_prob at uniform point [1/3, 1/3, 1/3]
        // log p = lgamma(3) - 3*lgamma(1) + 0 = ln(2) - 0 = ln(2)
        let dist = Dirichlet::new(vec![1.0, 1.0, 1.0]).unwrap();
        let x = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        let lp = <Dirichlet as Distribution<ThreadRng>>::log_prob(&dist, &x);
        // lgamma(3)=ln(2); 3*lgamma(1)=0; kernel = 0*(-ln3)*3 = 0
        assert!((lp - 2.0f64.ln()).abs() < 1e-9);

        // wrong length
        let lp_short = <Dirichlet as Distribution<ThreadRng>>::log_prob(&dist, &vec![0.5, 0.5]);
        assert_eq!(lp_short, f64::NEG_INFINITY);

        // component out of (0,1)
        let lp_bad = <Dirichlet as Distribution<ThreadRng>>::log_prob(&dist, &vec![0.0, 0.5, 0.5]);
        assert_eq!(lp_bad, f64::NEG_INFINITY);

        // sum != 1
        let lp_sum = <Dirichlet as Distribution<ThreadRng>>::log_prob(&dist, &vec![0.4, 0.4, 0.4]);
        assert_eq!(lp_sum, f64::NEG_INFINITY);

        assert!(!<Dirichlet as Distribution<ThreadRng>>::is_discrete(&dist));
    }

    #[test]
    #[should_panic]
    fn dirichlet_too_short() {
        Dirichlet::new(vec![1.0]).unwrap();
    }

    #[test]
    #[should_panic]
    fn dirichlet_zero_concentration() {
        Dirichlet::new(vec![1.0, 0.0, 1.0]).unwrap();
    }
}
