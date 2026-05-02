// Copyright 2022 The Ferric AI Project Developers

use nalgebra::{Cholesky, DMatrix, DVector, Dyn};
use rand::Rng;
use rand_distr::Distribution as Distribution2;
use rand_distr::Normal as Normal2;

use crate::distributions::Distribution;

/// Multivariate normal (Gaussian) distribution over $\mathbb{R}^k$.
///
/// The PDF is
///
/// $$p(\mathbf{x} \mid \boldsymbol\mu, \boldsymbol\Sigma) =
///     \frac{1}{(2\pi)^{k/2} |\boldsymbol\Sigma|^{1/2}}
///     \exp\!\left(-\tfrac{1}{2}(\mathbf{x}-\boldsymbol\mu)^\top
///     \boldsymbol\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)\right)$$
///
/// where $\boldsymbol\mu \in \mathbb{R}^k$ is the mean vector and
/// $\boldsymbol\Sigma$ is a $k \times k$ symmetric positive-definite
/// covariance matrix.
///
/// Internally the constructor performs a Cholesky decomposition
/// $\boldsymbol\Sigma = LL^\top$, which is reused for both sampling and
/// log-probability evaluation.
///
/// # Examples
///
/// ```
/// use ferric::distributions::{Distribution, MultivariateNormal};
/// use nalgebra::{DMatrix, DVector};
/// use rand::thread_rng;
///
/// let mean = DVector::from_vec(vec![0.0, 0.0]);
/// let cov  = DMatrix::identity(2, 2);
/// let dist = MultivariateNormal::new(mean, cov).unwrap();
/// let x: DVector<f64> = dist.sample(&mut thread_rng());
/// println!("sample = {:?}", x.as_slice());
/// ```
pub struct MultivariateNormal {
    mean: DVector<f64>,
    chol: Cholesky<f64, Dyn>,
}

impl MultivariateNormal {
    /// Construct a MultivariateNormal with mean vector `mean` ($\boldsymbol\mu$)
    /// and covariance matrix `cov` ($\boldsymbol\Sigma$).
    ///
    /// # Errors
    ///
    /// Returns `Err` if the dimensions of `cov` do not match the length of
    /// `mean`, or if `cov` is not symmetric positive-definite.
    pub fn new(mean: DVector<f64>, cov: DMatrix<f64>) -> Result<MultivariateNormal, String> {
        let k = mean.len();
        if cov.nrows() != k || cov.ncols() != k {
            return Err(format!(
                "MultivariateNormal: covariance {}×{} must match mean length {}",
                cov.nrows(),
                cov.ncols(),
                k
            ));
        }
        match Cholesky::new(cov) {
            Some(chol) => Ok(MultivariateNormal { mean, chol }),
            None => Err("MultivariateNormal: covariance is not positive definite".to_string()),
        }
    }
}

impl<R: Rng + ?Sized> Distribution<R> for MultivariateNormal {
    type Domain = DVector<f64>;

    /// Draw one sample via $\mathbf{x} = \boldsymbol\mu + L\mathbf{z}$
    /// where $L$ is the lower Cholesky factor of $\boldsymbol\Sigma$ and
    /// $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I_k)$.
    fn sample(&self, rng: &mut R) -> DVector<f64> {
        let k = self.mean.len();
        let normal = Normal2::new(0.0, 1.0).unwrap();
        let z = DVector::from_fn(k, |_, _| normal.sample(rng));
        &self.mean + self.chol.l() * z
    }

    /// Returns
    /// $-\tfrac{k}{2}\ln(2\pi)
    ///   - \tfrac{1}{2}\ln|\boldsymbol\Sigma|
    ///   - \tfrac{1}{2}(\mathbf{x}-\boldsymbol\mu)^\top
    ///     \boldsymbol\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)$.
    ///
    /// Uses the Cholesky factor $L$ stored at construction time:
    /// $\ln|\boldsymbol\Sigma| = 2\sum_i \ln L_{ii}$ and the Mahalanobis
    /// term is computed via back-substitution.
    fn log_prob(&self, x: &DVector<f64>) -> f64 {
        let k = self.mean.len() as f64;
        let diff = x - &self.mean;
        let sol = self.chol.solve(&diff);
        let mahal_sq = diff.dot(&sol);
        let l = self.chol.l();
        let log_det: f64 = (0..l.nrows()).map(|i| l[(i, i)].ln()).sum::<f64>() * 2.0;
        -0.5 * (k * (2.0 * std::f64::consts::PI).ln() + log_det + mahal_sq)
    }

    fn is_discrete(&self) -> bool {
        false
    }
}

impl std::fmt::Display for MultivariateNormal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MultivariateNormal {{ k = {}, mean = {} }}",
            self.mean.len(),
            self.mean.transpose()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn mvn_sample() {
        let mut rng = thread_rng();
        let k = 3;
        let mean = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let cov = DMatrix::identity(k, k);
        let dist = MultivariateNormal::new(mean.clone(), cov).unwrap();
        println!("dist = {}", dist);
        let trials = 10_000;
        let mut sums = DVector::zeros(k);
        for _ in 0..trials {
            sums += dist.sample(&mut rng);
        }
        let empirical_mean = sums / trials as f64;
        // 5-sigma error for each component
        let err = 5.0 / (trials as f64).sqrt();
        for i in 0..k {
            let got = empirical_mean[i];
            let expected = mean[i];
            assert!(
                (got - expected).abs() < err,
                "component {} empirical mean {:.4} != expected {:.4}",
                i,
                got,
                expected
            );
        }
    }

    #[test]
    fn mvn_log_prob() {
        // Standard bivariate normal: log p(0,0 | N(0,I)) = -ln(2π)
        let dist =
            MultivariateNormal::new(DVector::from_vec(vec![0.0, 0.0]), DMatrix::identity(2, 2))
                .unwrap();
        let x = DVector::from_vec(vec![0.0, 0.0]);
        let lp = <MultivariateNormal as Distribution<ThreadRng>>::log_prob(&dist, &x);
        let expected = -(2.0 * std::f64::consts::PI).ln();
        assert!(
            (lp - expected).abs() < 1e-10,
            "log_prob {:.6} != expected {:.6}",
            lp,
            expected
        );
        assert!(!<MultivariateNormal as Distribution<ThreadRng>>::is_discrete(&dist));
    }

    #[test]
    #[should_panic]
    fn mvn_nrows_mismatch() {
        // nrows != mean length: 3×3 against a 2-element mean.
        MultivariateNormal::new(DVector::from_vec(vec![0.0, 0.0]), DMatrix::identity(3, 3))
            .unwrap();
    }

    #[test]
    #[should_panic]
    fn mvn_ncols_mismatch() {
        // nrows == mean length but ncols != mean length: 2×3 against a 2-element mean.
        MultivariateNormal::new(DVector::from_vec(vec![0.0, 0.0]), DMatrix::zeros(2, 3)).unwrap();
    }

    #[test]
    #[should_panic]
    fn mvn_not_positive_definite() {
        // [[1, 2], [2, 1]] has eigenvalues 3 and -1 — not positive definite.
        MultivariateNormal::new(
            DVector::from_vec(vec![0.0, 0.0]),
            DMatrix::from_vec(2, 2, vec![1.0, 2.0, 2.0, 1.0]),
        )
        .unwrap();
    }

    #[test]
    fn mvn_display() {
        let dist =
            MultivariateNormal::new(DVector::from_vec(vec![1.0, 2.0]), DMatrix::identity(2, 2))
                .unwrap();
        let s = format!("{}", dist);
        assert!(s.contains("MultivariateNormal"), "missing type name: {}", s);
        assert!(s.contains("k = 2"), "missing dimension: {}", s);
    }

    #[test]
    fn mvn_log_prob_off_center() {
        // mu = [1, 2], Sigma = 2*I_2, x = [2, 4]
        // diff = [1, 2]
        // Mahalanobis^2 = diff^T (2I)^{-1} diff = 0.5*(1+4) = 2.5
        // log|Sigma| = log(det(2*I_2)) = 2*log(2)
        // log_prob = -0.5*(2*log(2π) + 2*log(2) + 2.5)
        let dist = MultivariateNormal::new(
            DVector::from_vec(vec![1.0, 2.0]),
            DMatrix::from_diagonal_element(2, 2, 2.0),
        )
        .unwrap();
        let x = DVector::from_vec(vec![2.0, 4.0]);
        let lp = <MultivariateNormal as Distribution<ThreadRng>>::log_prob(&dist, &x);
        let expected = -0.5 * (2.0 * (2.0 * std::f64::consts::PI).ln() + 2.0 * 2.0_f64.ln() + 2.5);
        assert!(
            (lp - expected).abs() < 1e-10,
            "log_prob {:.6} != expected {:.6}",
            lp,
            expected
        );
    }
}
