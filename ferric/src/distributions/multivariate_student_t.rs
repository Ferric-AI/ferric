// Copyright 2022 The Ferric AI Project Developers

use nalgebra::{Cholesky, DMatrix, DVector, Dyn};
use rand::Rng;
use rand_distr::ChiSquared as ChiSquared2;
use rand_distr::Distribution as Distribution2;
use rand_distr::Normal as Normal2;

use crate::distributions::Distribution;

/// Multivariate Student's t distribution over column vectors.
pub struct MultivariateStudentT {
    mean: DVector<f64>,
    chol: Cholesky<f64, Dyn>,
    df: f64,
}

impl MultivariateStudentT {
    /// Construct a multivariate Student's t distribution.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `df <= 0`, if `scale` does not match `mean`, or if
    /// `scale` is not symmetric positive-definite.
    pub fn new(
        mean: DVector<f64>,
        scale: DMatrix<f64>,
        df: f64,
    ) -> Result<MultivariateStudentT, String> {
        if df <= 0.0 {
            return Err(format!(
                "MultivariateStudentT: illegal df `{}` should be greater than 0",
                df
            ));
        }
        let k = mean.len();
        if scale.nrows() != k || scale.ncols() != k {
            return Err(format!(
                "MultivariateStudentT: scale {}×{} must match mean length {}",
                scale.nrows(),
                scale.ncols(),
                k
            ));
        }
        match Cholesky::new(scale.clone()) {
            Some(chol) => Ok(MultivariateStudentT { mean, chol, df }),
            None => Err("MultivariateStudentT: scale is not positive definite".to_string()),
        }
    }
}

impl<R: Rng + ?Sized> Distribution<R> for MultivariateStudentT {
    type Domain = DVector<f64>;

    fn sample(&self, rng: &mut R) -> DVector<f64> {
        let k = self.mean.len();
        let normal = Normal2::new(0.0, 1.0).unwrap();
        let z = DVector::from_fn(k, |_, _| normal.sample(rng));
        let g = ChiSquared2::new(self.df).unwrap().sample(rng);
        &self.mean + self.chol.l() * z * (self.df / g).sqrt()
    }

    fn log_prob(&self, x: &DVector<f64>) -> f64 {
        if x.len() != self.mean.len() {
            return f64::NEG_INFINITY;
        }
        let k = self.mean.len() as f64;
        let diff = x - &self.mean;
        let sol = self.chol.solve(&diff);
        let mahal_sq = diff.dot(&sol);
        let l = self.chol.l();
        let log_det: f64 = (0..l.nrows()).map(|i| l[(i, i)].ln()).sum::<f64>() * 2.0;
        libm::lgamma((self.df + k) / 2.0)
            - libm::lgamma(self.df / 2.0)
            - 0.5 * (k * (self.df * std::f64::consts::PI).ln() + log_det)
            - ((self.df + k) / 2.0) * (1.0 + mahal_sq / self.df).ln()
    }

    fn is_discrete(&self) -> bool {
        false
    }
}

impl std::fmt::Display for MultivariateStudentT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MultivariateStudentT {{ k = {}, df = {}, mean = {} }}",
            self.mean.len(),
            self.df,
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
    fn mv_student_t_log_prob() {
        let dist =
            MultivariateStudentT::new(DVector::from_vec(vec![0.0]), DMatrix::identity(1, 1), 1.0)
                .unwrap();
        let lp = <MultivariateStudentT as Distribution<ThreadRng>>::log_prob(
            &dist,
            &DVector::from_vec(vec![0.0]),
        );
        assert!((lp - (1.0 / std::f64::consts::PI).ln()).abs() < 1e-10);
        assert_eq!(
            <MultivariateStudentT as Distribution<ThreadRng>>::log_prob(
                &dist,
                &DVector::from_vec(vec![0.0, 1.0])
            ),
            f64::NEG_INFINITY
        );
        assert!(!<MultivariateStudentT as Distribution<ThreadRng>>::is_discrete(&dist));
    }

    #[test]
    fn mv_student_t_sample_and_display() {
        let dist = MultivariateStudentT::new(
            DVector::from_vec(vec![1.0, 2.0]),
            DMatrix::identity(2, 2),
            5.0,
        )
        .unwrap();
        assert_eq!(dist.sample(&mut thread_rng()).len(), 2);
        assert!(format!("{}", dist).contains("MultivariateStudentT"));
    }

    #[test]
    fn mv_student_t_invalid() {
        assert!(
            MultivariateStudentT::new(DVector::from_vec(vec![0.0]), DMatrix::identity(1, 1), 0.0)
                .is_err()
        );
        assert!(
            MultivariateStudentT::new(DVector::from_vec(vec![0.0]), DMatrix::identity(2, 2), 1.0)
                .is_err()
        );
        assert!(
            MultivariateStudentT::new(
                DVector::from_vec(vec![0.0, 0.0]),
                DMatrix::from_vec(2, 2, vec![1.0, 2.0, 2.0, 1.0]),
                1.0,
            )
            .is_err()
        );
    }
}
