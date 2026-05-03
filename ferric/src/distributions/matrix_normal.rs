// Copyright 2022 The Ferric AI Project Developers

use nalgebra::{Cholesky, DMatrix, Dyn};
use rand::Rng;
use rand_distr::Distribution as Distribution2;
use rand_distr::Normal as Normal2;

use crate::distributions::Distribution;

/// Matrix normal distribution over real matrices.
pub struct MatrixNormal {
    mean: DMatrix<f64>,
    row_chol: Cholesky<f64, Dyn>,
    col_chol: Cholesky<f64, Dyn>,
}

impl MatrixNormal {
    /// Construct a matrix normal distribution with mean matrix `mean`, row
    /// covariance `row_cov`, and column covariance `col_cov`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if covariance dimensions do not match the mean or either
    /// covariance is not symmetric positive-definite.
    pub fn new(
        mean: DMatrix<f64>,
        row_cov: DMatrix<f64>,
        col_cov: DMatrix<f64>,
    ) -> Result<MatrixNormal, String> {
        if row_cov.nrows() != mean.nrows() || row_cov.ncols() != mean.nrows() {
            return Err("MatrixNormal: row covariance must be nrows(mean) × nrows(mean)".into());
        }
        if col_cov.nrows() != mean.ncols() || col_cov.ncols() != mean.ncols() {
            return Err("MatrixNormal: column covariance must be ncols(mean) × ncols(mean)".into());
        }
        let row_chol = Cholesky::new(row_cov)
            .ok_or_else(|| "MatrixNormal: row covariance is not positive definite".to_string())?;
        let col_chol = Cholesky::new(col_cov).ok_or_else(|| {
            "MatrixNormal: column covariance is not positive definite".to_string()
        })?;
        Ok(MatrixNormal {
            mean,
            row_chol,
            col_chol,
        })
    }
}

impl<R: Rng + ?Sized> Distribution<R> for MatrixNormal {
    type Domain = DMatrix<f64>;

    fn sample(&self, rng: &mut R) -> DMatrix<f64> {
        let normal = Normal2::new(0.0, 1.0).unwrap();
        let z = DMatrix::from_fn(self.mean.nrows(), self.mean.ncols(), |_, _| {
            normal.sample(rng)
        });
        &self.mean + self.row_chol.l() * z * self.col_chol.l().transpose()
    }

    fn log_prob(&self, x: &DMatrix<f64>) -> f64 {
        if x.shape() != self.mean.shape() {
            return f64::NEG_INFINITY;
        }
        let n = self.mean.nrows() as f64;
        let p = self.mean.ncols() as f64;
        let diff = x - &self.mean;
        let row_solved = self.row_chol.solve(&diff);
        let trace_term = (row_solved.transpose() * diff * self.col_chol.inverse()).trace();
        let row_log_det = log_det_from_chol(&self.row_chol);
        let col_log_det = log_det_from_chol(&self.col_chol);
        -0.5 * (n * p * (2.0 * std::f64::consts::PI).ln()
            + p * row_log_det
            + n * col_log_det
            + trace_term)
    }

    fn is_discrete(&self) -> bool {
        false
    }
}

fn log_det_from_chol(chol: &Cholesky<f64, Dyn>) -> f64 {
    let l = chol.l();
    (0..l.nrows()).map(|i| l[(i, i)].ln()).sum::<f64>() * 2.0
}

impl std::fmt::Display for MatrixNormal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MatrixNormal {{ nrows = {}, ncols = {} }}",
            self.mean.nrows(),
            self.mean.ncols()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn matrix_normal_log_prob() {
        let dist = MatrixNormal::new(
            DMatrix::zeros(1, 1),
            DMatrix::identity(1, 1),
            DMatrix::identity(1, 1),
        )
        .unwrap();
        let lp = <MatrixNormal as Distribution<ThreadRng>>::log_prob(&dist, &DMatrix::zeros(1, 1));
        let expected = -0.5 * (2.0 * std::f64::consts::PI).ln();
        assert!((lp - expected).abs() < 1e-10);
        assert_eq!(
            <MatrixNormal as Distribution<ThreadRng>>::log_prob(&dist, &DMatrix::zeros(2, 1)),
            f64::NEG_INFINITY
        );
        assert!(!<MatrixNormal as Distribution<ThreadRng>>::is_discrete(
            &dist
        ));
    }

    #[test]
    fn matrix_normal_sample_and_display() {
        let dist = MatrixNormal::new(
            DMatrix::zeros(2, 3),
            DMatrix::identity(2, 2),
            DMatrix::identity(3, 3),
        )
        .unwrap();
        assert_eq!(dist.sample(&mut thread_rng()).shape(), (2, 3));
        assert!(format!("{}", dist).contains("MatrixNormal"));
    }

    #[test]
    fn matrix_normal_invalid() {
        assert!(
            MatrixNormal::new(
                DMatrix::zeros(2, 2),
                DMatrix::identity(3, 3),
                DMatrix::identity(2, 2)
            )
            .is_err()
        );
        assert!(
            MatrixNormal::new(
                DMatrix::zeros(2, 2),
                DMatrix::identity(2, 2),
                DMatrix::identity(3, 3)
            )
            .is_err()
        );
        assert!(
            MatrixNormal::new(
                DMatrix::zeros(2, 2),
                DMatrix::from_vec(2, 2, vec![1.0, 2.0, 2.0, 1.0]),
                DMatrix::identity(2, 2),
            )
            .is_err()
        );
        assert!(
            MatrixNormal::new(
                DMatrix::zeros(2, 2),
                DMatrix::identity(2, 2),
                DMatrix::from_vec(2, 2, vec![1.0, 2.0, 2.0, 1.0]),
            )
            .is_err()
        );
    }
}
