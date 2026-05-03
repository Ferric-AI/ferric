// Copyright 2022 The Ferric AI Project Developers

use nalgebra::{Cholesky, DMatrix, Dyn};
use rand::Rng;
use rand_distr::ChiSquared as ChiSquared2;
use rand_distr::Distribution as Distribution2;
use rand_distr::Normal as Normal2;

use crate::distributions::Distribution;

/// Wishart distribution over symmetric positive-definite matrices.
pub struct Wishart {
    df: f64,
    scale: DMatrix<f64>,
    chol: Cholesky<f64, Dyn>,
}

impl Wishart {
    /// Construct a Wishart distribution with degrees of freedom `df` and
    /// positive-definite scale matrix `scale`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `scale` is not square, `df <= p - 1`, or `scale` is
    /// not symmetric positive-definite.
    pub fn new(df: f64, scale: DMatrix<f64>) -> Result<Wishart, String> {
        if scale.nrows() != scale.ncols() {
            return Err("Wishart: scale matrix must be square".into());
        }
        let p = scale.nrows();
        if df <= p as f64 - 1.0 {
            return Err(format!(
                "Wishart: df `{}` must be greater than p - 1 ({})",
                df,
                p - 1
            ));
        }
        match Cholesky::new(scale.clone()) {
            Some(chol) => Ok(Wishart { df, scale, chol }),
            None => Err("Wishart: scale matrix is not positive definite".to_string()),
        }
    }
}

impl<R: Rng + ?Sized> Distribution<R> for Wishart {
    type Domain = DMatrix<f64>;

    fn sample(&self, rng: &mut R) -> DMatrix<f64> {
        let p = self.scale.nrows();
        let normal = Normal2::new(0.0, 1.0).unwrap();
        let mut a = DMatrix::zeros(p, p);
        for i in 0..p {
            a[(i, i)] = ChiSquared2::new(self.df - i as f64)
                .unwrap()
                .sample(rng)
                .sqrt();
            for j in 0..i {
                a[(i, j)] = normal.sample(rng);
            }
        }
        let la = self.chol.l() * a;
        &la * la.transpose()
    }

    fn log_prob(&self, x: &DMatrix<f64>) -> f64 {
        let p = self.scale.nrows();
        if x.nrows() != p || x.ncols() != p {
            return f64::NEG_INFINITY;
        }
        let Some(x_chol) = Cholesky::new(x.clone()) else {
            return f64::NEG_INFINITY;
        };
        let log_det_x = log_det_from_chol(&x_chol);
        let log_det_scale = log_det_from_chol(&self.chol);
        let trace_term = self.chol.solve(x).trace();
        ((self.df - p as f64 - 1.0) / 2.0) * log_det_x
            - 0.5 * trace_term
            - (self.df * p as f64 / 2.0) * 2.0_f64.ln()
            - (self.df / 2.0) * log_det_scale
            - multivariate_lgamma(p, self.df / 2.0)
    }

    fn is_discrete(&self) -> bool {
        false
    }
}

fn log_det_from_chol(chol: &Cholesky<f64, Dyn>) -> f64 {
    let l = chol.l();
    (0..l.nrows()).map(|i| l[(i, i)].ln()).sum::<f64>() * 2.0
}

fn multivariate_lgamma(p: usize, a: f64) -> f64 {
    (p as f64 * (p as f64 - 1.0) / 4.0) * std::f64::consts::PI.ln()
        + (1..=p)
            .map(|j| libm::lgamma(a + (1.0 - j as f64) / 2.0))
            .sum::<f64>()
}

impl std::fmt::Display for Wishart {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Wishart {{ p = {}, df = {} }}",
            self.scale.nrows(),
            self.df
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn wishart_log_prob() {
        let dist = Wishart::new(3.0, DMatrix::identity(1, 1)).unwrap();
        let lp = <Wishart as Distribution<ThreadRng>>::log_prob(
            &dist,
            &DMatrix::from_element(1, 1, 2.0),
        );
        let chi_squared = crate::distributions::ChiSquared::new(3.0).unwrap();
        let expected = <crate::distributions::ChiSquared as Distribution<ThreadRng>>::log_prob(
            &chi_squared,
            &2.0,
        );
        assert!((lp - expected).abs() < 1e-10);
        assert_eq!(
            <Wishart as Distribution<ThreadRng>>::log_prob(&dist, &DMatrix::identity(2, 2)),
            f64::NEG_INFINITY
        );
        assert_eq!(
            <Wishart as Distribution<ThreadRng>>::log_prob(
                &dist,
                &DMatrix::from_element(1, 1, -1.0)
            ),
            f64::NEG_INFINITY
        );
        assert!(!<Wishart as Distribution<ThreadRng>>::is_discrete(&dist));
    }

    #[test]
    fn wishart_sample_and_display() {
        let dist = Wishart::new(5.0, DMatrix::identity(2, 2)).unwrap();
        let x = dist.sample(&mut thread_rng());
        assert_eq!(x.shape(), (2, 2));
        assert!(Cholesky::new(x).is_some());
        assert!(format!("{}", dist).contains("Wishart"));
    }

    #[test]
    fn wishart_invalid() {
        assert!(Wishart::new(1.0, DMatrix::zeros(2, 3)).is_err());
        assert!(Wishart::new(1.0, DMatrix::identity(2, 2)).is_err());
        assert!(Wishart::new(3.0, DMatrix::from_vec(2, 2, vec![1.0, 2.0, 2.0, 1.0])).is_err());
    }
}
