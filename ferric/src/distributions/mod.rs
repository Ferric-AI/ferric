// Copyright 2022 The Ferric AI Project Developers

//! Probability distributions available to Ferric models.
//!
//! Every distribution implements [`Distribution`], which gives Ferric a common
//! way to draw prior samples and evaluate log probabilities for observed
//! values. Scalar distributions use native Rust numeric and boolean types,
//! vector distributions use `Vec` or `nalgebra::DVector<f64>`, and matrix
//! distributions use `nalgebra::DMatrix<f64>`.
mod bernoulli;
mod beta;
mod beta_binomial;
mod binomial;
mod categorical;
mod cauchy;
mod chi;
mod chi_squared;
mod dirac;
mod dirichlet;
mod discrete_uniform;
mod distribution;
mod empirical;
mod erlang;
mod exponential;
mod fisher_f;
mod frechet;
mod gamma;
mod geometric;
mod gumbel;
mod half_normal;
mod hypergeometric;
mod inverse_gamma;
mod inverse_gaussian;
mod laplace;
mod log_normal;
mod logistic;
mod matrix_normal;
mod multinomial;
mod multivariate_normal;
mod multivariate_student_t;
mod negative_binomial;
mod normal;
mod pareto;
mod poisson;
mod rayleigh;
mod student_t;
mod triangular;
mod uniform;
mod weibull;
mod wishart;

// Re-exports
pub use self::bernoulli::Bernoulli;
pub use self::beta::Beta;
pub use self::beta_binomial::BetaBinomial;
pub use self::binomial::Binomial;
pub use self::categorical::Categorical;
pub use self::cauchy::Cauchy;
pub use self::chi::Chi;
pub use self::chi_squared::ChiSquared;
pub use self::dirac::Dirac;
pub use self::dirichlet::Dirichlet;
pub use self::discrete_uniform::DiscreteUniform;
pub use self::distribution::Distribution;
pub use self::empirical::Empirical;
pub use self::erlang::Erlang;
pub use self::exponential::Exponential;
pub use self::fisher_f::FisherF;
pub use self::frechet::Frechet;
pub use self::gamma::Gamma;
pub use self::geometric::Geometric;
pub use self::gumbel::Gumbel;
pub use self::half_normal::HalfNormal;
pub use self::hypergeometric::Hypergeometric;
pub use self::inverse_gamma::InverseGamma;
pub use self::inverse_gaussian::InverseGaussian;
pub use self::laplace::Laplace;
pub use self::log_normal::LogNormal;
pub use self::logistic::Logistic;
pub use self::matrix_normal::MatrixNormal;
pub use self::multinomial::Multinomial;
pub use self::multivariate_normal::MultivariateNormal;
pub use self::multivariate_student_t::MultivariateStudentT;
pub use self::negative_binomial::NegativeBinomial;
pub use self::normal::Normal;
pub use self::pareto::Pareto;
pub use self::poisson::Poisson;
pub use self::rayleigh::Rayleigh;
pub use self::student_t::StudentT;
pub use self::triangular::Triangular;
pub use self::uniform::Uniform;
pub use self::weibull::Weibull;
pub use self::wishart::Wishart;
