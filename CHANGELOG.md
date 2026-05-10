# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.2.1] - 2026-05-10

- Changed `Proposer` trait: `initialize(&mut self, data)` is replaced by an associated
  constructor `fn new(data: &ObservedData) -> Self`.
- `importance_sampler`, `importance_sample_iter`, and their debug variants no longer accept
  a pre-built proposer value; use turbofish syntax instead: `model.importance_sampler::<MyProposer>()`.

## [0.2.0] - 2026-05-04

- Added indexed random variables, including one-dimensional, two-dimensional,
  and higher-dimensional array-valued declarations such as
  `let x[row of n, col of m] : bool ~ Bernoulli::new(0.5);`.
- Added explicit named indices so dependency expressions can refer to the
  current index by name instead of relying on implicit `i`, `j`, `k` names.
- Switched indexed variables to zero-based ranges, so `row of n` means
  `row` takes values from `0` through `n - 1`.
- Added direct indexed dependency syntax such as `alive[person, time - 1]`.
- Added runtime dependency-loop detection for scalar and indexed variables.
- Changed the model header syntax from Rust-flavored `mod name;` to Ferric's
  own `name model_name;`.
- Added `const` declarations for model instantiation parameters and required
  explicit `max ...` annotations for stochastic integer variables used as
  index upper bounds.
- Added bounded-domain semantics for `max ...` so sampled values are drawn from
  the distribution restricted to values at or below the maximum.
- Added bounded likelihood normalization via
  `Distribution::log_cum_prob`.
- Cached bounded-domain normalization terms in generated worlds so each
  bounded random variable value computes its log CDF at most once per sampled
  world state.
- Added mask-aware observations for indexed stochastic variables using nested
  vectors with `Option<T>` leaves.
- Added support for deterministic dependencies over indexed random variables,
  including deterministic observed aggregates over arrays with stochastic
  cardinality.
- Added an unknown-urn-size marble-color example using indexed random
  variables, stochastic cardinality, noisy observations, and deterministic
  summary queries.
- Added a radar forward-simulation example with named indices, bounded counts,
  3D latent aircraft paths, real and false blips, and a deterministic
  per-timestep blip-set query.
- Added a Gelfand rats hierarchical growth-model example with the full
  observed weight table.
- Reworked the crate-level documentation landing page into a language guide
  covering model declarations, observations, queries, indexed variables,
  bounded domains, samplers, distributions, and worked examples.

## [0.1.4] - 2026-05-03

- Added and re-exported a broader well-known distribution set: beta-binomial,
  categorical, chi, chi-squared, Dirac, discrete uniform, empirical, Erlang,
  Fisher F, Frechet, Gumbel, half-normal, hypergeometric, inverse-gamma,
  inverse-Gaussian, Laplace, logistic, matrix normal, multivariate Student-t,
  negative binomial, Pareto, Rayleigh, triangular, Weibull, and Wishart.
- Added Dirichlet, multinomial, and multivariate normal examples covering
  vector-valued random variables and conjugate Bayesian updates.
- Added a deterministic dependency example with a queried deterministic value.
- Updated README and crate documentation for the expanded distribution surface.
- Removed references to the retired project domain.
- Fixed a bug in weighted sampling.

## [0.1.3] - 2026-04-30

- Added continuous random variables and likelihood weighted sampling
- Added beta, binomial, cauchy, exponential, gamma, geometric, log-normal,
  student-t, uniform, Dirichlet, multinomial, and multivariate normal
  distributions
- Updated code to rust version 1.95.0

## [0.1.2] - 2022-6-24

- Added discrete variables and rejection sampling
