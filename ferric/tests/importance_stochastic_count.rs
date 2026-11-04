// Copyright 2022 The Ferric AI Project Developers
//! User-proposal importance sampling on a model whose indexed latents have a
//! stochastic count (`true_blue[marble of n]` with `n ~ Poisson max ...`).
//!
//! The model is a small urn: few enough marbles that the posterior over `n`
//! can be computed exactly by enumeration, so every estimator can be checked
//! against the true answer.
use ferric::distributions::{Bernoulli, Distribution, Poisson};
use ferric::make_model;
use rand::Rng;

make_model! {
    name urn_is;
    use ferric::distributions::Bernoulli;
    use ferric::distributions::DiscreteUniform;
    use ferric::distributions::Poisson;

    const max_marbles : u64;
    const draws : u64;

    let n : u64 ~ Poisson::new(2.0) max max_marbles;
    let true_blue[marble of n] : bool ~ Bernoulli::new(0.5);
    let draw[d of draws] : i64 ~
        if n == 0 { DiscreteUniform::new(0, 0) } else { DiscreteUniform::new(0, n as i64 - 1) };
    let observed_blue[d of draws] : bool ~ Bernoulli::new(
        if n == 0 { 0.5 } else if true_blue[draw[d]] { 0.9 } else { 0.1 }
    );

    observe observed_blue;
    query n;
}

type R = rand::rngs::ThreadRng;

const MAX: u64 = 4;
const OBS: [bool; 3] = [true, true, false];
const NUM_SAMPLES: usize = 100_000;
/// Proposal probability that a marble is blue (the prior uses 0.5).
const Q_BLUE: f64 = 0.7;

fn model() -> urn_is::Model {
    urn_is::Model {
        max_marbles: MAX,
        draws: OBS.len() as u64,
        observed_blue: OBS.iter().map(|&b| Some(b)).collect(),
    }
}

fn bernoulli_log_prob(p: f64, value: bool) -> f64 {
    <Bernoulli as Distribution<R>>::log_prob(&Bernoulli::new(p).unwrap(), &value)
}

/// Exact P(n | observed_blue) by enumerating n, every colouring of the n
/// marbles, and (per draw, independently) which marble was drawn.
fn exact_posterior() -> Vec<f64> {
    let prior = Poisson::new(2.0).unwrap();
    let joint: Vec<f64> = (0..=MAX)
        .map(|n| {
            let prior_n = <Poisson as Distribution<R>>::log_prob(&prior, &n).exp();
            let likelihood = if n == 0 {
                0.5f64.powi(OBS.len() as i32)
            } else {
                (0..(1u32 << n))
                    .map(|mask| {
                        let mut l = 0.5f64.powi(n as i32);
                        for &obs in OBS.iter() {
                            let per_draw: f64 = (0..n)
                                .map(|k| {
                                    let p_blue = if (mask >> k) & 1 == 1 { 0.9 } else { 0.1 };
                                    if obs { p_blue } else { 1.0 - p_blue }
                                })
                                .sum();
                            l *= per_draw / n as f64;
                        }
                        l
                    })
                    .sum()
            };
            // The `max` truncation renormalises the prior; it cancels here.
            prior_n * likelihood
        })
        .collect();
    let z: f64 = joint.iter().sum();
    joint.iter().map(|j| j / z).collect()
}

/// Weighted estimate of E[n] and P(n = k), with the standard error of E[n]
/// approximated as weighted std / sqrt(ESS).
struct Estimate {
    mean: f64,
    std_err: f64,
    ess: f64,
    probs: Vec<f64>,
}

fn estimate(samples: impl Iterator<Item = (u64, f64)>) -> Estimate {
    let (ns, log_weights): (Vec<u64>, Vec<f64>) = samples.unzip();
    let values: Vec<f64> = ns.iter().map(|&n| n as f64).collect();
    let mean = ferric::weighted_mean(&values, &log_weights);
    let std = ferric::weighted_std(&values, &log_weights);
    let ess = ferric::effective_sample_size(&log_weights);
    let max_lw = log_weights
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let mut probs = vec![0.0; MAX as usize + 1];
    for (&n, &lw) in ns.iter().zip(&log_weights) {
        probs[n as usize] += (lw - max_lw).exp();
    }
    let z: f64 = probs.iter().sum();
    probs.iter_mut().for_each(|p| *p /= z);
    Estimate {
        mean,
        std_err: std / ess.sqrt(),
        ess,
        probs,
    }
}

fn assert_matches_exact(label: &str, est: &Estimate) {
    let exact = exact_posterior();
    let exact_mean: f64 = exact.iter().enumerate().map(|(k, p)| k as f64 * p).sum();
    println!(
        "{label}: E[n] = {:.4} +/- {:.4} (exact {exact_mean:.4}), ESS = {:.0}",
        est.mean, est.std_err, est.ess
    );
    assert!(
        (est.mean - exact_mean).abs() < 5.0 * est.std_err,
        "{label}: E[n] = {} but exact is {exact_mean} (std err {})",
        est.mean,
        est.std_err
    );
    for (k, (p_est, p_exact)) in est.probs.iter().zip(&exact).enumerate() {
        assert!(
            (p_est - p_exact).abs() < 0.02,
            "{label}: P(n = {k}) = {p_est} but exact is {p_exact}"
        );
    }
}

#[test]
fn likelihood_weighting_matches_exact_posterior() {
    let est = estimate(
        model()
            .weighted_sample_iter()
            .take(NUM_SAMPLES)
            .map(|s| (s.sample.n, s.log_weight)),
    );
    assert_matches_exact("likelihood weighting", &est);
}

/// Proposes every marble's colour up to `max_marbles`, leaving `n` to the
/// prior.  The proposer cannot know how many marbles exist.
struct FillToMax;

impl urn_is::Proposer<R> for FillToMax {
    fn new(_data: &urn_is::ObservedData) -> Self {
        FillToMax
    }

    fn propose(&mut self, rng: &mut R) -> urn_is::Proposal {
        let mut log_prob = 0.0;
        let mut cells = Vec::new();
        for _ in 0..MAX {
            let blue = rng.gen_bool(Q_BLUE);
            log_prob += bernoulli_log_prob(Q_BLUE, blue);
            cells.push(Some(blue));
        }
        let mut proposal = urn_is::Proposal::new(log_prob);
        proposal.true_blue = cells;
        proposal
    }
}

/// Documents today's behaviour: the weight loop skips proposed cells past the
/// count, but `apply_proposal` writes every `Some` cell into a cache sized by
/// the prior-drawn `n`, so any draw with `n < max_marbles` indexes out of
/// bounds.  The suspected silent bias never gets a chance to appear.
///
/// P(n = max) is about 0.1 here, so 50 samples all surviving is ~1e-50.
#[test]
#[should_panic(expected = "index out of bounds")]
fn proposing_cells_past_a_prior_drawn_count_panics() {
    for _ in model().importance_sampler::<FillToMax>().take(50) {}
}

/// Proposes the count too (uniform over 0..=max) and then exactly that many
/// colours.  The weights are correct, so this should match the exact answer.
struct ProposeCountAndCells;

impl urn_is::Proposer<R> for ProposeCountAndCells {
    fn new(_data: &urn_is::ObservedData) -> Self {
        ProposeCountAndCells
    }

    fn propose(&mut self, rng: &mut R) -> urn_is::Proposal {
        let n = rng.gen_range(0..=MAX);
        let mut log_prob = -((MAX + 1) as f64).ln();
        let mut cells = Vec::new();
        for _ in 0..n {
            let blue = rng.gen_bool(Q_BLUE);
            log_prob += bernoulli_log_prob(Q_BLUE, blue);
            cells.push(Some(blue));
        }
        let mut proposal = urn_is::Proposal::new(log_prob);
        proposal.n = Some(n);
        proposal.true_blue = cells;
        proposal
    }
}

/// Regression test: `apply_proposal` used to pin variables in `HashMap`
/// order inside the proc macro, which changed from one compile to the next.
/// When `true_blue` was pinned before `n`, its cache was sized by a prior draw
/// of `n` and a larger proposed count indexed out of bounds.  Pins now follow
/// declaration order, so the count is always pinned first.
#[test]
fn proposing_count_and_cells_matches_exact_posterior() {
    let est = estimate(
        model()
            .importance_sampler::<ProposeCountAndCells>()
            .take(NUM_SAMPLES)
            .map(|s| (s.sample.n, s.log_weight)),
    );
    assert_matches_exact("importance sampling, count proposed", &est);
}
