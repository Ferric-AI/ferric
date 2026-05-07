// Copyright 2022 The Ferric AI Project Developers
// Rats weight model example from Gelfand et al. (1990), following the
// OpenBUGS example as closely as Ferric's syntax allows.
// https://chjackson.github.io/openbugsdoc/Examples/Rats.html
//
// Model overview:
// - Each rat has a straight-line growth curve.  alpha[rat] is that rat's
//   fitted weight at the centered age xbar, and beta[rat] is that rat's weekly
//   growth rate.
// - The model assumes each rat's alpha[rat] is drawn around a shared mean
//   alpha_c, and each beta[rat] is drawn around a shared mean beta_c.
// - The missing-data run uses the OpenBUGS missing-data pattern.  OpenBUGS
//   removes many observations but reports only rat 26's four missing weights,
//   so this example prints those same four predictions for comparison.
//
// Proposer overview:
// - initialize() copies the available observations for each rat.  It does not
//   use values removed by the missing-data experiment.
// - propose() handles every rat with at least two observed weights.  For each
//   such rat it picks one observed pair uniformly, solves the line through that
//   pair, and measures that line's residual variance on all observed weights
//   for the rat.
// - If the residual variance is nondegenerate, the proposer adds independent
//   Normal noise to that rat's alpha[rat] and beta[rat], using half the
//   residual variance as the noise variance.  If the variance is tiny, the
//   pair-implied alpha/beta are used exactly.
// - The proposed per-rat alpha/beta values then deterministically define
//   alpha_c, alpha_tau, beta_c, and beta_tau.  tau_c is also deterministic: it
//   is the precision corresponding to the residual standard deviation of
//   observed weights around the proposed per-rat lines.
// - Each experiment initializes its proposer from only the observations
//   available in that experiment.  The missing-data run does not peek at the
//   values removed by `missing_weights`.
// - Set FERRIC_DEBUG_IMPORTANCE to a positive integer to trace that many
//   importance samples in each experiment:
//     FERRIC_DEBUG_IMPORTANCE=1 cargo run -p ferric --example rats

use ferric::distributions::{Distribution, Normal};
use ferric::make_model;
use rand::Rng;

fn age(time: u64) -> f64 {
    match time {
        0 => 8.0,
        1 => 15.0,
        2 => 22.0,
        3 => 29.0,
        4 => 36.0,
        _ => panic!("rats example has five observation times"),
    }
}

fn precision_to_std_dev(tau: f64) -> f64 {
    1.0 / tau.max(1.0e-300).sqrt()
}

make_model! {
    name rats;
    use ferric::distributions::Gamma;
    use ferric::distributions::Normal;
    use super::age;
    use super::precision_to_std_dev;

    const num_rats : u64;
    const num_times : u64;
    const xbar : f64;

    let tau_c : f64 ~ Gamma::new(0.001, 1000.0);
    let sigma : f64 = precision_to_std_dev(tau_c);
    let alpha_c : f64 ~ Normal::new(0.0, 1000.0);
    let alpha_tau : f64 ~ Gamma::new(0.001, 1000.0);
    let alpha_std : f64 = precision_to_std_dev(alpha_tau);
    let beta_c : f64 ~ Normal::new(0.0, 1000.0);
    let beta_tau : f64 ~ Gamma::new(0.001, 1000.0);
    let beta_std : f64 = precision_to_std_dev(beta_tau);
    let alpha0 : f64 = alpha_c - xbar * beta_c;

    let alpha[rat of num_rats] : f64 ~ Normal::new(alpha_c, alpha_std);
    let beta[rat of num_rats] : f64 ~ Normal::new(beta_c, beta_std);
    let mu[rat of num_rats, time of num_times] : f64 =
        alpha[rat] + beta[rat] * (age(time) - xbar);
    let weight[rat of num_rats, time of num_times] : f64 ~ Normal::new(mu[rat, time], sigma);

    observe weight;
    query alpha_c;
    query beta_c;
    query tau_c;
    query alpha_tau;
    query beta_tau;
    query sigma;
    query alpha0;
    query weight;
}

fn observed_weights() -> Vec<Vec<Option<f64>>> {
    [
        [151.0, 199.0, 246.0, 283.0, 320.0],
        [145.0, 199.0, 249.0, 293.0, 354.0],
        [147.0, 214.0, 263.0, 312.0, 328.0],
        [155.0, 200.0, 237.0, 272.0, 297.0],
        [135.0, 188.0, 230.0, 280.0, 323.0],
        [159.0, 210.0, 252.0, 298.0, 331.0],
        [141.0, 189.0, 231.0, 275.0, 305.0],
        [159.0, 201.0, 248.0, 297.0, 338.0],
        [177.0, 236.0, 285.0, 350.0, 376.0],
        [134.0, 182.0, 220.0, 260.0, 296.0],
        [160.0, 208.0, 261.0, 313.0, 352.0],
        [143.0, 188.0, 220.0, 273.0, 314.0],
        [154.0, 200.0, 244.0, 289.0, 325.0],
        [171.0, 221.0, 270.0, 326.0, 358.0],
        [163.0, 216.0, 242.0, 281.0, 312.0],
        [160.0, 207.0, 248.0, 288.0, 324.0],
        [142.0, 187.0, 234.0, 280.0, 316.0],
        [156.0, 203.0, 243.0, 283.0, 317.0],
        [157.0, 212.0, 259.0, 307.0, 336.0],
        [152.0, 203.0, 246.0, 286.0, 321.0],
        [154.0, 205.0, 253.0, 298.0, 334.0],
        [139.0, 190.0, 225.0, 267.0, 302.0],
        [146.0, 191.0, 229.0, 272.0, 302.0],
        [157.0, 211.0, 250.0, 285.0, 323.0],
        [132.0, 185.0, 237.0, 286.0, 331.0],
        [160.0, 207.0, 257.0, 303.0, 345.0],
        [169.0, 216.0, 261.0, 295.0, 333.0],
        [157.0, 205.0, 248.0, 289.0, 316.0],
        [137.0, 180.0, 219.0, 258.0, 291.0],
        [153.0, 200.0, 244.0, 286.0, 324.0],
    ]
    .into_iter()
    .map(|row| row.into_iter().map(Some).collect())
    .collect()
}

fn missing_weights() -> Vec<Vec<Option<f64>>> {
    let mut weights = observed_weights();

    for row in weights.iter_mut().take(10).skip(5) {
        row[4] = None;
    }
    for row in weights.iter_mut().take(20).skip(10) {
        row[3] = None;
        row[4] = None;
    }
    for row in weights.iter_mut().take(25).skip(20) {
        row[2] = None;
        row[3] = None;
        row[4] = None;
    }
    for row in weights.iter_mut().take(30).skip(25) {
        row[1] = None;
        row[2] = None;
        row[3] = None;
        row[4] = None;
    }

    weights
}

#[derive(Clone, Default)]
struct RatsProposer {
    rats: Vec<Vec<(f64, f64)>>,
}

const MIN_PROPOSAL_STD_DEV: f64 = 1.0e-9;
const MIN_NOISE_VARIANCE: f64 = 1.0e-9;

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn std_dev(xs: &[f64], center: f64) -> f64 {
    let var = xs
        .iter()
        .map(|x| {
            let dx = x - center;
            dx * dx
        })
        .sum::<f64>()
        / (xs.len().saturating_sub(1).max(1) as f64);
    var.sqrt()
}

fn precision_from_std_dev(std_dev: f64) -> f64 {
    1.0 / std_dev.max(MIN_PROPOSAL_STD_DEV).powi(2)
}

fn pair_count(n: usize) -> usize {
    n * (n - 1) / 2
}

fn pair_from_index(n: usize, mut index: usize) -> (usize, usize) {
    for left in 0..n - 1 {
        let row_len = n - left - 1;
        if index < row_len {
            return (left, left + 1 + index);
        }
        index -= row_len;
    }
    unreachable!("pair index is always below n choose 2")
}

fn solve_pair(left: (f64, f64), right: (f64, f64)) -> (f64, f64) {
    let (x0, y0) = left;
    let (x1, y1) = right;
    let beta = (y1 - y0) / (x1 - x0);
    let alpha = y0 - beta * x0;
    (alpha, beta)
}

fn sample_normal(
    rng: &mut rand::rngs::ThreadRng,
    mean: f64,
    std_dev: f64,
    log_prob: &mut f64,
) -> f64 {
    let dist = Normal::new(mean, std_dev).unwrap();
    let x = dist.sample(rng);
    *log_prob += <Normal as Distribution<rand::rngs::ThreadRng>>::log_prob(&dist, &x);
    x
}

impl rats::Proposer<rand::rngs::ThreadRng> for RatsProposer {
    fn initialize(&mut self, data: &rats::ObservedData) {
        let xs = (0..data.num_times)
            .map(|time| age(time) - data.xbar)
            .collect::<Vec<_>>();

        self.rats = (0..data.num_rats as usize)
            .map(|rat| {
                data.weight[rat]
                    .iter()
                    .zip(&xs)
                    .filter_map(|(y, x)| y.map(|y| (*x, y)))
                    .collect::<Vec<_>>()
            })
            .collect();

        let eligible_rats = self.rats.iter().filter(|points| points.len() >= 2).count();
        let observed_weights = self.rats.iter().map(Vec::len).sum::<usize>();
        assert!(
            eligible_rats >= 2,
            "RatsProposer needs at least two rats with at least two observed weights"
        );

        println!(
            "RatsProposer initialized from {observed_weights} observed weights: {eligible_rats}/{} rats have at least two observations",
            self.rats.len()
        );
        println!(
            "  per eligible rat: choose one observed pair uniformly, solve alpha/beta, then add residual-variance noise when nondegenerate"
        );
        println!(
            "  alpha_c, alpha_tau, beta_c, beta_tau, and tau_c are deterministic functions of the proposed rat-level values"
        );
    }

    fn propose(&mut self, rng: &mut rand::rngs::ThreadRng) -> rats::Proposal {
        let mut log_prob = 0.0;
        let mut proposed_alphas = Vec::new();
        let mut proposed_betas = Vec::new();
        let mut sigma_residuals = Vec::new();
        let mut alpha = vec![None; self.rats.len()];
        let mut beta = vec![None; self.rats.len()];

        for (rat, points) in self.rats.iter().enumerate() {
            if points.len() < 2 {
                continue;
            }

            let choices = pair_count(points.len());
            let pair_index = rng.gen_range(0..choices);
            log_prob += -(choices as f64).ln();

            let (left, right) = pair_from_index(points.len(), pair_index);
            let (base_alpha, base_beta) = solve_pair(points[left], points[right]);
            let residual_variance = points
                .iter()
                .map(|(x, y)| {
                    let residual = y - (base_alpha + base_beta * x);
                    residual * residual
                })
                .sum::<f64>()
                / points.len() as f64;

            let (rat_alpha, rat_beta) = if residual_variance > MIN_NOISE_VARIANCE {
                let noise_std = (residual_variance / 2.0).sqrt();
                (
                    sample_normal(rng, base_alpha, noise_std, &mut log_prob),
                    sample_normal(rng, base_beta, noise_std, &mut log_prob),
                )
            } else {
                (base_alpha, base_beta)
            };

            alpha[rat] = Some(rat_alpha);
            beta[rat] = Some(rat_beta);
            proposed_alphas.push(rat_alpha);
            proposed_betas.push(rat_beta);
            sigma_residuals.extend(points.iter().map(|(x, y)| y - (rat_alpha + rat_beta * x)));
        }

        assert!(
            proposed_alphas.len() >= 2,
            "RatsProposer needs at least two proposed rats"
        );
        let alpha_c = mean(&proposed_alphas);
        let alpha_std = std_dev(&proposed_alphas, alpha_c);
        let alpha_tau = precision_from_std_dev(alpha_std);
        let beta_c = mean(&proposed_betas);
        let beta_std = std_dev(&proposed_betas, beta_c);
        let beta_tau = precision_from_std_dev(beta_std);
        let sigma = (sigma_residuals
            .iter()
            .map(|residual| residual * residual)
            .sum::<f64>()
            / sigma_residuals.len() as f64)
            .sqrt();
        let tau_c = precision_from_std_dev(sigma);

        let mut proposal = rats::Proposal::new(log_prob);
        proposal.tau_c = Some(tau_c);
        proposal.alpha_c = Some(alpha_c);
        proposal.alpha_tau = Some(alpha_tau);
        proposal.alpha = alpha;
        proposal.beta_c = Some(beta_c);
        proposal.beta_tau = Some(beta_tau);
        proposal.beta = beta;
        proposal
    }
}

fn print_summary(name: &str, values: &[f64], log_weights: &[f64], reference: Option<(f64, f64)>) {
    let mean = ferric::weighted_mean(values, log_weights);
    let sd = ferric::weighted_std(values, log_weights);
    if let Some((ref_mean, ref_sd)) = reference {
        println!(
            "{name:8} mean = {mean:8.3} sd = {sd:7.3}   OpenBUGS mean = {ref_mean:7.3} sd = {ref_sd:7.3}"
        );
    } else {
        println!("{name:8} mean = {mean:8.3} sd = {sd:7.3}");
    }
}

fn effective_sample_size(log_weights: &[f64]) -> f64 {
    let max_log_weight = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let mut sum_weight = 0.0;
    let mut sum_weight_squared = 0.0;
    for log_weight in log_weights {
        let weight = (log_weight - max_log_weight).exp();
        sum_weight += weight;
        sum_weight_squared += weight * weight;
    }

    if sum_weight_squared == 0.0 {
        0.0
    } else {
        sum_weight * sum_weight / sum_weight_squared
    }
}

fn print_proposal_coverage() {
    println!("Proposal coverage:");
    println!("  proposed: tau_c, alpha_c, alpha_tau, beta_c, beta_tau");
    println!("  proposed when a rat has at least two observations: alpha[rat], beta[rat]");
    println!("  prior-sampled when a rat has fewer than two observations: alpha[rat], beta[rat]");
    println!("  queried missing weights are sampled from the model, not proposed");
}

fn print_effective_sample_size(run: &RatsRun) {
    println!(
        "effective sample size: {:.1} / {}",
        effective_sample_size(&run.log_weights),
        run.log_weights.len()
    );
}

struct RatsRun {
    beta_c: Vec<f64>,
    sigma: Vec<f64>,
    alpha0: Vec<f64>,
    y_26_2: Vec<f64>,
    y_26_3: Vec<f64>,
    y_26_4: Vec<f64>,
    y_26_5: Vec<f64>,
    log_weights: Vec<f64>,
}

fn run_rats(
    weight: Vec<Vec<Option<f64>>>,
    num_samples: usize,
    proposer: RatsProposer,
    debug_samples: usize,
) -> RatsRun {
    let model = rats::Model {
        num_rats: 30,
        num_times: 5,
        xbar: 22.0,
        weight,
    };

    let mut beta_c = Vec::with_capacity(num_samples);
    let mut sigma = Vec::with_capacity(num_samples);
    let mut alpha0 = Vec::with_capacity(num_samples);
    let mut y_26_2 = Vec::with_capacity(num_samples);
    let mut y_26_3 = Vec::with_capacity(num_samples);
    let mut y_26_4 = Vec::with_capacity(num_samples);
    let mut y_26_5 = Vec::with_capacity(num_samples);
    let mut log_weights = Vec::with_capacity(num_samples);

    let sampler = if debug_samples > 0 {
        model.importance_sampler_debug(proposer, debug_samples)
    } else {
        model.importance_sampler(proposer)
    };

    for sample in sampler.take(num_samples) {
        beta_c.push(sample.sample.beta_c);
        sigma.push(sample.sample.sigma);
        alpha0.push(sample.sample.alpha0);
        y_26_2.push(sample.sample.weight[25][1]);
        y_26_3.push(sample.sample.weight[25][2]);
        y_26_4.push(sample.sample.weight[25][3]);
        y_26_5.push(sample.sample.weight[25][4]);
        log_weights.push(sample.log_weight);
    }

    RatsRun {
        beta_c,
        sigma,
        alpha0,
        y_26_2,
        y_26_3,
        y_26_4,
        y_26_5,
        log_weights,
    }
}

fn print_parameter_block(run: &RatsRun, references: bool) {
    if references {
        print_summary(
            "alpha0",
            &run.alpha0,
            &run.log_weights,
            Some((106.6, 3.666)),
        );
        print_summary(
            "beta.c",
            &run.beta_c,
            &run.log_weights,
            Some((6.186, 0.1088)),
        );
        print_summary("sigma", &run.sigma, &run.log_weights, Some((6.092, 0.4672)));
    } else {
        print_summary("alpha0", &run.alpha0, &run.log_weights, None);
        print_summary("beta.c", &run.beta_c, &run.log_weights, None);
        print_summary("sigma", &run.sigma, &run.log_weights, None);
    }
}

fn main() {
    let num_samples = 50000;
    let debug_samples = std::env::var("FERRIC_DEBUG_IMPORTANCE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    if debug_samples > 0 {
        println!("Tracing the first {debug_samples} importance sample(s) in each experiment");
    }

    println!(
        "Gelfand rats complete-data model, {} proposal-importance samples",
        num_samples
    );
    let complete = run_rats(
        observed_weights(),
        num_samples,
        RatsProposer::default(),
        debug_samples,
    );
    print_proposal_coverage();
    print_effective_sample_size(&complete);
    println!("Parameters in OpenBUGS order:");
    print_parameter_block(&complete, true);

    println!();
    println!(
        "Gelfand rats missing-data model, {} proposal-importance samples",
        num_samples
    );
    let missing = run_rats(
        missing_weights(),
        num_samples,
        RatsProposer::default(),
        debug_samples,
    );
    print_proposal_coverage();
    print_effective_sample_size(&missing);
    println!("Parameters in OpenBUGS order:");
    print_parameter_block(&missing, false);
    println!("Missing-data OpenBUGS order:");
    print_summary(
        "Y[26,2]",
        &missing.y_26_2,
        &missing.log_weights,
        Some((204.6, 8.689)),
    );
    print_summary(
        "Y[26,3]",
        &missing.y_26_3,
        &missing.log_weights,
        Some((250.2, 10.21)),
    );
    print_summary(
        "Y[26,4]",
        &missing.y_26_4,
        &missing.log_weights,
        Some((295.6, 12.5)),
    );
    print_summary(
        "Y[26,5]",
        &missing.y_26_5,
        &missing.log_weights,
        Some((341.2, 15.29)),
    );
    print_summary(
        "beta.c",
        &missing.beta_c,
        &missing.log_weights,
        Some((6.578, 0.1497)),
    );
}
