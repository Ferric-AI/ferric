// Copyright 2022 The Ferric AI Project Developers
//
// Network congestion Bayesian network.
//
// Generative model:
//   num_packets      ~ Poisson(5.0)
//   congested        ~ Bernoulli(0.9 if num_packets > 8 else 0.1)
//
// Observation: congested = true
// Query:       posterior over num_packets
//
// With Poisson(5) prior the posterior mean of num_packets shifts upward
// (above 5) given that congestion was observed.

use ferric::make_model;
use std::time::Instant;

make_model! {
    mod congestion;
    use ferric::distributions::Bernoulli;
    use ferric::distributions::Poisson;

    let num_packets : u64 ~ Poisson::new( 5.0 );

    let congested : bool ~ Bernoulli::new(
        if num_packets > 8 { 0.9 } else { 0.1 }
    );

    observe congested;
    query num_packets;
}

fn main() {
    let model = congestion::Model { congested: true };
    let num_samples = 200000;
    let start = Instant::now();

    let mut packet_vals = Vec::with_capacity(num_samples);
    let mut log_weights = Vec::with_capacity(num_samples);

    for ws in model.weighted_sample_iter().take(num_samples) {
        packet_vals.push(ws.sample.num_packets as f64);
        log_weights.push(ws.log_weight);
    }

    let post_mean = ferric::weighted_mean(&packet_vals, &log_weights);
    let post_std = ferric::weighted_std(&packet_vals, &log_weights);

    println!(
        "posterior num_packets: mean = {:.4} std = {:.4}. \
         Elapsed {} millisec for {} samples",
        post_mean,
        post_std,
        start.elapsed().as_millis(),
        num_samples,
    );
}
