// Copyright 2022 The Ferric AI Project Developers
use ferric::make_model;

// Network congestion model.
// Prior:        num_packets ~ Poisson(5.0)
// Likelihood:   congested   ~ Bernoulli(0.9 if num_packets > 8 else 0.1)
// Observation:  congested   = true
//
// Given the congestion observation the posterior mean of num_packets
// should rise above the prior mean of 5.

#[test]
fn congestion() {
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
    };

    let model = congestion::Model { congested: true };
    let num_samples = 200000;

    let mut packet_vals = Vec::with_capacity(num_samples);
    let mut log_weights = Vec::with_capacity(num_samples);
    for ws in model.weighted_sample_iter().take(num_samples) {
        packet_vals.push(ws.sample.num_packets as f64);
        log_weights.push(ws.log_weight);
    }
    let post_mean = ferric::weighted_mean(&packet_vals, &log_weights);
    println!("posterior num_packets mean = {}", post_mean);

    // Prior mean is 5.0; observing congestion pulls it above 5 and below ~10.
    assert!(
        post_mean > 5.5 && post_mean < 10.0,
        "posterior mean {} outside expected range (5.5, 10)",
        post_mean
    );
}
