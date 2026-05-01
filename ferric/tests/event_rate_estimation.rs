// Copyright 2022 The Ferric AI Project Developers
use ferric::make_model;

// Poisson-Normal mixed model.
// Prior: event_count ~ Poisson(3.0)
// Likelihood: noisy_reading ~ Normal(event_count as f64, 1.0)
// Observation: noisy_reading = 6.0
//
// The posterior mean should be pulled from the Poisson(3) prior toward
// the observation of 6.  For large Normal std relative to the Poisson
// rate, a rough expectation is a posterior mean between 4 and 6.

#[test]
fn event_rate_estimation() {
    make_model! {
        mod event_rate_estimation;
        use ferric::distributions::Normal;
        use ferric::distributions::Poisson;

        let event_count : u64 ~ Poisson::new( 3.0 );

        let noisy_reading : f64 ~ Normal::new( event_count as f64, 1.0 );

        observe noisy_reading;
        query event_count;
    };

    let model = event_rate_estimation::Model { noisy_reading: 6.0 };
    let num_samples = 200000;

    let mut count_vals = Vec::with_capacity(num_samples);
    let mut log_weights = Vec::with_capacity(num_samples);
    for ws in model.weighted_sample_iter().take(num_samples) {
        count_vals.push(ws.sample.event_count as f64);
        log_weights.push(ws.log_weight);
    }

    let post_mean = ferric::weighted_mean(&count_vals, &log_weights);
    println!("posterior event_count mean = {}", post_mean);

    // With Poisson(3) prior and observation 6.0, posterior mean lands between 4 and 6
    assert!(
        post_mean > 4.0 && post_mean < 6.0,
        "posterior mean {} outside expected range [4, 6]",
        post_mean
    );
}
