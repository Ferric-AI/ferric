// Copyright 2022 The Ferric AI Project Developers
//
// Event rate estimation via a Poisson-Normal mixed model.
//
// Generative model:
//   event_count   ~ Poisson(3.0)                      (true count of events)
//   noisy_reading ~ Normal(event_count as f64, 1.0)   (noisy real-valued sensor)
//
// Observation: noisy_reading = 6.0
//
// With the Poisson(3.0) prior and a reading far above the prior mean, the
// posterior over event_count shifts toward higher values.

use ferric::make_model;
use std::time::Instant;

make_model! {
    mod event_rate_estimation;
    use ferric::distributions::Normal;
    use ferric::distributions::Poisson;

    let event_count : u64 ~ Poisson::new( 3.0 );

    let noisy_reading : f64 ~ Normal::new( event_count as f64, 1.0 );

    observe noisy_reading;
    query event_count;
}

fn main() {
    let observed_reading = 6.0f64;
    let model = event_rate_estimation::Model {
        noisy_reading: observed_reading,
    };
    let num_samples = 100000;
    let start = Instant::now();

    let mut count_vals = Vec::with_capacity(num_samples);
    let mut log_weights = Vec::with_capacity(num_samples);

    for ws in model.weighted_sample_iter().take(num_samples) {
        count_vals.push(ws.sample.event_count as f64);
        log_weights.push(ws.log_weight);
    }

    let post_mean = ferric::weighted_mean(&count_vals, &log_weights);
    let post_std = ferric::weighted_std(&count_vals, &log_weights);

    println!(
        "observed_reading = {}  \
         posterior event_count: mean = {:.4} std = {:.4}. \
         Elapsed {} millisec for {} samples",
        observed_reading,
        post_mean,
        post_std,
        start.elapsed().as_millis(),
        num_samples,
    );
}
