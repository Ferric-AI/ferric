// Copyright 2022 The Ferric AI Project Developers
//
// Signal estimation via conjugate Normal-Normal model.
//
// Generative model:
//   true_signal    ~ Normal(0.0, 2.0)           (uncertain prior over the signal)
//   sensor_reading ~ Normal(true_signal, 1.0)   (noisy sensor measurement)
//
// Observation: sensor_reading = 2.5
//
// Analytical posterior (conjugate Normal-Normal):
//   posterior precision = 1/σ²_prior + 1/σ²_noise = 1/4 + 1/1 = 5/4
//   posterior mean      = (0/4 + 2.5/1) / (5/4) = 2.0
//   posterior std       = sqrt(4/5) ≈ 0.894

use ferric::make_model;
use std::time::Instant;

make_model! {
    mod signal_estimation;
    use ferric::distributions::Normal;

    let true_signal : f64 ~ Normal::new( 0.0, 2.0 );

    let sensor_reading : f64 ~ Normal::new( true_signal, 1.0 );

    observe sensor_reading;
    query true_signal;
}

fn main() {
    let model = signal_estimation::Model {
        sensor_reading: 2.5,
    };
    let num_samples = 100000;
    let start = Instant::now();

    let mut signal_vals = Vec::with_capacity(num_samples);
    let mut log_weights = Vec::with_capacity(num_samples);

    for ws in model.weighted_sample_iter().take(num_samples) {
        signal_vals.push(ws.sample.true_signal);
        log_weights.push(ws.log_weight);
    }

    let post_mean = ferric::weighted_mean(&signal_vals, &log_weights);
    let post_std = ferric::weighted_std(&signal_vals, &log_weights);

    println!(
        "posterior: true_signal mean = {:.4} std = {:.4} \
         (analytical: mean = 2.0000, std = 0.8944). \
         Elapsed {} millisec for {} samples",
        post_mean,
        post_std,
        start.elapsed().as_millis(),
        num_samples,
    );
}
