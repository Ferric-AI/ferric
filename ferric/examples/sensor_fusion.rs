// Copyright 2022 The Ferric AI Project Developers
//
// Sensor fusion via a Normal-Normal conjugate model with two observed
// continuous variables.  Demonstrates that weighted sampling correctly
// accumulates the log-likelihood over *multiple* continuous observations.
//
// Generative model:
//   true_mean ~ Normal(0.0, 3.0)           (unknown signal mean)
//   sensor_1  ~ Normal(true_mean, 1.0)     (first noisy sensor)
//   sensor_2  ~ Normal(true_mean, 1.0)     (second noisy sensor)
//
// Observations: sensor_1 = 1.5, sensor_2 = 2.5
//
// Analytical posterior (conjugate Normal-Normal with two observations):
//   posterior precision = 1/σ²_prior + 2/σ²_noise = 1/9 + 2 = 19/9
//   posterior mean      = (0/9 + 1.5 + 2.5) / (19/9) = 36/19 ≈ 1.895
//   posterior std       = sqrt(9/19)             ≈ 0.688

use ferric::make_model;
use std::time::Instant;

make_model! {
    mod sensor_fusion;
    use ferric::distributions::Normal;

    let true_mean : f64 ~ Normal::new( 0.0, 3.0 );

    let sensor_1 : f64 ~ Normal::new( true_mean, 1.0 );

    let sensor_2 : f64 ~ Normal::new( true_mean, 1.0 );

    observe sensor_1;
    observe sensor_2;
    query true_mean;
}

fn main() {
    let model = sensor_fusion::Model {
        sensor_1: 1.5,
        sensor_2: 2.5,
    };
    let num_samples = 100000;
    let start = Instant::now();

    let mut mean_vals = Vec::with_capacity(num_samples);
    let mut log_weights = Vec::with_capacity(num_samples);

    for ws in model.weighted_sample_iter().take(num_samples) {
        mean_vals.push(ws.sample.true_mean);
        log_weights.push(ws.log_weight);
    }

    let post_mean = ferric::weighted_mean(&mean_vals, &log_weights);
    let post_std = ferric::weighted_std(&mean_vals, &log_weights);

    println!(
        "posterior: true_mean = {:.4} ± {:.4} \
         (analytical: {:.4} ± {:.4}). \
         Elapsed {} ms for {} samples",
        post_mean,
        post_std,
        36.0_f64 / 19.0,
        (9.0_f64 / 19.0).sqrt(),
        start.elapsed().as_millis(),
        num_samples,
    );
}
