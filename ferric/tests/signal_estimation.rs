// Copyright 2022 The Ferric AI Project Developers
use ferric::make_model;

// Normal-Normal conjugate model.
// Prior: true_signal ~ Normal(0.0, 2.0)
// Likelihood: sensor_reading ~ Normal(true_signal, 1.0)
// Observation: sensor_reading = 2.5
// Analytical posterior: Normal(mean=2.0, std=sqrt(0.8)≈0.894)

#[test]
fn signal_estimation() {
    make_model! {
        mod signal_estimation;
        use ferric::distributions::Normal;

        let true_signal : f64 ~ Normal::new( 0.0, 2.0 );

        let sensor_reading : f64 ~ Normal::new( true_signal, 1.0 );

        observe sensor_reading;
        query true_signal;
    };

    let model = signal_estimation::Model {
        sensor_reading: 2.5,
    };
    let num_samples = 100000;

    let mut signal_vals = Vec::with_capacity(num_samples);
    let mut log_weights = Vec::with_capacity(num_samples);
    for ws in model.weighted_sample_iter().take(num_samples) {
        signal_vals.push(ws.sample.true_signal);
        log_weights.push(ws.log_weight);
    }

    let post_mean = ferric::weighted_mean(&signal_vals, &log_weights);
    let post_std = ferric::weighted_std(&signal_vals, &log_weights);

    println!("posterior mean = {} std = {}", post_mean, post_std);

    // Analytical: mean = 2.0, std = sqrt(0.8) ≈ 0.894
    // Allow generous margin since importance sampling variance is higher for continuous variables
    let mean_ans = 2.0f64;
    let std_ans = (0.8f64).sqrt();
    assert!(
        (post_mean - mean_ans).abs() < 0.1,
        "posterior mean {} not close to analytical {}",
        post_mean,
        mean_ans
    );
    assert!(
        (post_std - std_ans).abs() < 0.1,
        "posterior std {} not close to analytical {}",
        post_std,
        std_ans
    );
}
