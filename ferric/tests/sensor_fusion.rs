// Copyright 2022 The Ferric AI Project Developers
use ferric::make_model;

// Normal-Normal conjugate model with two observed continuous variables.
//
// Verifies that weighted_sample_iter correctly loops over ALL continuous
// observations and accumulates their log-likelihoods into log_weight.
//
// Prior:        true_mean ~ Normal(0.0, 3.0)
// Likelihoods:  sensor_1  ~ Normal(true_mean, 1.0)
//               sensor_2  ~ Normal(true_mean, 1.0)
// Observations: sensor_1 = 1.5, sensor_2 = 2.5
//
// Analytical posterior:
//   precision = 1/9 + 2 = 19/9
//   mean      = 36/19 ≈ 1.895
//   std       = sqrt(9/19) ≈ 0.688

#[test]
fn sensor_fusion() {
    make_model! {
        mod sensor_fusion;
        use ferric::distributions::Normal;

        let true_mean : f64 ~ Normal::new( 0.0, 3.0 );

        let sensor_1 : f64 ~ Normal::new( true_mean, 1.0 );

        let sensor_2 : f64 ~ Normal::new( true_mean, 1.0 );

        observe sensor_1;
        observe sensor_2;
        query true_mean;
    };

    let model = sensor_fusion::Model {
        sensor_1: 1.5,
        sensor_2: 2.5,
    };
    let num_samples = 100000;

    let mut mean_vals = Vec::with_capacity(num_samples);
    let mut log_weights = Vec::with_capacity(num_samples);
    for ws in model.weighted_sample_iter().take(num_samples) {
        mean_vals.push(ws.sample.true_mean);
        log_weights.push(ws.log_weight);
    }

    let post_mean = ferric::weighted_mean(&mean_vals, &log_weights);
    let post_std = ferric::weighted_std(&mean_vals, &log_weights);

    println!("posterior mean = {:.4} std = {:.4}", post_mean, post_std);

    let mean_ans = 36.0_f64 / 19.0; // ≈ 1.895
    let std_ans = (9.0_f64 / 19.0).sqrt(); // ≈ 0.688

    assert!(
        (post_mean - mean_ans).abs() < 0.1,
        "posterior mean {:.4} not close to analytical {:.4}",
        post_mean,
        mean_ans
    );
    assert!(
        (post_std - std_ans).abs() < 0.1,
        "posterior std {:.4} not close to analytical {:.4}",
        post_std,
        std_ans
    );
}
