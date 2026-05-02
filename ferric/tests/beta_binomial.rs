// Copyright 2022 The Ferric AI Project Developers
use ferric::make_model;

// Beta-Binomial conjugate model.
//
// Prior:       theta ~ Beta(1.0, 1.0)  (uniform on [0, 1])
// Likelihood:  successes ~ Binomial(10, theta)
// Observation: successes = 7
//
// Conjugate posterior: Beta(1+7, 1+3) = Beta(8, 4)
// Posterior mean of theta: 8/12 = 2/3 ≈ 0.667

#[test]
fn beta_binomial() {
    make_model! {
        mod beta_binomial;
        use ferric::distributions::Beta;
        use ferric::distributions::Binomial;

        let theta : f64 ~ Beta::new( 1.0, 1.0 );

        let successes : u64 ~ Binomial::new( 10, theta );

        observe successes;
        query theta;
    };

    let model = beta_binomial::Model { successes: 7 };

    let num_samples = 100000;

    let mut theta_vals = Vec::with_capacity(num_samples);
    let mut log_weights = Vec::with_capacity(num_samples);

    for ws in model.weighted_sample_iter().take(num_samples) {
        theta_vals.push(ws.sample.theta);
        log_weights.push(ws.log_weight);
    }

    // Conjugate posterior mean: Beta(8, 4) => 8/12 = 2/3
    let mean = ferric::weighted_mean(&theta_vals, &log_weights);
    println!("posterior theta mean = {:.4}", mean);

    let expected = 8.0 / 12.0;
    let tol = 0.03;
    assert!(
        (mean - expected).abs() < tol,
        "theta posterior mean {:.4} != expected {:.4}",
        mean,
        expected
    );
}
