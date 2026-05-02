// Copyright 2022 The Ferric AI Project Developers
use ferric::make_model;

// Dirichlet-Multinomial conjugate model.
// Prior:       theta ~ Dirichlet([1.0, 1.0, 1.0])  (uniform over simplex)
// Likelihood:  counts ~ Multinomial(10, theta)
// Observation: counts = [5, 3, 2]
//
// The conjugate posterior is Dirichlet([1+5, 1+3, 1+2]) = Dirichlet([6, 4, 3]).
// Posterior means: [6/13, 4/13, 3/13] ≈ [0.462, 0.308, 0.231].
// MLE estimate (counts/n) = [0.5, 0.3, 0.2].
// The posterior mean is pulled between the prior (1/3 each) and the MLE.

#[test]
fn dirichlet_multinomial() {
    make_model! {
        mod dirichlet_multinomial;
        use ferric::distributions::Dirichlet;
        use ferric::distributions::Multinomial;

        let theta : Vec<f64> ~ Dirichlet::new( vec![1.0, 1.0, 1.0] );

        let counts : Vec<u64> ~ Multinomial::new( 10, theta.clone() );

        observe counts;
        query theta;
    };

    let model = dirichlet_multinomial::Model {
        counts: vec![5, 3, 2],
    };

    let num_samples = 100000;

    let mut theta0_vals = Vec::with_capacity(num_samples);
    let mut theta1_vals = Vec::with_capacity(num_samples);
    let mut theta2_vals = Vec::with_capacity(num_samples);
    let mut log_weights = Vec::with_capacity(num_samples);

    for ws in model.weighted_sample_iter().take(num_samples) {
        theta0_vals.push(ws.sample.theta[0]);
        theta1_vals.push(ws.sample.theta[1]);
        theta2_vals.push(ws.sample.theta[2]);
        log_weights.push(ws.log_weight);
    }

    // Conjugate posterior means: Dirichlet([6,4,3]) => [6/13, 4/13, 3/13]
    let mean0 = ferric::weighted_mean(&theta0_vals, &log_weights);
    let mean1 = ferric::weighted_mean(&theta1_vals, &log_weights);
    let mean2 = ferric::weighted_mean(&theta2_vals, &log_weights);

    println!(
        "posterior theta means = [{:.4}, {:.4}, {:.4}]",
        mean0, mean1, mean2
    );

    let expected0 = 6.0 / 13.0;
    let expected1 = 4.0 / 13.0;
    let expected2 = 3.0 / 13.0;
    let tol = 0.03;

    assert!(
        (mean0 - expected0).abs() < tol,
        "theta[0] posterior mean {:.4} != expected {:.4}",
        mean0,
        expected0
    );
    assert!(
        (mean1 - expected1).abs() < tol,
        "theta[1] posterior mean {:.4} != expected {:.4}",
        mean1,
        expected1
    );
    assert!(
        (mean2 - expected2).abs() < tol,
        "theta[2] posterior mean {:.4} != expected {:.4}",
        mean2,
        expected2
    );
}
