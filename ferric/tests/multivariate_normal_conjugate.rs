// Copyright 2022 The Ferric AI Project Developers
use ferric::make_model;
use nalgebra::DVector;

// Normal-Normal conjugate model (multivariate).
//
// Prior:        mu ~ MultivariateNormal([0, 0], I₂)
// Likelihood:   x  ~ MultivariateNormal(mu, 0.1·I₂)
// Observation:  x  = [1, 2]
//
// Conjugate posterior (Normal-Normal with known covariance):
//   Prior precision:       Λ₀ = I₂
//   Likelihood precision:  Λ  = (0.1·I₂)⁻¹ = 10·I₂
//   Posterior precision:   Λₙ = Λ₀ + Λ = 11·I₂
//   Posterior covariance:  Σₙ = (1/11)·I₂
//   Posterior mean:        μₙ = Σₙ·(Λ₀·μ₀ + Λ·x)
//                             = (1/11)·(I₂·[0,0] + 10·I₂·[1,2])
//                             = [10/11, 20/11] ≈ [0.9091, 1.8182]

#[test]
fn multivariate_normal_conjugate() {
    make_model! {
        mod mvn_conjugate;
        use ferric::distributions::MultivariateNormal;
        use nalgebra::DVector;
        use nalgebra::DMatrix;

        let mu : DVector<f64> ~ MultivariateNormal::new(
            DVector::from_vec(vec![0.0, 0.0]),
            DMatrix::identity(2, 2)
        );

        let x : DVector<f64> ~ MultivariateNormal::new(
            mu,
            DMatrix::from_diagonal_element(2, 2, 0.1)
        );

        observe x;
        query mu;
    };

    let model = mvn_conjugate::Model {
        x: DVector::from_vec(vec![1.0, 2.0]),
    };

    let num_samples = 200_000;
    let mut mu0_vals = Vec::with_capacity(num_samples);
    let mut mu1_vals = Vec::with_capacity(num_samples);
    let mut log_weights = Vec::with_capacity(num_samples);

    for ws in model.weighted_sample_iter().take(num_samples) {
        mu0_vals.push(ws.sample.mu[0]);
        mu1_vals.push(ws.sample.mu[1]);
        log_weights.push(ws.log_weight);
    }

    let mean0 = ferric::weighted_mean(&mu0_vals, &log_weights);
    let mean1 = ferric::weighted_mean(&mu1_vals, &log_weights);

    println!("posterior mu = [{:.4}, {:.4}]", mean0, mean1);

    // Analytical posterior means: [10/11, 20/11]
    let expected0 = 10.0 / 11.0;
    let expected1 = 20.0 / 11.0;
    let tol = 0.05;

    assert!(
        (mean0 - expected0).abs() < tol,
        "mu[0] posterior mean {:.4} != expected {:.4}",
        mean0,
        expected0
    );
    assert!(
        (mean1 - expected1).abs() < tol,
        "mu[1] posterior mean {:.4} != expected {:.4}",
        mean1,
        expected1
    );
}
