// Copyright 2022 The Ferric AI Project Developers

use ferric::make_model;
use nalgebra::DVector;

make_model! {
    mod multivariate_normal_example;
    use ferric::distributions::MultivariateNormal;
    use nalgebra::DMatrix;
    use nalgebra::DVector;

    let latent_position : DVector<f64> ~ MultivariateNormal::new(
        DVector::from_vec(vec![0.0, 0.0]),
        DMatrix::from_vec(2, 2, vec![1.0, 0.6, 0.6, 1.0])
    );
    let sensor_position : DVector<f64> ~ MultivariateNormal::new(
        latent_position.clone(),
        DMatrix::from_diagonal_element(2, 2, 0.2)
    );

    observe sensor_position;
    query latent_position;
}

fn main() {
    let model = multivariate_normal_example::Model {
        sensor_position: DVector::from_vec(vec![1.2, 0.8]),
    };
    let num_samples = 150_000;

    let mut x0 = Vec::with_capacity(num_samples);
    let mut x1 = Vec::with_capacity(num_samples);
    let mut log_weights = Vec::with_capacity(num_samples);

    for ws in model.weighted_sample_iter().take(num_samples) {
        x0.push(ws.sample.latent_position[0]);
        x1.push(ws.sample.latent_position[1]);
        log_weights.push(ws.log_weight);
    }

    let mean0 = ferric::weighted_mean(&x0, &log_weights);
    let mean1 = ferric::weighted_mean(&x1, &log_weights);

    println!(
        "posterior latent position = [{:.3}, {:.3}] from sensor reading {:?}",
        mean0,
        mean1,
        model.sensor_position.as_slice()
    );
}
