// Copyright 2022 The Ferric AI Project Developers

use ferric::make_model;

make_model! {
    mod dirichlet_distribution;
    use ferric::distributions::Dirichlet;
    use ferric::distributions::Multinomial;

    let category_probs : Vec<f64> ~ Dirichlet::new(vec![1.0, 1.0, 1.0]);
    let observed_counts : Vec<u64> ~ Multinomial::new(30, category_probs.clone());

    observe observed_counts;
    query category_probs;
}

fn main() {
    let model = dirichlet_distribution::Model {
        observed_counts: vec![18, 8, 4],
    };
    let num_samples = 100_000;

    let mut p0 = Vec::with_capacity(num_samples);
    let mut p1 = Vec::with_capacity(num_samples);
    let mut p2 = Vec::with_capacity(num_samples);
    let mut log_weights = Vec::with_capacity(num_samples);

    for ws in model.weighted_sample_iter().take(num_samples) {
        p0.push(ws.sample.category_probs[0]);
        p1.push(ws.sample.category_probs[1]);
        p2.push(ws.sample.category_probs[2]);
        log_weights.push(ws.log_weight);
    }

    let mean0 = ferric::weighted_mean(&p0, &log_weights);
    let mean1 = ferric::weighted_mean(&p1, &log_weights);
    let mean2 = ferric::weighted_mean(&p2, &log_weights);

    println!(
        "posterior category probabilities = [{:.3}, {:.3}, {:.3}]",
        mean0, mean1, mean2
    );
    println!("analytical posterior mean        = [0.576, 0.273, 0.152]");
}
