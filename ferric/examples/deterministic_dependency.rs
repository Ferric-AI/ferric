// Copyright 2022 The Ferric AI Project Developers

use ferric::make_model;

make_model! {
    mod deterministic_dependency;
    use ferric::distributions::Bernoulli;
    use ferric::distributions::Normal;

    let switch_on : bool ~ Bernoulli::new(0.35);
    let expected_voltage : f64 = if switch_on { 5.0 } else { 0.0 };
    let measured_voltage : f64 ~ Normal::new(expected_voltage, 0.4);

    observe measured_voltage;
    query switch_on;
    query expected_voltage;
}

fn main() {
    let model = deterministic_dependency::Model {
        measured_voltage: 4.7,
    };
    let num_samples = 100_000;

    let mut switch_on = Vec::with_capacity(num_samples);
    let mut expected_voltage = Vec::with_capacity(num_samples);
    let mut log_weights = Vec::with_capacity(num_samples);

    for ws in model.weighted_sample_iter().take(num_samples) {
        switch_on.push(if ws.sample.switch_on { 1.0 } else { 0.0 });
        expected_voltage.push(ws.sample.expected_voltage);
        log_weights.push(ws.log_weight);
    }

    println!(
        "P(switch_on | measured_voltage = 4.7) = {:.3}",
        ferric::weighted_mean(&switch_on, &log_weights)
    );
    println!(
        "E(expected_voltage | measured_voltage = 4.7) = {:.3}",
        ferric::weighted_mean(&expected_voltage, &log_weights)
    );
}
