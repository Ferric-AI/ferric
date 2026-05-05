use ferric::make_model;

make_model! {
    name urn;
    use ferric::distributions::Bernoulli;
    use ferric::distributions::DiscreteUniform;
    use ferric::distributions::Poisson;

    const max_marbles : u64;
    const draws : u64;

    let n : u64 ~ Poisson::new(5.0) max max_marbles;
    let true_blue[marble of n] : bool ~ Bernoulli::new(0.5);
    let draw[draw_idx of draws] : i64 ~
        if n == 0 { DiscreteUniform::new(0, 0) } else { DiscreteUniform::new(0, n as i64 - 1) };
    let observed_blue[draw_idx of draws] : bool ~ Bernoulli::new(
        if n == 0 {
            0.5
        } else if true_blue[draw[draw_idx]] {
            0.9
        } else {
            0.1
        }
    );

    let observed_blue_count : u64 =
        observed_blue.iter().filter(|&&is_blue| is_blue).count() as u64;
    let true_blue_count : u64 =
        true_blue.iter().filter(|&&is_blue| is_blue).count() as u64;
    let true_green_count : u64 = n - true_blue_count;

    observe observed_blue_count;
    query n;
    query true_blue_count;
    query true_green_count;
}

fn main() {
    let observed_blue_count = 4;
    let num_samples = 5_000;
    let model = urn::Model {
        max_marbles: 8,
        draws: 6,
        observed_blue_count,
    };

    let mut n_sum = 0.0;
    let mut blue_sum = 0.0;
    let mut green_sum = 0.0;

    for sample in model.sample_iter().take(num_samples) {
        n_sum += sample.n as f64;
        blue_sum += sample.true_blue_count as f64;
        green_sum += sample.true_green_count as f64;
    }

    let denom = num_samples as f64;
    println!(
        "observed {}/{} blue draws: E[n] = {:.2}, E[true blue] = {:.2}, E[true green] = {:.2}",
        observed_blue_count,
        model.draws,
        n_sum / denom,
        blue_sum / denom,
        green_sum / denom
    );
}
