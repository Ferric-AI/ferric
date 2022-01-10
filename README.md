[![Github Actions Tests](https://github.com/ferric-ai/ferric/actions/workflows/ci.yml/badge.svg?event=push)](https://github.com/Ferric-AI/ferric/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/ferric.svg)](https://crates.io/crates/ferric)
[![Coverage Status](https://coveralls.io/repos/github/Ferric-AI/ferric/badge.svg)](https://coveralls.io/github/Ferric-AI/ferric)

# Ferric
A Probabilistic Programming Language in Rust with a declarative syntax.

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
ferric = "0.1"
```

## Example

```rust
use std::time::Instant;
use ferric::make_model;

make_model! {
    mod grass;
    use ferric::distributions::Bernoulli;

    let rain : bool ~ Bernoulli::new( 0.2 );

    let sprinkler : bool ~
        if rain {
            Bernoulli::new( 0.01 )
        } else {
            Bernoulli::new( 0.4 )
        };

    let grass_wet : bool ~ Bernoulli::new(
        if sprinkler && rain { 0.99 }
        else if sprinkler && !rain { 0.9 }
        else if !sprinkler && rain { 0.8 }
        else { 0.0 }
    );

    observe grass_wet;
    query rain;
    query sprinkler;
}

fn main() {
    let model = grass::Model {grass_wet: true};
    let mut num_rain = 0;
    let mut num_sprinkler = 0;
    let num_samples = 100000;
    let start = Instant::now();
    for sample in model.sample_iter().take(num_samples) {
        if sample.rain {
            num_rain += 1;
        }
        if sample.sprinkler {
            num_sprinkler += 1;
        }
    }
    let num_samples = num_samples as f64;
    println!(
        "posterior: rain = {} sprinkler = {}. Elapsed {} millisec for {} samples",
        (num_rain as f64) / num_samples,
        (num_sprinkler as f64) / num_samples,
        start.elapsed().as_millis(),
        num_samples,
    );
}
```

## License

Licensed under either of

 * [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0)
 * [MIT license](http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
