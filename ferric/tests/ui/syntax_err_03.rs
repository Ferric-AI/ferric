// Copyright 2022 The Ferric AI Project Developers
use ferric::make_model;

make_model! {
    mod grass;
    use ferric::distributions::Bernoulli;

    letu rain : bool ~ Bernoulli::new( 0.2 );

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
    let _model = grass::Model { grass_wet: true };
}
