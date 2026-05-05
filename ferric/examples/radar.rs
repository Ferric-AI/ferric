use ferric::make_model;
use nalgebra::{DMatrix, DVector};
use std::hash::{Hash, Hasher};

#[derive(Clone, Debug)]
pub struct BlipLocation(DVector<f64>);

impl From<DVector<f64>> for BlipLocation {
    fn from(value: DVector<f64>) -> Self {
        Self(value)
    }
}

impl PartialEq for BlipLocation {
    fn eq(&self, other: &Self) -> bool {
        self.0.len() == other.0.len()
            && self
                .0
                .iter()
                .zip(other.0.iter())
                .all(|(left, right)| left.to_bits() == right.to_bits())
    }
}

impl Eq for BlipLocation {}

impl Hash for BlipLocation {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.len().hash(state);
        for value in self.0.iter() {
            value.to_bits().hash(state);
        }
    }
}

impl std::ops::Index<usize> for BlipLocation {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

fn wide_covariance() -> DMatrix<f64> {
    DMatrix::from_diagonal(&DVector::from_vec(vec![10_000.0, 10_000.0, 900.0]))
}

fn transition_covariance() -> DMatrix<f64> {
    DMatrix::from_diagonal(&DVector::from_vec(vec![0.25, 0.25, 0.04]))
}

fn blip_covariance() -> DMatrix<f64> {
    DMatrix::from_diagonal(&DVector::from_vec(vec![0.25, 0.25, 0.04]))
}

fn origin_3d() -> DVector<f64> {
    DVector::from_element(3, 0.0)
}

make_model! {
    name radar;
    use ferric::distributions::MultivariateNormal;
    use ferric::distributions::Poisson;
    use nalgebra::DVector;
    use std::collections::HashSet;
    use super::BlipLocation;
    use super::blip_covariance;
    use super::origin_3d;
    use super::transition_covariance;
    use super::wide_covariance;

    const max_aircraft : u64;
    const max_real_blips_per_aircraft : u64;
    const max_false_blips_per_timestep : u64;
    const time_steps : u64;

    let n : u64 ~ Poisson::new(2.0) max max_aircraft;
    let aircraft_location[aircraft of n, time of time_steps] : DVector<f64> ~ if time == 0 {
        MultivariateNormal::new(origin_3d(), wide_covariance())
    } else {
        MultivariateNormal::new(
            aircraft_location[aircraft, time - 1].clone(),
            transition_covariance(),
        )
    };

    let num_real_blip[aircraft of n, time of time_steps] : u64 ~
        Poisson::new(1.2) max max_real_blips_per_aircraft;
    let real_blip_location[aircraft of n, blip of max_real_blips_per_aircraft, time of time_steps] : DVector<f64> ~
        MultivariateNormal::new(
            aircraft_location[aircraft, time].clone(),
            blip_covariance(),
        );

    let num_false_blip[time of time_steps] : u64 ~
        Poisson::new(1.0) max max_false_blips_per_timestep;
    let false_blip_location[false_blip of max_false_blips_per_timestep, time of time_steps] : DVector<f64> ~
        MultivariateNormal::new(origin_3d(), wide_covariance());

    let all_blip_locations[time of time_steps] : HashSet<BlipLocation> = {
        let mut locations = HashSet::new();
        for aircraft in 0..n {
            let count = num_real_blip[aircraft, time];
            for blip in 0..count {
                locations.insert(BlipLocation::from(real_blip_location[aircraft, blip, time].clone()));
            }
        }
        let false_count = num_false_blip[time];
        for false_blip in 0..false_count {
            locations.insert(BlipLocation::from(false_blip_location[false_blip, time].clone()));
        }
        locations
    };

    query n;
    query all_blip_locations;
}

fn main() {
    let model = radar::Model {
        max_aircraft: 4,
        max_real_blips_per_aircraft: 3,
        max_false_blips_per_timestep: 3,
        time_steps: 5,
    };

    let sample = model.sample_iter().next().unwrap();
    let total_blips: usize = sample
        .all_blip_locations
        .iter()
        .map(|blips| blips.len())
        .sum();
    println!(
        "simulated {} aircraft and {} radar blips over {} timesteps",
        sample.n,
        total_blips,
        sample.all_blip_locations.len()
    );
    for (time_idx, blips) in sample.all_blip_locations.iter().enumerate() {
        println!("time {:02}: {} blips", time_idx, blips.len());
        for blip in blips.iter().take(2) {
            println!("  [{:.2}, {:.2}, {:.2}]", blip[0], blip[1], blip[2]);
        }
    }
}
