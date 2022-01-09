// Copyright 2022 The Ferric AI Project Developers

use crate::distributions::Distribution;
use rand::Rng;

pub struct Bernoulli {
    /// Probability of success.
    p: f64,
}

impl Bernoulli {
    pub fn new(p: f64) -> Result<Bernoulli, String> {
        if !(0f64..=1f64).contains(&p) {
            Err(format! {"Bernoulli: illegal probability `{}` should be between 0 and 1", p})
        } else {
            Ok(Bernoulli { p })
        }
    }
}

impl<R: Rng + ?Sized> Distribution<R> for Bernoulli {
    type Domain = bool;
    fn sample(&self, rng: &mut R) -> bool {
        let val: f64 = rng.gen();
        val < self.p
    }
}

impl std::fmt::Display for Bernoulli {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Bernoulli {{ p = {} }}", self.p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    #[test]
    fn bernoulli_sample() {
        let mut rng = thread_rng();
        let dist = Bernoulli::new(0.1).unwrap();
        let mut succ = 0;
        let trials = 10000;
        for _ in 0..trials {
            if dist.sample(&mut rng) {
                succ += 1;
            }
        }
        let mean = (succ as f64) / (trials as f64);
        assert!((mean - 0.1).abs() < 0.01);
        // both of the following extreme distributios should be allowed
        let _dist2 = Bernoulli::new(0.0).unwrap();
        let _dist3 = Bernoulli::new(1.0).unwrap();
    }

    #[test]
    #[should_panic]
    fn bernoulli_too_low() {
        let _dist = Bernoulli::new(-0.01).unwrap();
    }

    #[test]
    #[should_panic]
    fn bernoulli_too_high() {
        let _dist = Bernoulli::new(1.01).unwrap();
    }
}
