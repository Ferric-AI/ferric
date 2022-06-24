// Copyright 2022 The Ferric AI Project Developers

// re-export make_model from the ferric-macros crate
pub use ferric_macros::make_model;

// Public modules
pub mod core;
pub mod distributions;
pub use self::core::FeOption; // re-export FeOption
