// Copyright 2022 The Ferric AI Project Developers

// re-export make_model from the ferric-macros crate
pub use ferric_macros::make_model;

// Public modules
pub mod core;
pub mod distributions;

// re-export FeOption and its variants
pub use self::core::FeOption;
pub use FeOption::{Known, Null, Unknown};
