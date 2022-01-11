// Copyright 2022 The Ferric AI Project Developers
use trybuild;

#[test]
fn ui() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
    t.pass("examples/*.rs");
}
