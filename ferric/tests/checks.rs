// Copyright 2022 The Ferric AI Project Developers
use std::process::Command;

#[test]
fn check_formatting() {
    let output = Command::new("cargo")
        .arg("fmt")
        .arg("--all")
        .arg("--")
        .arg("--check")
        .output()
        .expect("failed to check formatting");

    println!("status: {}", output.status);
    println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
    println!("stderr: {}", String::from_utf8_lossy(&output.stderr));

    assert!(output.status.success());
}
