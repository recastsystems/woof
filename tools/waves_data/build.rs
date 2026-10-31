use std::process::Command;
fn main(){
    let run=|args:&[&str]|Command::new("git").args(args).output().ok().filter(|v|v.status.success()).map(|v|String::from_utf8_lossy(&v.stdout).trim().to_owned());
    let rev=std::env::var("GPUWM_WAVES_DATA_REV").ok().or_else(||run(&["rev-parse","HEAD"])).expect("wave data build needs source revision");
    println!("cargo:rustc-env=GPUWM_WAVES_DATA_REV={rev}");
    println!("cargo:rerun-if-env-changed=GPUWM_WAVES_DATA_REV");
    if let Some(p)=run(&["rev-parse","--git-path","HEAD"]){println!("cargo:rerun-if-changed={p}");}
    if let Some(branch)=run(&["symbolic-ref","HEAD"]){if let Some(p)=run(&["rev-parse","--git-path",&branch]){println!("cargo:rerun-if-changed={p}");}}
}
