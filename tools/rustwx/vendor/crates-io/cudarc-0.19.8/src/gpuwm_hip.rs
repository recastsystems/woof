// SPDX-License-Identifier: Apache-2.0
//! Local backend bridge. Driver ownership, slices, events and launch argument
//! lifetimes continue to use the upstream safe layer. No CUDA driver name is
//! replaced in the process or on the system.
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::OnceLock;

pub fn requested() -> bool {
    matches!(std::env::var("GPUWM_RUST_BACKEND").or_else(|_|std::env::var("GPUWM_BACKEND")).as_deref(),Ok("amd"|"hip"|"rocm"|"hip-nvidia"))
}
fn python() -> String {
    std::env::var("GPUWM_PYTHON").unwrap_or_else(|_|"python3".into())
}
pub fn driver_path() -> Result<String,String> {
    static DRIVER: OnceLock<Result<String,String>>=OnceLock::new();
    DRIVER.get_or_init(|| {
        let out=Command::new(python()).args(["-m","gpuwm.hip_rust","--driver"]).output().map_err(|e|e.to_string())?;
        if !out.status.success() {return Err(String::from_utf8_lossy(&out.stderr).into_owned());}
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_owned())
    }).clone()
}
pub fn architecture(ordinal: usize) -> Result<String,String> {
    unsafe {
        let lib=libloading::Library::new(driver_path()?).map_err(|e|e.to_string())?;
        let call=lib.get::<unsafe extern "C" fn(i32,*mut i8,i32)->i32>(b"gpuwmArchitecture").map_err(|e|e.to_string())?;
        let mut text=[0i8;256];
        if call(ordinal as i32,text.as_mut_ptr(),text.len() as i32)!=0 {return Err("HIP architecture query failed".into());}
        Ok(std::ffi::CStr::from_ptr(text.as_ptr()).to_string_lossy().into_owned())
    }
}
fn quoted(s: &str) -> String {
    let mut out=String::from("\"");
    for c in s.chars() {match c {
        '"'=>out.push_str("\\\""),'\\'=>out.push_str("\\\\"),'\n'=>out.push_str("\\n"),'\r'=>out.push_str("\\r"),'\t'=>out.push_str("\\t"),
        c if c<' '=>out.push_str(&format!("\\u{:04x}",c as u32)),c=>out.push(c),
    }}
    out.push('"');out
}
pub fn compile(source: &str,options: &[String]) -> Result<Vec<u8>,String> {
    let options=options.iter().map(|s|quoted(s)).collect::<Vec<_>>().join(",");
    let payload=format!("{{\"source\":{},\"options\":[{}]}}",quoted(source),options);
    let mut child=Command::new(python()).args(["-m","gpuwm.hip_rust"])
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().map_err(|e|e.to_string())?;
    child.stdin.take().ok_or("compiler stdin")?.write_all(payload.as_bytes()).map_err(|e|e.to_string())?;
    let out=child.wait_with_output().map_err(|e|e.to_string())?;
    if !out.status.success() {return Err(String::from_utf8_lossy(&out.stderr).into_owned());}
    if !out.stdout.starts_with(b"\x7fELF") {return Err("HIP compiler did not return an ELF image".into());}
    Ok(out.stdout)
}

/// Resolve quoted local includes at build time with the canonical sources
/// embedded by each helper; no generated HIP source is stored.
pub fn compose(source: &str, headers: &[(&str,&str)]) -> String {
    let mut result=source.to_owned();
    for _ in 0..headers.len()+1 {
        let old=result.clone();
        for (name,text) in headers {
            let guard=format!("GPUWM_EMBEDDED_{}",name.chars().map(|c|if c.is_ascii_alphanumeric(){c.to_ascii_uppercase()}else{'_'}).collect::<String>());
            let body=if text.contains("#pragma once") {
                format!("\n#ifndef {guard}\n#define {guard}\n{}\n#endif\n",text.replace("#pragma once",""))
            } else {text.to_string()};
            result=result.replace(&format!("#include \"{name}\""),&body);
        }
        if result==old {return result;}
    }
    result
}

#[cfg(test)]
mod tests {
    #[test] fn json_source_and_nested_headers() {
        assert_eq!(super::quoted("a\n\"\\\t"),"\"a\\n\\\"\\\\\\t\"");
        assert_eq!(super::compose("#include \"a\"",&[("a","#include \"b\""),("b","body")]),"body");
    }
}
