//! `rw_mpas_static` -- the MPAS static builder and its stored-word tools, in
//! ONE executable.
//!
//! WHY ONE. Every platform wheel must stay under PyPI's 100,000,000-byte
//! per-file limit, and the Linux wheel had about 4.6 MB of room at 2.9.0's
//! base. The static builder, its explicit-path arm, the GWD composition and
//! review tools, the raw-word static comparator and the stored-word stream
//! comparator are six programs over one library; shipped as six binaries
//! each would carry its own copy of the netCDF, serde and static code. Here
//! they share it, and a subcommand costs only its own code.
//!
//! WHY NO LINKED netCDF. `verify` reads native MPAS streams (netCDF-4/HDF5
//! included) through the netCDF C library. Linked at build time, that one
//! comparator made every build of this crate need `-lnetcdf`: the pinned
//! manylinux_2_28 release container and the public CI runners have no
//! libnetcdf, so the whole `tools/rustwx` workspace failed to link, and a
//! wheel binary with `libnetcdf.so` in DT_NEEDED would fail the manylinux
//! policy and refuse to start on any host without it -- taking the static
//! builder `gpuwm mesh` runs down with it. [`netcdf_c`] loads the library at
//! run time, only when a tool that reads through it starts, and refuses by
//! name when it is absent.
//!
//! HOW A TOOL IS CHOSEN, first match wins:
//! 1. the name the executable was invoked under: a link or copy named after
//!    a former binary (`rw_mpas_verify`, `rw_mpas_static_words`, ... or that
//!    name plus a `-suffix`, the staged-copy convention) runs that tool with
//!    the same arguments it always took;
//! 2. a first argument naming a tool (`rw_mpas_static verify --request ...`);
//! 3. otherwise the static door, with the argument vector it always had.

mod bounded;
mod door;
mod gwd_replace;
mod gwd_review;
mod netcdf_c;
mod verify;
mod words;

use std::ffi::OsString;
use std::path::Path;
use std::process::ExitCode;

pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

/// One tool of this executable.
pub struct Tool {
    /// The subcommand: `rw_mpas_static <name> ...`.
    pub name: &'static str,
    /// The binary this tool used to be. Invoked under that name (a link or
    /// a copy), this executable runs the tool with its old argument vector.
    pub legacy: &'static str,
    pub summary: &'static str,
    pub run: fn(Vec<OsString>) -> ExitCode,
}

/// Every tool besides the door, in the order `--abi` and `--help` list them.
/// A new tool is a row here and a module; nothing else dispatches.
pub const TOOLS: [Tool; 5] = [
    Tool {
        name: "bounded",
        legacy: "rw_mpas_static_bounded",
        summary: "the same static builder with every geography directory and the host-memory knobs explicit",
        run: bounded::main_with,
    },
    Tool {
        name: "words",
        legacy: "rw_mpas_static_words",
        summary: "raw stored-word comparison of two complete classic netCDF statics (no tolerance)",
        run: words::main_with,
    },
    Tool {
        name: "gwd-replace",
        legacy: "rw_mpas_gwd_replace",
        summary: "copy a complete init and replace only the ten native GWD fields",
        run: gwd_replace::main_with,
    },
    Tool {
        name: "gwd-review",
        legacy: "rw_mpas_gwd_review",
        summary: "nearest-cell review frame of the ten GWD fields for the renderer (pictures only)",
        run: gwd_review::main_with,
    },
    Tool {
        name: "verify",
        legacy: "rw_mpas_verify",
        summary: "stored-word MPAS stream comparison (reads through the netCDF C library)",
        run: verify::main_with,
    },
];

/// The line `--abi` adds after the door's contract, so a caller that needs a
/// tool can tell this executable from a build that predates it.
pub fn tools_abi_line() -> String {
    let names: Vec<&str> = TOOLS.iter().map(|tool| tool.name).collect();
    format!("TOOLS\t{}", names.join("\t"))
}

/// Does an invoked file stem name `legacy`, exactly or as a `-suffix` copy?
fn stem_names(stem: &str, legacy: &str) -> bool {
    let same = |a: &str, b: &str| if cfg!(windows) { a.eq_ignore_ascii_case(b) } else { a == b };
    if same(stem, legacy) {
        return true;
    }
    stem.len() > legacy.len() + 1
        && stem.is_char_boundary(legacy.len())
        && same(&stem[..legacy.len()], legacy)
        && stem[legacy.len()..].starts_with('-')
}

fn tool_invoked_as(program: &OsString) -> Option<&'static Tool> {
    let stem = Path::new(program).file_stem()?.to_str()?;
    TOOLS.iter().find(|tool| stem_names(stem, tool.legacy))
}

fn tool_named(argument: &OsString) -> Option<&'static Tool> {
    let name = argument.to_str()?;
    TOOLS.iter().find(|tool| tool.name == name)
}

/// Arguments as UTF-8, or the exit a tool that parses text gives for them.
pub fn utf8_arguments(tool: &str, argv: Vec<OsString>) -> Result<Vec<String>, ExitCode> {
    argv.into_iter()
        .map(|argument| {
            argument.into_string().map_err(|raw| {
                eprintln!("{tool}: argument {raw:?} is not valid Unicode");
                ExitCode::from(2)
            })
        })
        .collect()
}

fn main() -> ExitCode {
    rw_mpas::execution::bridge(run_bridge)
}

fn run_bridge() -> ExitCode {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    let mut argv: Vec<OsString> = std::env::args_os().collect();
    let program = if argv.is_empty() { OsString::new() } else { argv.remove(0) };
    if let Some(tool) = tool_invoked_as(&program) {
        return (tool.run)(argv);
    }
    if let Some(tool) = argv.first().and_then(tool_named) {
        argv.remove(0);
        return (tool.run)(argv);
    }
    door::run(argv)
}

#[cfg(test)]
mod dispatch_tests {
    use super::*;

    #[test]
    fn legacy_names_and_staged_copies_select_their_tool() {
        for tool in &TOOLS {
            for program in [
                tool.legacy.to_string(),
                format!("/opt/bin/{}", tool.legacy),
                format!("{}-foundation", tool.legacy),
                format!("{}.exe", tool.legacy),
            ] {
                let chosen = tool_invoked_as(&OsString::from(&program)).map(|t| t.name);
                assert_eq!(chosen, Some(tool.name), "{program}");
            }
        }
    }

    #[test]
    fn the_door_name_and_near_misses_select_no_tool() {
        for program in [
            "rw_mpas_static",
            "/usr/lib/gpuwm/libexec/bridges/rw_mpas_static",
            "rw_mpas_static.exe",
            "rw_mpas_static-r2",
            "rw_mpas_verifyx",
            "rw_mpas_static_wordsmith",
            "",
        ] {
            assert!(tool_invoked_as(&OsString::from(program)).is_none(), "{program}");
        }
    }

    #[test]
    fn subcommands_are_unique_and_listed_in_the_abi_line() {
        let line = tools_abi_line();
        let listed: Vec<&str> = line.split('\t').skip(1).collect();
        assert_eq!(line.split('\t').next(), Some("TOOLS"));
        assert_eq!(listed, TOOLS.iter().map(|t| t.name).collect::<Vec<_>>());
        let mut names = listed.clone();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), TOOLS.len());
        for tool in &TOOLS {
            assert!(tool_named(&OsString::from(tool.name)).is_some());
            assert!(tool_named(&OsString::from(tool.legacy)).is_none(),
                "a former binary name is an invocation name, not a subcommand: {}", tool.legacy);
        }
        // Door flags never collide with a subcommand.
        for flag in ["--grid", "--compare", "--abi", "--version", "--help"] {
            assert!(tool_named(&OsString::from(flag)).is_none());
        }
    }
}
