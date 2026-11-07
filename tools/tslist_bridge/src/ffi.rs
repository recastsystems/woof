//! The forecast submits whole pinned batches through this C ABI. The bridge
//! owns all formatting and file I/O. Every error crosses as a status, never an
//! unwind. Calls on a handle must be serialized by its one owning worker.
use super::*;
use std::{
    cell::RefCell,
    ffi::{CStr, c_char},
    panic::{AssertUnwindSafe, catch_unwind},
    ptr,
};

thread_local! {static LAST_ERROR:RefCell<String>=const {RefCell::new(String::new())};}
fn guarded<T>(failure: T, body: impl FnOnce() -> io::Result<T>) -> T {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(value)) => value,
        other => {
            let message = match other {
                Ok(Err(e)) => e.to_string(),
                _ => "tslist bridge panic".into(),
            };
            LAST_ERROR.with(|s| *s.borrow_mut() = message);
            failure
        }
    }
}
unsafe fn text<'a>(p: *const c_char) -> io::Result<&'a str> {
    if p.is_null() {
        return Err(invalid("null tslist string"));
    }
    unsafe { CStr::from_ptr(p) }
        .to_str()
        .map_err(|_| invalid("non-UTF8 tslist string"))
}
fn hex(s: &str) -> io::Result<u64> {
    u64::from_str_radix(s, 16).map_err(|_| invalid("invalid tslist receipt hex"))
}
fn number<T: std::str::FromStr>(s: &str) -> io::Result<T> {
    s.parse().map_err(|_| invalid("invalid tslist plan number"))
}
fn bits(s: &str) -> io::Result<f32> {
    Ok(f32::from_bits(
        u32::try_from(hex(s)?).map_err(|_| invalid("invalid REAL bits"))?,
    ))
}
fn plan(text: &str) -> io::Result<Vec<PlacedStation>> {
    let mut stations = Vec::new();
    for line in text.lines() {
        let p: Vec<_> = line.split('\t').collect();
        if p.len() != 11 {
            return Err(invalid("tslist plan needs eleven tab-separated fields"));
        }
        stations.push(PlacedStation {
            station: Station {
                id: number(p[0])?,
                prefix: p[9].into(),
                description: p[10].into(),
                first: bits(p[7])?,
                second: bits(p[8])?,
            },
            i: number(p[1])?,
            j: number(p[2])?,
            grid_lat: bits(p[3])?,
            grid_lon: bits(p[4])?,
            elevation: bits(p[5])?,
        });
        if p[6] != "v1" {
            return Err(invalid("unsupported tslist plan ABI"));
        }
    }
    Ok(stations)
}
fn snapshot_text(s: &Snapshot) -> String {
    let mut text = format!(
        "{:016x}\t{}\n",
        s.layout_hash,
        s.last_hour_bits
            .map_or("none".into(), |b| format!("{b:08x}"))
    );
    for r in &s.files {
        writeln!(text, "{}\t{:016x}", r.length, r.hash).unwrap();
    }
    text
}
fn snapshot_parse(text: &str) -> io::Result<Snapshot> {
    let mut lines = text.lines();
    let first: Vec<_> = lines
        .next()
        .ok_or_else(|| invalid("empty tslist checkpoint"))?
        .split('\t')
        .collect();
    if first.len() != 2 {
        return Err(invalid("invalid tslist checkpoint header"));
    }
    let last_hour_bits = if first[1] == "none" {
        None
    } else {
        Some(u32::try_from(hex(first[1])?).map_err(|_| invalid("invalid clock bits"))?)
    };
    let files = lines
        .map(|line| {
            let p: Vec<_> = line.split('\t').collect();
            if p.len() != 2 {
                return Err(invalid("invalid file receipt"));
            }
            Ok(FileReceipt {
                length: number(p[0])?,
                hash: hex(p[1])?,
            })
        })
        .collect::<io::Result<_>>()?;
    Ok(Snapshot {
        files,
        layout_hash: hex(first[0])?,
        last_hour_bits,
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_tslist_abi_version() -> u32 {
    1
}

fn return_text(value: &str, out: *mut c_char, capacity: usize) -> io::Result<isize> {
    let needed = value
        .len()
        .checked_add(1)
        .ok_or_else(|| invalid("text overflow"))?;
    if capacity >= needed && !out.is_null() {
        unsafe {
            ptr::copy_nonoverlapping(value.as_ptr(), out.cast(), value.len());
            *out.add(value.len()) = 0;
        }
    }
    isize::try_from(needed).map_err(|_| invalid("text overflow"))
}

fn plan_text(stations: &[PlacedStation]) -> String {
    let mut result = String::new();
    for p in stations {
        writeln!(
            result,
            "{}\t{}\t{}\t{:08x}\t{:08x}\t{:08x}\tv1\t{:08x}\t{:08x}\t{}\t{}",
            p.station.id,
            p.i,
            p.j,
            p.grid_lat.to_bits(),
            p.grid_lon.to_bits(),
            p.elevation.to_bits(),
            p.station.first.to_bits(),
            p.station.second.to_bits(),
            p.station.prefix,
            p.station.description
        )
        .unwrap();
    }
    result
}

/// Setup only: read and parse the stock WRF file in Rust. The first line is
/// the coordinate mode; remaining lines use the writer's placement ABI.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_tslist_read(
    path: *const c_char,
    maximum: usize,
    ideal: u32,
    out: *mut c_char,
    capacity: usize,
) -> isize {
    guarded(-1, || {
        let list = parse(
            &std::fs::read_to_string(unsafe { text(path)? })?,
            maximum,
            ideal != 0,
        )?;
        let stations = list
            .stations
            .into_iter()
            .map(|station| PlacedStation {
                station,
                i: 0,
                j: 0,
                grid_lat: 0.,
                grid_lon: 0.,
                elevation: 0.,
            })
            .collect::<Vec<_>>();
        let value = format!(
            "{}\n{}",
            u8::from(list.coordinates == Coordinates::Geographic),
            plan_text(&stations)
        );
        return_text(&value, out, capacity)
    })
}

/// Setup only: nearest grid cells, preserving original station ids and native
/// REAL rounding before NINT. x/y are native one-based projection coordinates.
/// Every containing domain independently owns the same source row.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_tslist_place(
    source: *const c_char,
    x: *const f64,
    y: *const f64,
    count: usize,
    nx: i32,
    ny: i32,
    out: *mut c_char,
    capacity: usize,
) -> isize {
    guarded(-1, || {
        let mut stations = plan(unsafe { text(source)? })?;
        if stations.len() != count || (count > 0 && (x.is_null() || y.is_null())) {
            return Err(invalid("tslist coordinate count differs from source rows"));
        }
        if nx < 1 || ny < 1 {
            return Err(invalid("empty tslist grid"));
        }
        let xs = if count == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(x, count) }
        };
        let ys = if count == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(y, count) }
        };
        for (k, p) in stations.iter_mut().enumerate() {
            let (a, b) = (xs[k] as f32, ys[k] as f32);
            if !a.is_finite() || !b.is_finite() {
                return Err(invalid("nonfinite station projection"));
            }
            p.i = a.round() as i32;
            p.j = b.round() as i32;
        }
        stations.retain(|p| (1..=nx).contains(&p.i) && (1..=ny).contains(&p.j));
        return_text(&plan_text(&stations), out, capacity)
    })
}

/// Returns bytes required including NUL. Insufficient capacity leaves the
/// buffer untouched. The pointer is never retained by Rust.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_tslist_last_error(out: *mut c_char, capacity: usize) -> usize {
    LAST_ERROR.with(|s| {
        let s = s.borrow();
        let needed = s.len() + 1;
        if capacity >= needed && !out.is_null() {
            unsafe {
                ptr::copy_nonoverlapping(s.as_ptr(), out.cast(), s.len());
                *out.add(s.len()) = 0;
            }
        }
        needed
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_tslist_open(
    directory: *const c_char,
    placements: *const c_char,
    geographic: u32,
    domain: u8,
    levels: usize,
    start: *const c_char,
    saved: *const c_char,
) -> *mut Writer {
    guarded(ptr::null_mut(), || {
        let directory = Path::new(unsafe { text(directory)? });
        let stations = plan(unsafe { text(placements)? })?;
        let coordinates = match geographic {
            0 => Coordinates::Grid,
            1 => Coordinates::Geographic,
            _ => return Err(invalid("bad tslist coordinate mode")),
        };
        let start = unsafe { text(start)? };
        let writer = if saved.is_null() {
            Writer::create(directory, stations, coordinates, domain, levels, start)?
        } else {
            Writer::resume(
                directory,
                stations,
                coordinates,
                domain,
                levels,
                start,
                &snapshot_parse(unsafe { text(saved)? })?,
            )?
        };
        Ok(Box::into_raw(Box::new(writer)))
    })
}

/// Only the verified roster's pending-member path may request initial rollback.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_tslist_rewind_initial(
    directory: *const c_char,
    placements: *const c_char,
    geographic: u32,
    domain: u8,
    levels: usize,
    start: *const c_char,
) -> *mut Writer {
    guarded(ptr::null_mut(), || {
        let coordinates = match geographic {
            0 => Coordinates::Grid,
            1 => Coordinates::Geographic,
            _ => return Err(invalid("bad tslist coordinate mode")),
        };
        let writer = Writer::rewind_initial(
            Path::new(unsafe { text(directory)? }),
            plan(unsafe { text(placements)? })?,
            coordinates,
            domain,
            levels,
            unsafe { text(start)? },
        )?;
        Ok(Box::into_raw(Box::new(writer)))
    })
}

/// Validate all file prefixes without truncation, for the model's complete-set
/// restart validation phase. No handle or file mutation survives this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_tslist_validate(
    directory: *const c_char,
    placements: *const c_char,
    geographic: u32,
    domain: u8,
    levels: usize,
    start: *const c_char,
    saved: *const c_char,
) -> i32 {
    guarded(-1, || {
        Writer::validate_resume(
            Path::new(unsafe { text(directory)? }),
            plan(unsafe { text(placements)? })?,
            if geographic != 0 {
                Coordinates::Geographic
            } else {
                Coordinates::Grid
            },
            domain,
            levels,
            unsafe { text(start)? },
            &snapshot_parse(unsafe { text(saved)? })?,
        )?;
        Ok(0)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_tslist_write(
    handle: *mut Writer,
    values: *const f32,
    length: usize,
    samples: usize,
) -> i32 {
    guarded(-1, || {
        if handle.is_null() || values.is_null() || length > isize::MAX as usize / 4 {
            return Err(invalid("bad tslist batch pointer"));
        }
        unsafe { (&mut *handle).write_batch(std::slice::from_raw_parts(values, length), samples) }?;
        Ok(0)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_tslist_checkpoint(
    handle: *mut Writer,
    out: *mut c_char,
    capacity: usize,
) -> isize {
    guarded(-1, || {
        if handle.is_null() {
            return Err(invalid("null tslist writer"));
        }
        let text = snapshot_text(&unsafe { &mut *handle }.receipt()?);
        let needed = text.len() + 1;
        if needed > isize::MAX as usize {
            return Err(invalid("tslist receipt too large"));
        }
        if capacity >= needed && !out.is_null() {
            unsafe {
                ptr::copy_nonoverlapping(text.as_ptr(), out.cast(), text.len());
                *out.add(text.len()) = 0;
            }
        }
        Ok(needed as isize)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_tslist_close(handle: *mut Writer) {
    // Dropping a File has no panicking destructor. No asynchronous task may
    // retain this handle when the owning worker closes it.
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}
