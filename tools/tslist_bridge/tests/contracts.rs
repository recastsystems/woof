use gpuwm_tslist::*;
use std::{
    fs,
    io::Write,
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

fn scratch(name: &str) -> PathBuf {
    let root = std::env::var_os("TSLIST_TEST_SCRATCH")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../build/tslist-tests")
        });
    let path = root.join(format!(
        "{name}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&path).unwrap();
    path
}

fn station() -> PlacedStation {
    PlacedStation {
        station: Station {
            id: 17,
            description: "Native format oracle".into(),
            prefix: "proof".into(),
            first: 35.125,
            second: -97.125,
        },
        i: 35,
        j: 30,
        grid_lat: 35.123,
        grid_lon: -97.123,
        elevation: 321.5,
    }
}
const START: &str = "2000-01-24_12:00:00";

#[test]
fn native_fortran_formatted_write() {
    let dir = scratch("format");
    let compiler = std::env::var("FC").unwrap_or_else(|_| "gfortran".into());
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/native_format.f90");
    let build = Command::new(&compiler)
        .args(["-O2", "-fno-fast-math", "-ffp-contract=off"])
        .arg(source)
        .arg("-o")
        .arg(dir.join("oracle"))
        .output()
        .expect("native Fortran compiler required for format proof");
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    assert!(
        Command::new(dir.join("oracle"))
            .current_dir(&dir)
            .status()
            .unwrap()
            .success()
    );
    let data = fs::read(dir.join("native-values.bin")).unwrap();
    let row: Vec<f32> = data
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
        .collect();
    let actual = header(&station(), Coordinates::Geographic, 2, START).unwrap()
        + &surface_row(2, &station(), &row).unwrap()
        + &profile_row(row[0], &row[1..]);
    fs::write(dir.join("rust-text.txt"), &actual).unwrap();
    let expected = fs::read_to_string(dir.join("native-text.txt")).unwrap();
    assert_eq!(
        actual,
        expected,
        "native text proof artifacts: {}",
        dir.display()
    );
    let fields = [
        "12345-6", " 12D-2 ", "-1.2+1 ", "3 51 25", "1.25E-3", "  -0   ",
    ];
    let data = fs::read(dir.join("native-read.bin")).unwrap();
    for (text, bytes) in fields.into_iter().zip(data.chunks_exact(4)) {
        let native = u32::from_ne_bytes(bytes.try_into().unwrap());
        let got = parse(&list(&[("read", text, "0")]), 1, false).unwrap();
        assert_eq!(
            got.stations[0].first.to_bits(),
            native,
            "native F7.3 {text:?}"
        );
    }
}

fn list(rows: &[(&str, &str, &str)]) -> String {
    let mut text = format!(
        "# tslist\n{:32}{:7}  {:7}\n# fixed columns\n",
        "", "LAT", "LON"
    );
    for (prefix, a, b) in rows {
        text += &format!("{:25} {:5} {:>7} {:>8}\n", "station", prefix, a, b);
    }
    text
}

#[test]
fn fixed_columns_limits_and_collisions() {
    let text = list(&[("abc", "35125", "-97125"), ("bad!", "broken", "broken")]);
    let got = parse(&text, 1, false).unwrap();
    assert_eq!(got.stations[0].first.to_bits(), 35.125_f32.to_bits());
    assert_eq!(got.stations[0].second.to_bits(), (-97.125_f32).to_bits());
    assert!(got.truncated);
    assert!(parse(&text, 2, false).is_err());
    assert!(
        parse(
            &list(&[("same", "1.0", "2.0"), ("same", "3.0", "4.0")]),
            5,
            false
        )
        .is_err()
    );
    assert!(parse(&list(&[("../x", "1.0", "2.0")]), 5, false).is_err());
    assert!(parse("", 0, false).unwrap().stations.is_empty());
}

#[test]
fn checkpoint_rollback_and_read_only_validation() {
    let dir = scratch("resume");
    let mut writer =
        Writer::create(&dir, vec![station()], Coordinates::Geographic, 2, 2, START).unwrap();
    let stride = 15 + 7 * 2;
    let mut values = vec![1.0; stride];
    values[0] = 0.5;
    writer.write_batch(&values, 1).unwrap();
    let saved = writer.receipt().unwrap();
    values[0] = 1.0;
    writer.write_batch(&values, 1).unwrap();
    let continued: Vec<Vec<u8>> = ENDINGS
        .iter()
        .map(|e| fs::read(dir.join(format!("proof.d02.{e}"))).unwrap())
        .collect();
    drop(writer);
    // A last-file corruption must leave every earlier uncheckpointed tail.
    let last = dir.join("proof.d02.PR");
    let mut damaged = continued[7].clone();
    damaged[0] = b'!';
    fs::write(&last, &damaged).unwrap();
    assert!(
        Writer::resume(
            &dir,
            vec![station()],
            Coordinates::Geographic,
            2,
            2,
            START,
            &saved
        )
        .is_err()
    );
    assert_eq!(fs::read(dir.join("proof.d02.TS")).unwrap(), continued[0]);
    fs::write(&last, &continued[7]).unwrap();
    let mut writer = Writer::resume(
        &dir,
        vec![station()],
        Coordinates::Geographic,
        2,
        2,
        START,
        &saved,
    )
    .unwrap();
    writer.write_batch(&values, 1).unwrap();
    for (index, e) in ENDINGS.iter().enumerate() {
        assert_eq!(
            fs::read(dir.join(format!("proof.d02.{e}"))).unwrap(),
            continued[index]
        );
    }
    assert!(writer.write_batch(&values, 1).is_err());
    assert!(Writer::create(&dir, vec![station()], Coordinates::Geographic, 2, 2, START).is_err());
    let mut wrong = saved.clone();
    wrong.layout_hash ^= 1;
    assert!(
        Writer::resume(
            &dir,
            vec![station()],
            Coordinates::Geographic,
            2,
            2,
            START,
            &wrong
        )
        .is_err()
    );
    // Files shorter than the saved prefix are never reconstructed from zeros.
    fs::OpenOptions::new()
        .write(true)
        .open(&last)
        .unwrap()
        .set_len(1)
        .unwrap();
    assert!(
        Writer::resume(
            &dir,
            vec![station()],
            Coordinates::Geographic,
            2,
            2,
            START,
            &saved
        )
        .is_err()
    );
}

#[test]
fn pending_member_rollback_checks_all_headers_before_truncation() {
    let dir = scratch("pending");
    let mut writer =
        Writer::rewind_initial(&dir, vec![station()], Coordinates::Geographic, 2, 2, START)
            .unwrap();
    let mut values = vec![1.0; 29];
    values[0] = 0.5;
    writer.write_batch(&values, 1).unwrap();
    drop(writer);
    let first = fs::read(dir.join("proof.d02.TS")).unwrap();
    let last = dir.join("proof.d02.PR");
    let original = fs::read(&last).unwrap();
    let mut corrupt = original.clone();
    corrupt[0] = b'!';
    fs::write(&last, corrupt).unwrap();
    assert!(
        Writer::rewind_initial(&dir, vec![station()], Coordinates::Geographic, 2, 2, START,)
            .is_err()
    );
    assert_eq!(fs::read(dir.join("proof.d02.TS")).unwrap(), first);
    fs::write(&last, original).unwrap();
    let mut writer =
        Writer::rewind_initial(&dir, vec![station()], Coordinates::Geographic, 2, 2, START)
            .unwrap();
    writer.write_batch(&values, 1).unwrap();
    assert_eq!(fs::read(dir.join("proof.d02.TS")).unwrap(), first);
}

#[test]
fn writer_orders_batches_and_station_clocks() {
    let dir = scratch("batch");
    let mut second = station();
    second.station.prefix = "other".into();
    second.station.id = 18;
    let mut writer = Writer::create(
        &dir,
        vec![station(), second],
        Coordinates::Grid,
        2,
        1,
        START,
    )
    .unwrap();
    let stride = 22;
    let mut values = vec![0.0; 2 * stride];
    values[0] = 0.25;
    values[stride] = 0.5;
    let before = fs::read(dir.join("proof.d02.TS")).unwrap();
    assert!(writer.write_batch(&values, 1).is_err());
    assert_eq!(before, fs::read(dir.join("proof.d02.TS")).unwrap());
    values[stride] = 0.25;
    writer.write_batch(&values, 1).unwrap();
    let saved = writer.receipt().unwrap();
    for (p, r) in ENDINGS
        .iter()
        .map(|e| dir.join(format!("proof.d02.{e}")))
        .zip(&saved.files[..8])
    {
        assert_eq!(fs::metadata(p).unwrap().len(), r.length);
    }
    // A checkpoint's prefix hash detects an external append within the prefix.
    let mut f = fs::OpenOptions::new()
        .append(true)
        .open(dir.join("other.d02.PR"))
        .unwrap();
    f.write_all(b"tail\n").unwrap();
    assert_eq!(saved.files.len(), 16);
}
