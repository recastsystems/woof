//! Public IEM station temperatures, retained without interpolation or unit conversion.
use std::{env, fs};
fn main() {
    let args: Vec<_> = env::args().collect();
    assert_eq!(args.len(), 3, "observations.csv points.csv");
    let source = fs::read_to_string(&args[1]).unwrap();
    let mut output = String::from("station,valid,lon,lat,value\n");
    let mut count = 0;
    for line in source.lines().skip(1) {
        let p: Vec<_> = line.split(',').collect();
        assert_eq!(p.len(), 8);
        if p[4].parse::<f32>().is_err() {
            continue;
        }
        let valid = p[1].replace(' ', "T");
        output.push_str(&format!(
            "{},{valid}:00Z,{},{},{}\n",
            p[0], p[2], p[3], p[4]
        ));
        count += 1;
    }
    fs::write(&args[2], output).unwrap();
    println!("retained_temperature_reports={count}");
}
