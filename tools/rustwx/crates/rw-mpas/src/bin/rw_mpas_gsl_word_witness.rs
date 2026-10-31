//! Stored-word witness for the complete GSL output, including read-error paths.
//! Native sources: mpas_gsl_oro_data_sm_scale.F:159-169;
//! mpas_gsl_oro_data_lg_scale.F:163-174; mpas_init_atm_gwd_gsl.F:167-203.
use rw_mpas::gsl_oro::BASE_NAMES;

fn capture(input: &std::path::Path) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let file = netcrust::File::open(input)?;
    let mut fields = Vec::new();
    for suffix in ["ls", "ss"] {
        for base in BASE_NAMES {
            let name = format!("{base}{suffix}");
            let var = file.variable(&name).ok_or_else(|| format!("missing {name}"))?;
            if *var.dtype() != netcrust::DataType::F32 {
                return Err(format!("{name} must contain stored binary32 words").into());
            }
            let words: Vec<u32> = file.read_array::<f32>(&name)?.iter().map(|v| v.to_bits()).collect();
            let mut histogram = std::collections::BTreeMap::<u32, usize>::new();
            for word in &words { *histogram.entry(*word).or_default() += 1; }
            fields.push(serde_json::json!({"field":name,"shape":var.shape(),"words":words,
                                         "word_histogram":histogram}));
        }
    }
    Ok(serde_json::json!({"schema":"gsl-stored-word-witness/v1",
        "source_revision":env!("GPUWM_BRIDGE_SOURCE_REV"),
        "input_sha256":rw_mpas::sha256_file(input)?,"rkind_bytes":4,"fields":fields}))
}

// Compare retained native words after disposable native NetCDF bytes are retired.
// Native driver field inventory: mpas_init_atm_gwd_gsl.F:60-93,179-202.
fn compare(witness: &std::path::Path, input: &std::path::Path)
    -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let reference: serde_json::Value = serde_json::from_slice(&std::fs::read(witness)?)?;
    if reference["schema"] != "gsl-stored-word-witness/v1" || reference["rkind_bytes"] != 4 {
        return Err("unsupported stored-word witness".into());
    }
    let candidate = capture(input)?;
    let originals = reference["fields"].as_array().ok_or("missing native fields")?;
    let candidates = candidate["fields"].as_array().ok_or("missing candidate fields")?;
    if originals.len() != candidates.len() { return Err("incomplete native field inventory".into()); }
    let mut fields = Vec::new();
    let mut total = 0usize;
    let mut unequal = 0usize;
    for (original, current) in originals.iter().zip(candidates) {
        if original["field"] != current["field"] || original["shape"] != current["shape"] {
            return Err("native field inventory or shape differs".into());
        }
        let expected = original["words"].as_array().ok_or("missing native word payload")?;
        let actual = current["words"].as_array().ok_or("missing candidate word payload")?;
        if expected.len() != actual.len() { return Err("incomplete native word payload".into()); }
        let mut differing = 0usize;
        let mut first = None;
        for (i, (left, right)) in expected.iter().zip(actual).enumerate() {
            let word = left.as_u64().filter(|v| *v <= u32::MAX as u64)
                .ok_or("invalid native binary32 word")?;
            let other = right.as_u64().ok_or("invalid candidate binary32 word")?;
            if word != other {
                differing += 1;
                if first.is_none() { first = Some(serde_json::json!({"index":i,"native":word,"candidate":other})); }
            }
        }
        total += expected.len(); unequal += differing;
        fields.push(serde_json::json!({"field":current["field"],"shape":current["shape"],
            "words":expected.len(),"unequal_words":differing,"first_difference":first}));
    }
    Ok(serde_json::json!({"schema":"gsl-stored-word-comparison/v1",
        "source_revision":env!("GPUWM_BRIDGE_SOURCE_REV"),"witness_sha256":rw_mpas::sha256_file(witness)?,
        "native_input_sha256":reference["input_sha256"],"candidate_sha256":candidate["input_sha256"],
        "rkind_bytes":4,"fields":fields,"words":total,"unequal_words":unequal,"pass":unequal==0}))
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let comparison = args.len() == 4 && args[0] == "--compare-witness";
    if !comparison && args.len() != 2 {
        return Err("usage: rw_mpas_gsl_word_witness INPUT.nc RECEIPT.json | --compare-witness WORDS.json INPUT.nc RECEIPT.json".into());
    }
    let input = std::path::Path::new(&args[if comparison { 2 } else { 0 }]);
    let out = std::path::Path::new(&args[if comparison { 3 } else { 1 }]);
    let witness = comparison.then(|| std::path::Path::new(&args[1]));
    for source in std::iter::once(input).chain(witness) {
        if source == out || (out.exists() && source.canonicalize()? == out.canonicalize()?) {
            return Err("receipt cannot overwrite its input".into());
        }
    }
    let receipt = match witness { Some(path) => compare(path, input)?, None => capture(input)? };
    std::fs::write(out, serde_json::to_vec(&receipt)?)?;
    if comparison && receipt["pass"] != true { return Err("stored native words differ".into()); }
    Ok(())
}
fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => { eprintln!("{e}"); std::process::ExitCode::FAILURE }
    }
}
