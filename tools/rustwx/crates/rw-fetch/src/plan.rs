//! The pure half of `rw_fetch`: strict `.idx` validation, byte-range
//! coalescing, and the probe-based full-file/idx-subset decision.
//!
//! Everything here is a function of already-observed facts, so the whole
//! decision surface is unit-testable without a network.  The networked
//! half (`crate::net`) does nothing but gather these facts and act on
//! the verdict.

/// How the operator asked for the object's bytes to be moved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModeRequest {
    /// Let the probe decide (the default).
    Auto,
    /// Always take the whole object, `.idx` or not.
    FullFile,
    /// Always take `.idx`-selected byte ranges; refuse if that is unsafe.
    IdxSubset,
}

impl ModeRequest {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "auto" => Ok(Self::Auto),
            "full-file" | "full_file" | "whole" => Ok(Self::FullFile),
            "idx-subset" | "idx_subset" | "subset" => Ok(Self::IdxSubset),
            other => Err(format!(
                "unknown --mode {other:?}; expected auto, full-file, or idx-subset"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::FullFile => "full-file",
            Self::IdxSubset => "idx-subset",
        }
    }
}

/// The byte transport actually chosen for one object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    FullFile,
    IdxSubset,
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FullFile => "full-file",
            Self::IdxSubset => "idx-subset",
        }
    }
}

/// One validated `.idx` line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdxRow {
    /// 1-based sequence number, which must equal the line number.
    pub sequence: u32,
    /// Byte offset of this message inside the GRIB object.
    pub offset: u64,
    /// The variable column (field 4).
    pub variable: String,
    /// The level column (field 5).
    pub level: String,
    /// The whole raw line, for selectors that need the trailing columns.
    pub raw: String,
}

/// Strictly validate an `.idx` payload against the object it indexes.
///
/// This is the truncated / mid-publish `.idx` defence, ported from the
/// rw-wps authorities subsetter (`_parse_index`).  `wx_core`'s
/// `parse_idx` is deliberately lenient -- it skips unparseable lines --
/// so a `.idx` truncated mid-write parses cleanly there and silently
/// drops every trailing message.  Here every one of the following is a
/// hard error:
///
/// * an empty index;
/// * fewer than six colon-delimited fields on any line (NCEP's own
///   format is six, with extended publishers appending more);
/// * a sequence number that does not equal its 1-based line number;
/// * a non-numeric byte offset;
/// * a first record that does not start at byte zero;
/// * offsets that are not strictly increasing;
/// * a last offset at or past the end of the object.
pub fn validate_idx(text: &str, object_bytes: Option<u64>) -> Result<Vec<IdxRow>, String> {
    let mut rows: Vec<IdxRow> = Vec::new();
    for raw_line in text.lines() {
        let line = raw_line.trim_end_matches(['\r', '\n']);
        if line.trim().is_empty() {
            continue;
        }
        let line_number = rows.len() as u32 + 1;
        let fields: Vec<&str> = line.trim_end_matches(':').split(':').collect();
        if fields.len() < 6 {
            return Err(format!(
                "index line {line_number} has {} colon-delimited fields, expected at least 6",
                fields.len()
            ));
        }
        // Sub-message indices are published as `N.M`; the message number
        // is the part before the dot and must still be monotone by line.
        let sequence_token = fields[0].split_once('.').map_or(fields[0], |(base, _)| base);
        let sequence: u32 = sequence_token
            .trim()
            .parse()
            .map_err(|_| format!("index line {line_number} has a non-numeric sequence number"))?;
        if sequence != line_number {
            return Err(format!(
                "index line {line_number} declares sequence {sequence}; \
                 a canonical NOAA index numbers its lines consecutively"
            ));
        }
        let offset: u64 = fields[1]
            .trim()
            .parse()
            .map_err(|_| format!("index line {line_number} has a non-numeric byte offset"))?;
        if line_number == 1 && offset != 0 {
            return Err(format!(
                "index must begin with record 1 at byte zero, found {offset}"
            ));
        }
        if let Some(previous) = rows.last() {
            if offset <= previous.offset {
                return Err(format!(
                    "index offsets are not strictly increasing: line {line_number} \
                     offset {offset} follows offset {}",
                    previous.offset
                ));
            }
        }
        rows.push(IdxRow {
            sequence,
            offset,
            variable: fields[3].to_string(),
            level: fields[4].to_string(),
            raw: line.to_string(),
        });
    }
    if rows.is_empty() {
        return Err("index is empty".to_string());
    }
    if let Some(size) = object_bytes {
        let last = rows[rows.len() - 1].offset;
        if last >= size {
            return Err(format!(
                "index exceeds its source object: last offset {last} >= object size {size}"
            ));
        }
    }
    Ok(rows)
}

/// Total length of the GRIB2 message whose first 16 octets are `head`.
///
/// Section 0 is `"GRIB"`, two reserved octets, the discipline, the
/// edition number, then the big-endian 64-bit total length.  Anything
/// that is not an edition-2 indicator returns `None`, which the caller
/// treats as "coverage unprovable" and answers with the whole file.
pub fn grib2_message_length(head: &[u8]) -> Option<u64> {
    if head.len() < 16 || &head[0..4] != b"GRIB" || head[7] != 2 {
        return None;
    }
    let mut length: u64 = 0;
    for byte in &head[8..16] {
        length = (length << 8) | u64::from(*byte);
    }
    if length < 16 {
        return None;
    }
    Some(length)
}

/// Everything the probe learned about one (object, index) pair.
///
/// Every field here is gathered through `wx_core`'s **public** client
/// methods, which means every request that produced them went through
/// the cross-process NOMADS rate governor (`client.rs:178-232`).  That
/// is why coverage is proven with a bounded tail range GET rather than
/// a bare `HEAD`/`Content-Length`: the client exposes no response
/// headers, and reaching around it to `agent()` would put unpaced
/// traffic on NOMADS.
#[derive(Debug, Clone, Default)]
pub struct ProbeFacts {
    /// The GRIB object answered an existence probe.
    pub object_present: bool,
    /// The existence probe got no answer at all (the network, or a 429 or
    /// 5xx on both attempts): why.  `None` when the origin answered,
    /// present or not.
    pub object_unreachable: Option<String>,
    /// Total object size, known exactly only when the index was proven
    /// to cover it (`last_offset + last_message_bytes`).  A short index
    /// leaves this `None`; the full-file transport then measures the
    /// object itself.
    pub object_bytes: Option<u64>,
    /// The source published an `.idx` URL at all.
    pub idx_declared: bool,
    /// The `.idx` was fetched successfully.
    pub idx_fetched: bool,
    /// The `.idx` passed [`validate_idx`].  Carries the failure reason.
    pub idx_error: Option<String>,
    /// Byte offset of the last indexed message.
    pub idx_last_offset: Option<u64>,
    /// Declared total length of the last indexed message, read from its
    /// own Section 0 with a 16-byte range GET.
    pub idx_last_message_bytes: Option<u64>,
    /// Number of validated index rows.
    pub idx_rows: usize,
    /// Does the index account for every byte of the object?
    ///
    /// `Some(true)` only when the object provably ends where the last
    /// indexed message ends.  `Some(false)` is a proven-short index --
    /// the object carries messages the index never mentions.  `None`
    /// means the question could not be answered, and the caller must
    /// not guess.
    pub idx_covers_object: Option<bool>,
}

/// The verdict, with the sentence that goes in the fetch record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    Take(Mode, String),
    Refuse(String),
}

/// Choose a byte transport from probe facts alone.
///
/// The rule is the project's, and it has no time constants in it: **if the
/// GRIB is there and its `.idx` is absent, unusable, or shorter than
/// the GRIB implies, take the full file.**  A `.idx` that provably ends
/// where the object ends is the only case that earns a range subset.
/// An explicit `--mode` always wins -- and `--mode idx-subset` refuses
/// loudly rather than quietly degrading, because whoever asked for a
/// subset has to hear that the index could not carry it.
///
/// The reason strings say WHAT chose, never WHO: these two branches ran
/// because `--mode` was on this process's command line, and this
/// process cannot see whether a person typed it or a caller's default
/// supplied it.  They used to claim "operator override", which was
/// simply false for every `woof fetch` -- that front door passes
/// `--mode full-file` as its own default, so the line credited an
/// operator for a choice no operator had made.  The layer that knows
/// says so in its own words (`gpuwm/fetch.py`).
pub fn decide(request: ModeRequest, facts: &ProbeFacts, patterns: usize) -> Decision {
    if !facts.object_present {
        if let Some(reason) = &facts.object_unreachable {
            return Decision::Refuse(format!(
                "this source could not be reached to ask for the GRIB object -- {reason}"
            ));
        }
        return Decision::Refuse("the GRIB object is not present at this source".to_string());
    }
    match request {
        ModeRequest::FullFile => Decision::Take(
            Mode::FullFile,
            "--mode full-file: requested explicitly, no index consulted".to_string(),
        ),
        ModeRequest::IdxSubset => {
            if patterns == 0 {
                return Decision::Refuse(
                    "--mode idx-subset needs at least one --var-pattern to select with".to_string(),
                );
            }
            if !facts.idx_declared {
                return Decision::Refuse(
                    "--mode idx-subset: this source publishes no .idx".to_string(),
                );
            }
            if !facts.idx_fetched {
                return Decision::Refuse(
                    "--mode idx-subset: the .idx could not be fetched".to_string(),
                );
            }
            if let Some(error) = &facts.idx_error {
                return Decision::Refuse(format!(
                    "--mode idx-subset: the .idx failed strict validation: {error}"
                ));
            }
            match facts.idx_covers_object {
                Some(true) => Decision::Take(
                    Mode::IdxSubset,
                    "--mode idx-subset: requested explicitly, index verified complete"
                        .to_string(),
                ),
                Some(false) => Decision::Refuse(
                    "--mode idx-subset: the .idx is short -- its last message does not \
                     end at the end of the object, so a subset would silently drop \
                     trailing records"
                        .to_string(),
                ),
                None => Decision::Refuse(
                    "--mode idx-subset: index coverage could not be proven against the \
                     object: the origin stated no object length and did not answer the \
                     past-the-end probe, so a subset could silently drop records the \
                     index does not list; re-run with --mode full-file"
                        .to_string(),
                ),
            }
        }
        ModeRequest::Auto => {
            if patterns == 0 {
                return Decision::Take(
                    Mode::FullFile,
                    "no --var-pattern given, so there is nothing to subset".to_string(),
                );
            }
            if !facts.idx_declared {
                return Decision::Take(
                    Mode::FullFile,
                    "this source publishes no .idx".to_string(),
                );
            }
            if !facts.idx_fetched {
                return Decision::Take(
                    Mode::FullFile,
                    "the .idx is absent -- the run is still landing, or this hour has none"
                        .to_string(),
                );
            }
            if let Some(error) = &facts.idx_error {
                return Decision::Take(
                    Mode::FullFile,
                    format!("the .idx failed strict validation ({error})"),
                );
            }
            match facts.idx_covers_object {
                Some(true) => Decision::Take(
                    Mode::IdxSubset,
                    format!(
                        "the .idx accounts for all {} bytes of the object across {} records",
                        facts.object_bytes.unwrap_or_default(),
                        facts.idx_rows
                    ),
                ),
                Some(false) => Decision::Take(
                    Mode::FullFile,
                    format!(
                        "the .idx is short: its {} records end at byte {}, and the object \
                         carries bytes beyond that",
                        facts.idx_rows,
                        facts
                            .idx_last_offset
                            .and_then(|o| facts.idx_last_message_bytes.map(|m| o + m))
                            .unwrap_or_default(),
                    ),
                ),
                None => Decision::Take(
                    Mode::FullFile,
                    "index coverage could not be proven against the object".to_string(),
                ),
            }
        }
    }
}

/// Merge byte ranges that touch, preserving order.
///
/// `wx_core::download::byte_ranges` emits one `(start, end)` per
/// selected message; consecutive messages share a boundary, so a
/// 561-record HRRR selection collapses to a handful of GETs.  Ranges
/// are inclusive on both ends, exactly as the `Range:` header wants.
pub fn coalesce_ranges(ranges: &[(u64, u64)]) -> Vec<(u64, u64)> {
    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(ranges.len());
    for &(start, end) in ranges {
        match merged.last_mut() {
            Some(last) if last.1 != u64::MAX && start == last.1 + 1 => last.1 = end,
            _ => merged.push((start, end)),
        }
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idx_text(rows: &[(u32, u64)]) -> String {
        rows.iter()
            .map(|(seq, off)| format!("{seq}:{off}:d=2026072812:TMP:2 m above ground:anl:\n"))
            .collect()
    }

    #[test]
    fn strict_idx_accepts_a_canonical_index() {
        let rows = validate_idx(&idx_text(&[(1, 0), (2, 100), (3, 250)]), Some(400)).unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[2].offset, 250);
        assert_eq!(rows[0].variable, "TMP");
        assert_eq!(rows[0].level, "2 m above ground");
    }

    #[test]
    fn strict_idx_rejects_a_sequence_that_skips_a_line() {
        // The signature of a truncated-then-appended index: the leading
        // records are fine, the numbering is not.
        let error = validate_idx(&idx_text(&[(1, 0), (3, 100)]), Some(400)).unwrap_err();
        assert!(error.contains("declares sequence 3"), "{error}");
    }

    #[test]
    fn strict_idx_rejects_a_first_record_off_byte_zero() {
        let error = validate_idx(&idx_text(&[(1, 64), (2, 100)]), Some(400)).unwrap_err();
        assert!(error.contains("byte zero"), "{error}");
    }

    #[test]
    fn strict_idx_rejects_non_increasing_offsets() {
        let error = validate_idx(&idx_text(&[(1, 0), (2, 100), (3, 100)]), Some(400)).unwrap_err();
        assert!(error.contains("strictly increasing"), "{error}");
    }

    #[test]
    fn strict_idx_rejects_an_index_past_the_end_of_its_object() {
        let error = validate_idx(&idx_text(&[(1, 0), (2, 500)]), Some(400)).unwrap_err();
        assert!(error.contains("exceeds its source object"), "{error}");
    }

    #[test]
    fn strict_idx_rejects_a_short_field_count_lenient_parsers_skip() {
        // wx_core::parse_idx `continue`s past this line and returns the
        // rest; here it is a hard error.
        let text = "1:0:d=2026072812:TMP:anl:\n2:100:d=2026072812:TMP:2 m above ground:anl:\n";
        let error = validate_idx(text, Some(400)).unwrap_err();
        assert!(error.contains("expected at least 6"), "{error}");
    }

    #[test]
    fn strict_idx_accepts_extended_columns() {
        let text = "1:0:d=2026072812:TMP:2 m above ground:anl:ENS=+1:\n";
        assert_eq!(validate_idx(text, Some(400)).unwrap().len(), 1);
    }

    #[test]
    fn grib2_length_reads_section_zero() {
        let mut head = Vec::from(*b"GRIB");
        head.extend_from_slice(&[0, 0, 0, 2]);
        head.extend_from_slice(&1_234_567u64.to_be_bytes());
        assert_eq!(grib2_message_length(&head), Some(1_234_567));
    }

    #[test]
    fn grib2_length_refuses_a_non_grib_or_edition_one_head() {
        let mut head = Vec::from(*b"GRUB");
        head.extend_from_slice(&[0, 0, 0, 2]);
        head.extend_from_slice(&64u64.to_be_bytes());
        assert_eq!(grib2_message_length(&head), None);
        let mut edition1 = Vec::from(*b"GRIB");
        edition1.extend_from_slice(&[0, 0, 0, 1]);
        edition1.extend_from_slice(&64u64.to_be_bytes());
        assert_eq!(grib2_message_length(&edition1), None);
    }

    fn complete_facts() -> ProbeFacts {
        ProbeFacts {
            object_present: true,
            object_unreachable: None,
            object_bytes: Some(1_000),
            idx_declared: true,
            idx_fetched: true,
            idx_error: None,
            idx_last_offset: Some(900),
            idx_last_message_bytes: Some(100),
            idx_rows: 12,
            idx_covers_object: Some(true),
        }
    }

    fn short_index_facts() -> ProbeFacts {
        // The mid-publish case: the tail probe proved the object runs
        // past the last indexed message, so the object size is not yet
        // known and a range subset would drop the tail in silence.
        ProbeFacts {
            object_bytes: None,
            idx_covers_object: Some(false),
            ..complete_facts()
        }
    }

    #[test]
    fn auto_takes_the_subset_when_the_index_covers_the_object() {
        let decision = decide(ModeRequest::Auto, &complete_facts(), 3);
        assert_eq!(
            decision,
            Decision::Take(
                Mode::IdxSubset,
                "the .idx accounts for all 1000 bytes of the object across 12 records".to_string()
            )
        );
    }

    #[test]
    fn auto_takes_the_full_file_when_the_index_is_short() {
        match decide(ModeRequest::Auto, &short_index_facts(), 3) {
            Decision::Take(Mode::FullFile, why) => {
                assert!(why.contains("the .idx is short"), "{why}");
                assert!(why.contains("end at byte 1000"), "{why}");
            }
            other => panic!("expected a full-file take, got {other:?}"),
        }
    }

    #[test]
    fn auto_takes_the_full_file_when_the_index_is_absent() {
        let mut facts = complete_facts();
        facts.idx_fetched = false;
        facts.idx_last_offset = None;
        facts.idx_last_message_bytes = None;
        facts.idx_covers_object = None;
        match decide(ModeRequest::Auto, &facts, 3) {
            Decision::Take(Mode::FullFile, why) => assert!(why.contains("absent"), "{why}"),
            other => panic!("expected a full-file take, got {other:?}"),
        }
    }

    #[test]
    fn auto_takes_the_full_file_when_the_index_fails_validation() {
        let mut facts = complete_facts();
        facts.idx_error = Some("offsets are not strictly increasing".to_string());
        match decide(ModeRequest::Auto, &facts, 3) {
            Decision::Take(Mode::FullFile, why) => {
                assert!(why.contains("strict validation"), "{why}")
            }
            other => panic!("expected a full-file take, got {other:?}"),
        }
    }

    #[test]
    fn auto_takes_the_full_file_when_coverage_is_unprovable() {
        let mut facts = complete_facts();
        facts.idx_last_message_bytes = None;
        facts.idx_covers_object = None;
        match decide(ModeRequest::Auto, &facts, 3) {
            Decision::Take(Mode::FullFile, why) => {
                assert!(why.contains("could not be proven"), "{why}")
            }
            other => panic!("expected a full-file take, got {other:?}"),
        }
    }

    #[test]
    fn auto_takes_the_full_file_when_nothing_was_asked_to_be_subset() {
        match decide(ModeRequest::Auto, &complete_facts(), 0) {
            Decision::Take(Mode::FullFile, why) => {
                assert!(why.contains("nothing to subset"), "{why}")
            }
            other => panic!("expected a full-file take, got {other:?}"),
        }
    }

    #[test]
    fn explicit_full_file_never_consults_the_index() {
        let mut facts = complete_facts();
        facts.idx_declared = false;
        facts.idx_fetched = false;
        match decide(ModeRequest::FullFile, &facts, 0) {
            Decision::Take(Mode::FullFile, why) => {
                assert!(why.contains("requested explicitly"), "{why}");
                // The reason names WHAT chose, not WHO: this process
                // cannot tell a typed flag from a caller's default, and
                // it used to credit an "operator" for every `gpuwm
                // fetch`, whose own default supplies this mode.
                assert!(!why.contains("operator"), "{why}");
            }
            other => panic!("expected a full-file take, got {other:?}"),
        }
    }

    #[test]
    fn explicit_idx_subset_refuses_a_short_index_rather_than_degrading() {
        match decide(ModeRequest::IdxSubset, &short_index_facts(), 3) {
            Decision::Refuse(why) => assert!(why.contains("short"), "{why}"),
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn explicit_idx_subset_refuses_a_source_without_an_index() {
        let mut facts = complete_facts();
        facts.idx_declared = false;
        match decide(ModeRequest::IdxSubset, &facts, 3) {
            Decision::Refuse(why) => assert!(why.contains("publishes no .idx"), "{why}"),
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_missing_object_is_refused_in_every_mode() {
        let facts = ProbeFacts::default();
        for mode in [ModeRequest::Auto, ModeRequest::FullFile, ModeRequest::IdxSubset] {
            match decide(mode, &facts, 3) {
                Decision::Refuse(why) => assert!(why.contains("not present"), "{why}"),
                other => panic!("expected a refusal for {mode:?}, got {other:?}"),
            }
        }
    }

    #[test]
    fn coalescing_merges_touching_ranges_only() {
        assert_eq!(
            coalesce_ranges(&[(0, 99), (100, 199), (400, 499)]),
            vec![(0, 199), (400, 499)]
        );
    }

    #[test]
    fn coalescing_leaves_an_open_ended_tail_alone() {
        assert_eq!(
            coalesce_ranges(&[(0, 99), (100, u64::MAX)]),
            vec![(0, u64::MAX)]
        );
        assert_eq!(
            coalesce_ranges(&[(0, u64::MAX), (10, 20)]),
            vec![(0, u64::MAX), (10, 20)]
        );
    }

    #[test]
    fn coalescing_an_empty_selection_is_empty() {
        assert!(coalesce_ranges(&[]).is_empty());
    }
}
