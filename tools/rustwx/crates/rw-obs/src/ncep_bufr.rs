//! NCEP's BUFR dialect: a file that carries its own dictionary.
//!
//! The observation files NCEP posts (the prepbufr among them) are BUFR
//! containers, but not the WMO layout `bufr.rs` reads.  Three things
//! differ, and this module is those three things:
//!
//! * **The dictionary rides in the file.**  Messages of data category 11
//!   with a zeroed date are dictionary (DX table) messages.  Each holds
//!   byte-aligned character rows: Table A (message type number, mnemonic,
//!   description), Table B (FXY, mnemonic, description, unit, scale,
//!   reference, width) and Table D (FXY, mnemonic, description, child
//!   FXY list).  A run of consecutive dictionary messages is one
//!   dictionary; a later run replaces it for the messages that follow.
//!   The strings are owned: nothing here touches the vendored WMO tables.
//! * **Replication is spelled in the dictionary, not in section 3.**  A
//!   Table D child may be preceded by a replication marker: `101yyy`
//!   (fixed, yyy times), `360001` (16-bit count), `360002` (8-bit count),
//!   `360003` (8-bit count, an event stack) or `360004` (1-bit count), or
//!   by the standard factor descriptors `031002`, `031001`, `031000`.
//!   The tree built from a Table A mnemonic is the template of every
//!   subset of that message type.
//! * **PREPBUFR subsets are byte counted and byte aligned.**  Section 3 of a data
//!   message names the byte counter `063000` and then one Table A
//!   sequence.  Each subset starts on a byte boundary with a 16-bit count
//!   of its own bytes (the count included), and the next subset starts
//!   that many bytes on. Standard satellite layouts omit this counter and
//!   use contiguous uncompressed bits or descriptor-wise WMO compression.
//!
//! Values follow the NCEP library's arithmetic so that a value read here
//! is the same double the library hands a Fortran caller: a number is
//! `(raw + reference) * (1 / 10^scale)`, all bits set is missing, and a
//! code or flag table entry is a number like any other.  That is why the
//! element decoding of `bufr.rs` (which divides, and types codes) is not
//! reused; its bit reader is.
//!
//! Reading by mnemonic follows the library's window rule.  A list of
//! mnemonics is read once per replicate of the nearest enclosing 8-bit or
//! 16-bit replicated sequence of the first mnemonic the template carries
//! (or once for the whole subset where there is none), and each mnemonic
//! yields its first occurrence inside that replicate.  An event stack is
//! an 8-bit replicated sequence nested inside a level: the first
//! occurrence is the value in force, and the later ones are its history.
//!
//! What this reader refuses, by name: a dictionary message whose layout
//! is not version 1, a table row it cannot parse (naming the mnemonic), a
//! Table D child that is in neither table (naming the descriptor and the
//! sequence), an operator other than 201, 202, 207 and 208 (naming it and
//! the sequence), compressed replication counts that differ between subsets, a message type the dictionary does not
//! carry, a subset that runs past its byte count or past section 4
//! (naming the mnemonic and the subset), and bytes between messages that
//! are neither zero padding nor a Fortran record control word (naming the
//! offset).

use std::collections::HashMap;
use std::error::Error;
use std::ops::Range;

use crate::bufr::{be24, descriptor_text, fxy, BitReader};
use crate::err;

/// The NCEP library's missing value (`bmiss`).
pub const BMISS: f64 = 10.0e10;

/// The descriptors of a version-1 dictionary message's section 3.
const DX_DESCRIPTORS: [u32; 15] = [
    103_000, 31_001, 1, 2, 3, 101_000, 31_001, 300_004, 105_000, 31_001, 300_003, 205_064, 101_000,
    31_001, 30,
];
/// The byte counter that opens section 3 of an NCEP data message.
const BYTE_COUNT_DESCRIPTOR: u32 = 63_000;
const TABLE_A_ROW: usize = 67;
const TABLE_B_ROW: usize = 112;
const TABLE_D_ROW: usize = 70;
const CHILD_FXY_CHARS: usize = 6;
const MAX_TREE_DEPTH: usize = 16;
const MAX_NUMBER_WIDTH: u32 = 64;

// ---------------------------------------------------------------- framing

/// One BUFR message of a file: where it starts and its bytes.
#[derive(Debug, Clone, Copy)]
pub struct RawMessage<'a> {
    pub offset: usize,
    pub bytes: &'a [u8],
}

/// How the messages sit in the file.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Framing {
    pub messages: usize,
    /// Messages wrapped in Fortran sequential record control words.
    pub control_word_records: usize,
    /// Zero bytes between and after messages.
    pub padding_bytes: usize,
}

fn control_word(bytes: &[u8], at: usize) -> Option<(usize, usize)> {
    // A record control word precedes `BUFR`: four bytes holding the
    // record's length in either byte order, the message's own length
    // rounded up by at most eight bytes of padding.
    let word = bytes.get(at..at + 4)?;
    if bytes.get(at + 4..at + 8)? != b"BUFR" {
        return None;
    }
    let total = be24(bytes, at + 8)?;
    let little = u32::from_le_bytes([word[0], word[1], word[2], word[3]]) as usize;
    let big = u32::from_be_bytes([word[0], word[1], word[2], word[3]]) as usize;
    [little, big]
        .into_iter()
        .find(|&record| record >= total && record <= total + 8)
        .map(|record| (record, total))
}

/// Split a file into its BUFR messages.  Both forms NCEP writes are read:
/// messages laid end to end with zero padding, and messages wrapped in
/// Fortran record control words.
pub fn split_messages<'a>(bytes: &'a [u8], what: &str) -> Result<(Vec<RawMessage<'a>>, Framing), Box<dyn Error>> {
    let mut messages = Vec::new();
    let mut framing = Framing::default();
    let mut at = 0usize;
    while at < bytes.len() {
        if bytes[at..].starts_with(b"BUFR") {
            let total = be24(bytes, at + 4)
                .ok_or_else(|| err(format!("{what}: the message at byte {at} is cut inside section 0")))?;
            let end = at + total;
            if total < 12 || end > bytes.len() {
                return Err(err(format!(
                    "{what}: the message at byte {at} states {total} bytes and {} remain",
                    bytes.len() - at
                )));
            }
            if &bytes[end - 4..end] != b"7777" {
                return Err(err(format!("{what}: the message at byte {at} does not end with 7777 at its stated length {total}")));
            }
            messages.push(RawMessage { offset: at, bytes: &bytes[at..end] });
            at = end;
        } else if let Some((record, total)) = control_word(bytes, at) {
            let start = at + 4;
            let end = start + total;
            let trailer = start + record;
            if trailer + 4 > bytes.len() || bytes[trailer..trailer + 4] != bytes[at..at + 4] {
                return Err(err(format!(
                    "{what}: the record control word at byte {at} states a {record}-byte record and no matching word closes it"
                )));
            }
            if &bytes[end - 4..end] != b"7777" {
                return Err(err(format!("{what}: the message at byte {start} does not end with 7777 at its stated length {total}")));
            }
            messages.push(RawMessage { offset: start, bytes: &bytes[start..end] });
            framing.control_word_records += 1;
            framing.padding_bytes += record - total;
            at = trailer + 4;
        } else if bytes[at] == 0 {
            framing.padding_bytes += 1;
            at += 1;
        } else {
            return Err(err(format!(
                "{what}: byte {at} (0x{:02X}) is neither a BUFR message, zero padding nor a record control word",
                bytes[at]
            )));
        }
    }
    if messages.is_empty() {
        return Err(err(format!("{what}: no BUFR message in {} bytes", bytes.len())));
    }
    framing.messages = messages.len();
    Ok((messages, framing))
}

// --------------------------------------------------------------- envelope

/// Sections 0, 1 and 3 of one message, and where section 4's data sit.
#[derive(Debug, Clone, PartialEq)]
pub struct Envelope {
    pub edition: u8,
    pub centre: u16,
    pub subcentre: u16,
    pub data_category: u8,
    pub local_subcategory: u8,
    pub local_table_version: u8,
    pub year: u16,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    pub subsets: usize,
    pub compressed: bool,
    pub descriptors: Vec<u32>,
    /// Byte offsets inside the message: the first data byte of section 4
    /// and one past its last byte.
    pub data_start: usize,
    pub data_end: usize,
}

impl Envelope {
    /// The library's test for one of its own dictionary messages:
    /// category 11 with the date zeroed.
    pub fn is_dictionary(&self) -> bool {
        self.data_category == 11 && self.month == 0 && self.day == 0 && self.hour == 0
    }

    /// `YYYYMMDDHH`, the cycle stamp a ten-digit date call returns.
    pub fn date10(&self) -> u32 {
        u32::from(self.year) * 1_000_000 + u32::from(self.month) * 10_000 + u32::from(self.day) * 100 + u32::from(self.hour)
    }
}

pub fn envelope(b: &[u8], what: &str) -> Result<Envelope, Box<dyn Error>> {
    if b.len() < 12 || &b[..4] != b"BUFR" {
        return Err(err(format!("{what}: {} bytes do not start a BUFR message", b.len())));
    }
    let edition = b[7];
    let s1 = 8;
    let l1 = be24(b, s1).ok_or_else(|| err(format!("{what}: section 1 is cut")))?;
    let at = |offset: usize, name: &str| -> Result<u8, Box<dyn Error>> {
        if offset >= l1 {
            return Err(err(format!("{what}: section 1 ({l1} bytes) lacks its {name}")));
        }
        b.get(s1 + offset).copied().ok_or_else(|| err(format!("{what}: section 1 lacks its {name}")))
    };
    let (centre, subcentre, optional, data_category, local_subcategory, local_table_version, year, month, day, hour, minute) =
        match edition {
            3 => {
                let yy = u16::from(at(12, "year")?);
                // NCEP writes the century in the last byte of an 18-byte
                // section 1; without it the two-digit year is windowed
                // as the library does (above 40 is the 1900s).
                let century = if l1 >= 18 { u16::from(at(17, "century")?) } else { 0 };
                let year = if century > 0 {
                    (century - 1) * 100 + yy
                } else if yy > 40 {
                    1900 + yy
                } else {
                    2000 + yy
                };
                (
                    u16::from(at(5, "centre")?),
                    u16::from(at(4, "subcentre")?),
                    at(7, "optional flag")? & 0x80 != 0,
                    at(8, "category")?,
                    at(9, "local subcategory")?,
                    at(11, "local table version")?,
                    year,
                    at(13, "month")?,
                    at(14, "day")?,
                    at(15, "hour")?,
                    at(16, "minute")?,
                )
            }
            4 => (
                u16::from_be_bytes([at(4, "centre")?, at(5, "centre")?]),
                u16::from_be_bytes([at(6, "subcentre")?, at(7, "subcentre")?]),
                at(9, "optional flag")? & 0x80 != 0,
                at(10, "category")?,
                at(12, "local subcategory")?,
                at(14, "local table version")?,
                u16::from_be_bytes([at(15, "year")?, at(16, "year")?]),
                at(17, "month")?,
                at(18, "day")?,
                at(19, "hour")?,
                at(20, "minute")?,
            ),
            other => return Err(err(format!("{what}: BUFR edition {other} is not one this reader carries (3 and 4)"))),
        };
    let mut s3 = s1 + l1;
    if optional {
        s3 += be24(b, s3).ok_or_else(|| err(format!("{what}: section 2 is cut")))?;
    }
    let l3 = be24(b, s3).ok_or_else(|| err(format!("{what}: section 3 is cut")))?;
    if l3 < 7 || s3 + l3 > b.len() {
        return Err(err(format!("{what}: section 3 runs past the message")));
    }
    let subsets = u16::from_be_bytes([b[s3 + 4], b[s3 + 5]]) as usize;
    let compressed = b[s3 + 6] & 0x40 != 0;
    let mut descriptors = Vec::new();
    let mut p = s3 + 7;
    while p + 1 < s3 + l3 {
        let raw = u16::from_be_bytes([b[p], b[p + 1]]);
        descriptors.push(u32::from(raw >> 14) * 100_000 + u32::from((raw >> 8) & 0x3F) * 1000 + u32::from(raw & 0xFF));
        p += 2;
    }
    let s4 = s3 + l3;
    let l4 = be24(b, s4).ok_or_else(|| err(format!("{what}: section 4 is cut")))?;
    if l4 < 4 || s4 + l4 > b.len() {
        return Err(err(format!("{what}: section 4 runs past the message")));
    }
    Ok(Envelope {
        edition,
        centre,
        subcentre,
        data_category,
        local_subcategory,
        local_table_version,
        year,
        month,
        day,
        hour,
        minute,
        subsets,
        compressed,
        descriptors,
        data_start: s4 + 4,
        data_end: s4 + l4,
    })
}

// ------------------------------------------------------------- dictionary

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableAEntry {
    pub message_type: u16,
    pub message_subtype: u16,
    pub mnemonic: String,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElementEntry {
    pub fxy: u32,
    pub mnemonic: String,
    pub description: String,
    /// Upper case, as the library keeps it.
    pub unit: String,
    pub scale: i32,
    pub reference: i64,
    pub width: u32,
}

impl ElementEntry {
    pub fn is_text(&self) -> bool {
        self.unit.starts_with("CCITT")
    }

    /// Code and flag tables keep their table width, scale and reference
    /// under operators 201, 202 and 207.
    fn is_code_or_flag(&self) -> bool {
        self.unit.starts_with("CODE") || self.unit.starts_with("FLAG")
    }
}

/// How a Table D child repeats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Replication {
    None,
    Fixed(u32),
    /// `(X)`: a 16-bit count.
    Delayed16,
    /// `{X}`: an 8-bit count.
    Delayed8,
    /// `[X]`: an 8-bit count, an event stack.
    Stack8,
    /// `<X>`: a 1-bit count (present or absent).
    Bit1,
}

impl Replication {
    fn count_bits(self) -> u32 {
        match self {
            Replication::None | Replication::Fixed(_) => 0,
            Replication::Delayed16 => 16,
            Replication::Delayed8 | Replication::Stack8 => 8,
            Replication::Bit1 => 1,
        }
    }

    fn brackets(self) -> (&'static str, &'static str) {
        match self {
            Replication::None => ("", ""),
            Replication::Fixed(_) => ("\"", "\""),
            Replication::Delayed16 => ("(", ")"),
            Replication::Delayed8 => ("{", "}"),
            Replication::Stack8 => ("[", "]"),
            Replication::Bit1 => ("<", ">"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Member {
    pub fxy: u32,
    pub replication: Replication,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceEntry {
    pub fxy: u32,
    pub mnemonic: String,
    pub description: String,
    pub members: Vec<Member>,
}

/// One dictionary: the tables of a run of dictionary messages.
#[derive(Debug, Clone, Default)]
pub struct Dictionary {
    pub table_a: Vec<TableAEntry>,
    pub elements: Vec<ElementEntry>,
    pub sequences: Vec<SequenceEntry>,
    /// Dictionary messages absorbed (the end marker with no subset is not one).
    pub messages: usize,
    element_by_fxy: HashMap<u32, usize>,
    sequence_by_fxy: HashMap<u32, usize>,
    sequence_by_mnemonic: HashMap<String, usize>,
    element_by_mnemonic: HashMap<String, usize>,
}

fn ascii(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| if (32..127).contains(&b) { b as char } else { ' ' }).collect()
}

fn parse_fxy_text(text: &str) -> Option<u32> {
    let t = text.trim();
    if t.len() != 6 || !t.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    t.parse().ok()
}

/// A signed integer with blanks anywhere (`"+  1"`, `"-      1024"`).
fn parse_spaced_integer(text: &str) -> Option<i64> {
    let packed: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    if packed.is_empty() {
        return None;
    }
    packed.parse().ok()
}

impl Dictionary {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.table_a.is_empty() && self.elements.is_empty() && self.sequences.is_empty()
    }

    pub fn element(&self, fxy: u32) -> Option<&ElementEntry> {
        self.element_by_fxy.get(&fxy).map(|&i| &self.elements[i])
    }

    pub fn sequence(&self, fxy: u32) -> Option<&SequenceEntry> {
        self.sequence_by_fxy.get(&fxy).map(|&i| &self.sequences[i])
    }

    pub fn element_named(&self, mnemonic: &str) -> Option<&ElementEntry> {
        self.element_by_mnemonic.get(mnemonic).map(|&i| &self.elements[i])
    }

    pub fn sequence_named(&self, mnemonic: &str) -> Option<&SequenceEntry> {
        self.sequence_by_mnemonic.get(mnemonic).map(|&i| &self.sequences[i])
    }

    pub fn table_a_named(&self, mnemonic: &str) -> Option<&TableAEntry> {
        self.table_a.iter().find(|a| a.mnemonic == mnemonic)
    }

    /// The program code of an event-writing program: the dictionary lists
    /// each program as a Table D mnemonic in category 63, and the code is
    /// the descriptor's Y.
    pub fn program_code(&self, mnemonic: &str) -> Result<u32, Box<dyn Error>> {
        let sequence = self
            .sequence_named(mnemonic)
            .ok_or_else(|| err(format!("the dictionary carries no Table D mnemonic {mnemonic}, so its program code is unknown")))?;
        let (f, x, y) = fxy(sequence.fxy);
        if f != 3 || x != 63 {
            return Err(err(format!(
                "Table D mnemonic {mnemonic} is {} and a program code lives in category 63",
                descriptor_text(sequence.fxy)
            )));
        }
        Ok(y)
    }

    /// Take in one dictionary message.  The end marker (a dictionary
    /// message with no subset) carries nothing and is accepted.
    pub fn absorb(&mut self, message: &[u8], env: &Envelope, what: &str) -> Result<(), Box<dyn Error>> {
        if env.subsets == 0 {
            return Ok(());
        }
        if env.local_subcategory != 1 {
            return Err(err(format!(
                "{what}: dictionary message layout version {} is not one this reader carries (version 1: Table A rows of \
                 {TABLE_A_ROW} characters, Table B rows of {TABLE_B_ROW}, Table D rows of {TABLE_D_ROW})",
                env.local_subcategory
            )));
        }
        if env.descriptors != DX_DESCRIPTORS {
            let listed: Vec<String> = env.descriptors.iter().map(|&d| descriptor_text(d)).collect();
            return Err(err(format!(
                "{what}: section 3 of this dictionary message lists {} and a version-1 dictionary lists 103000 031001 \
                 000001 000002 000003 101000 031001 300004 105000 031001 300003 205064 101000 031001 000030",
                listed.join(" ")
            )));
        }
        let data = &message[env.data_start..env.data_end];
        let need = |at: usize, count: usize, which: &str| -> Result<&[u8], Box<dyn Error>> {
            data.get(at..at + count)
                .ok_or_else(|| err(format!("{what}: {which} runs past section 4 ({} data bytes)", data.len())))
        };
        let mut at = 0usize;
        let table_a_rows = usize::from(need(at, 1, "the Table A count")?[0]);
        at += 1;
        for row in 0..table_a_rows {
            let text = ascii(need(at, TABLE_A_ROW, &format!("Table A row {}", row + 1))?);
            at += TABLE_A_ROW;
            let number = text[0..3].trim().to_string();
            let mnemonic = text[3..11].trim().to_string();
            let description = text[12..67].trim().to_string();
            // A mnemonic spelled `NCtttsss` states the type and subtype
            // itself; otherwise the type is the row's number.
            let digits = mnemonic.len() == 8 && mnemonic.as_bytes()[2..].iter().all(|b| b.is_ascii_digit());
            let (message_type, message_subtype) = if digits {
                (mnemonic[2..5].parse::<u16>().ok(), mnemonic[5..8].parse::<u16>().ok())
            } else {
                (number.parse::<u16>().ok(), Some(0))
            };
            let (Some(message_type), Some(message_subtype)) = (message_type, message_subtype) else {
                return Err(err(format!("{what}: Table A row {mnemonic:?} carries the number {number:?}, which is no message type")));
            };
            if self.table_a.iter().any(|a| a.mnemonic == mnemonic) {
                return Err(err(format!("{what}: Table A mnemonic {mnemonic} is defined twice in one dictionary")));
            }
            self.table_a.push(TableAEntry { message_type, message_subtype, mnemonic, description });
        }
        let table_b_rows = usize::from(need(at, 1, "the Table B count")?[0]);
        at += 1;
        for row in 0..table_b_rows {
            let text = ascii(need(at, TABLE_B_ROW, &format!("Table B row {}", row + 1))?);
            at += TABLE_B_ROW;
            let mnemonic = text[6..14].trim().to_string();
            let fxy = parse_fxy_text(&text[0..6])
                .ok_or_else(|| err(format!("{what}: Table B row {mnemonic:?} carries the descriptor {:?}", &text[0..6])))?;
            let scale = parse_spaced_integer(&text[94..98]);
            let reference = parse_spaced_integer(&text[98..109]);
            let width = parse_spaced_integer(&text[109..112]);
            let (Some(scale), Some(reference), Some(width)) = (scale, reference, width) else {
                return Err(err(format!(
                    "{what}: Table B mnemonic {mnemonic} ({}) has scale {:?}, reference {:?}, width {:?}, which are not all numbers",
                    descriptor_text(fxy), &text[94..98], &text[98..109], &text[109..112]
                )));
            };
            if width < 0 || scale.abs() > 22 {
                return Err(err(format!("{what}: Table B mnemonic {mnemonic} has width {width} and scale {scale}, outside what a value can hold")));
            }
            if self.element_by_fxy.contains_key(&fxy) || self.element_by_mnemonic.contains_key(&mnemonic) {
                return Err(err(format!("{what}: Table B mnemonic {mnemonic} ({}) is defined twice in one dictionary", descriptor_text(fxy))));
            }
            self.element_by_fxy.insert(fxy, self.elements.len());
            self.element_by_mnemonic.insert(mnemonic.clone(), self.elements.len());
            self.elements.push(ElementEntry {
                fxy,
                mnemonic,
                description: text[15..70].trim().to_string(),
                unit: text[70..94].trim().to_ascii_uppercase(),
                scale: scale as i32,
                reference,
                width: width as u32,
            });
        }
        let table_d_rows = usize::from(need(at, 1, "the Table D count")?[0]);
        at += 1;
        for row in 0..table_d_rows {
            let text = ascii(need(at, TABLE_D_ROW, &format!("Table D row {}", row + 1))?);
            at += TABLE_D_ROW;
            let mnemonic = text[6..14].trim().to_string();
            let fxy = parse_fxy_text(&text[0..6])
                .ok_or_else(|| err(format!("{what}: Table D row {mnemonic:?} carries the descriptor {:?}", &text[0..6])))?;
            let children = usize::from(need(at, 1, &format!("the child count of Table D mnemonic {mnemonic}"))?[0]);
            at += 1;
            let mut members = Vec::new();
            let mut pending = Replication::None;
            for child in 0..children {
                let cell = ascii(need(at, CHILD_FXY_CHARS, &format!("child {} of Table D mnemonic {mnemonic}", child + 1))?);
                at += CHILD_FXY_CHARS;
                let code = parse_fxy_text(&cell)
                    .ok_or_else(|| err(format!("{what}: child {} of Table D mnemonic {mnemonic} is {cell:?}, not a descriptor", child + 1)))?;
                let (f, x, y) = crate::bufr::fxy(code);
                match code {
                    360_001 | 31_002 => pending = Replication::Delayed16,
                    360_002 | 31_001 => pending = Replication::Delayed8,
                    360_003 => pending = Replication::Stack8,
                    360_004 | 31_000 => pending = Replication::Bit1,
                    _ if f == 1 && x == 1 => {
                        // 101yyy: fixed replication of the next child; 101000
                        // announces a delayed one whose factor follows.
                        if y > 0 {
                            pending = Replication::Fixed(y);
                        }
                    }
                    _ if f == 1 => {
                        return Err(err(format!(
                            "{what}: Table D mnemonic {mnemonic} holds replication {} over {x} descriptors; an NCEP \
                             dictionary repeats one child at a time",
                            descriptor_text(code)
                        )));
                    }
                    _ => {
                        let replication = if f == 2 { Replication::None } else { std::mem::replace(&mut pending, Replication::None) };
                        members.push(Member { fxy: code, replication });
                    }
                }
            }
            // The library steps over one zero byte after a row where one is there.
            if data.get(at) == Some(&0) {
                at += 1;
            }
            if self.sequence_by_fxy.contains_key(&fxy) || self.sequence_by_mnemonic.contains_key(&mnemonic) {
                return Err(err(format!("{what}: Table D mnemonic {mnemonic} ({}) is defined twice in one dictionary", descriptor_text(fxy))));
            }
            self.sequence_by_fxy.insert(fxy, self.sequences.len());
            self.sequence_by_mnemonic.insert(mnemonic.clone(), self.sequences.len());
            self.sequences.push(SequenceEntry { fxy, mnemonic, description: text[15..70].trim().to_string(), members });
        }
        self.messages += 1;
        Ok(())
    }

    /// The tree every subset of one message type follows.
    pub fn template(&self, mnemonic: &str) -> Result<Template, Box<dyn Error>> {
        let entry = self
            .table_a_named(mnemonic)
            .ok_or_else(|| err(format!("the dictionary's Table A carries no message type {mnemonic}")))?;
        let sequence = self
            .sequence_named(mnemonic)
            .ok_or_else(|| err(format!("message type {mnemonic} is in Table A and has no Table D sequence")))?;
        let mut builder = Builder { dictionary: self, nodes: Vec::new(), subset: mnemonic, operators: Operators::default() };
        let root = builder.members(sequence, None, 0)?;
        if builder.operators != Operators::default() {
            return Err(err(format!(
                "message type {mnemonic}: an operator 201, 202, 207 or 208 is still in force at the end of the sequence"
            )));
        }
        let mut first_by_tag = HashMap::new();
        for (index, node) in builder.nodes.iter().enumerate() {
            first_by_tag.entry(node.tag.clone()).or_insert(index as u32);
        }
        Ok(Template {
            mnemonic: mnemonic.to_string(),
            message_type: entry.message_type,
            nodes: builder.nodes,
            root,
            first_by_tag,
        })
    }
}

// --------------------------------------------------------------- template

#[derive(Debug, Clone, PartialEq)]
pub enum Body {
    Number { width: u32, scale: i32, reference: i64 },
    Text { width: u32 },
    /// A sequence, or a replicated element.  `sequence` is false for a
    /// replicated Table B element, which opens no reading window.
    Group { replication: Replication, sequence: bool, children: Vec<u32> },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    /// The mnemonic; a replicated element's group carries it in brackets.
    pub tag: String,
    pub parent: Option<u32>,
    pub body: Body,
}

/// The tree of one message type, nodes in the order the subset is read.
#[derive(Debug, Clone)]
pub struct Template {
    pub mnemonic: String,
    pub message_type: u16,
    pub nodes: Vec<Node>,
    pub root: Vec<u32>,
    first_by_tag: HashMap<String, u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Operators {
    width_delta: i32,
    scale_delta: i32,
    reference_factor: i64,
    char_bytes: u32,
}

impl Default for Operators {
    fn default() -> Self {
        Self { width_delta: 0, scale_delta: 0, reference_factor: 1, char_bytes: 0 }
    }
}

struct Builder<'a> {
    dictionary: &'a Dictionary,
    nodes: Vec<Node>,
    subset: &'a str,
    operators: Operators,
}

impl Builder<'_> {
    fn members(&mut self, sequence: &SequenceEntry, parent: Option<u32>, depth: usize) -> Result<Vec<u32>, Box<dyn Error>> {
        if depth > MAX_TREE_DEPTH {
            return Err(err(format!(
                "message type {}: Table D mnemonic {} nests deeper than {MAX_TREE_DEPTH}; a sequence names itself",
                self.subset, sequence.mnemonic
            )));
        }
        let mut children = Vec::new();
        for member in &sequence.members {
            let (f, x, y) = fxy(member.fxy);
            if f == 2 {
                self.operator(x, y, member.fxy, &sequence.mnemonic)?;
                continue;
            }
            if let Some(element) = self.dictionary.element(member.fxy) {
                let leaf_parent = if member.replication == Replication::None {
                    parent
                } else {
                    let (open, close) = member.replication.brackets();
                    let group = self.nodes.len() as u32;
                    self.nodes.push(Node {
                        tag: format!("{open}{}{close}", element.mnemonic),
                        parent,
                        body: Body::Group { replication: member.replication, sequence: false, children: Vec::new() },
                    });
                    children.push(group);
                    Some(group)
                };
                let leaf = self.nodes.len() as u32;
                let body = self.leaf(element, &sequence.mnemonic)?;
                self.nodes.push(Node { tag: element.mnemonic.clone(), parent: leaf_parent, body });
                match leaf_parent {
                    Some(group) if member.replication != Replication::None => {
                        if let Body::Group { children: kids, .. } = &mut self.nodes[group as usize].body {
                            kids.push(leaf);
                        }
                    }
                    _ => children.push(leaf),
                }
            } else if let Some(child) = self.dictionary.sequence(member.fxy) {
                let group = self.nodes.len() as u32;
                self.nodes.push(Node {
                    tag: child.mnemonic.clone(),
                    parent,
                    body: Body::Group { replication: member.replication, sequence: true, children: Vec::new() },
                });
                children.push(group);
                let kids = self.members(child, Some(group), depth + 1)?;
                if let Body::Group { children: slot, .. } = &mut self.nodes[group as usize].body {
                    *slot = kids;
                }
            } else {
                return Err(err(format!(
                    "message type {}: descriptor {} in Table D mnemonic {} is in neither Table B nor Table D of the file's dictionary",
                    self.subset,
                    descriptor_text(member.fxy),
                    sequence.mnemonic
                )));
            }
        }
        Ok(children)
    }

    fn operator(&mut self, x: u32, y: u32, code: u32, sequence: &str) -> Result<(), Box<dyn Error>> {
        match x {
            1 => self.operators.width_delta = if y == 0 { 0 } else { y as i32 - 128 },
            2 => self.operators.scale_delta = if y == 0 { 0 } else { y as i32 - 128 },
            7 => {
                if y == 0 {
                    self.operators.width_delta = 0;
                    self.operators.scale_delta = 0;
                    self.operators.reference_factor = 1;
                } else if y > 18 {
                    return Err(err(format!(
                        "message type {}: operator {} in Table D mnemonic {sequence} scales a reference by 10^{y}, past a 64-bit integer",
                        self.subset,
                        descriptor_text(code)
                    )));
                } else {
                    self.operators.width_delta = ((10 * y + 2) / 3) as i32;
                    self.operators.scale_delta = y as i32;
                    self.operators.reference_factor = 10_i64.pow(y);
                }
            }
            8 => self.operators.char_bytes = y,
            _ => {
                return Err(err(format!(
                    "message type {}: operator {} in Table D mnemonic {sequence} is not one this reader carries (201, 202, 207, 208)",
                    self.subset,
                    descriptor_text(code)
                )))
            }
        }
        Ok(())
    }

    fn leaf(&self, element: &ElementEntry, sequence: &str) -> Result<Body, Box<dyn Error>> {
        if element.is_text() {
            let width = if self.operators.char_bytes > 0 { self.operators.char_bytes * 8 } else { element.width };
            if width % 8 != 0 {
                return Err(err(format!(
                    "message type {}: character mnemonic {} in {sequence} is {width} bits wide, not whole bytes",
                    self.subset, element.mnemonic
                )));
            }
            return Ok(Body::Text { width });
        }
        let (mut width, mut scale, mut reference) = (element.width as i64, element.scale, element.reference);
        if !element.is_code_or_flag() {
            width += i64::from(self.operators.width_delta);
            scale += self.operators.scale_delta;
            reference = reference.checked_mul(self.operators.reference_factor).ok_or_else(|| {
                err(format!("message type {}: mnemonic {} in {sequence} has a reference past a 64-bit integer", self.subset, element.mnemonic))
            })?;
        }
        if width < 1 || width > i64::from(MAX_NUMBER_WIDTH) || scale.abs() > 22 {
            return Err(err(format!(
                "message type {}: mnemonic {} in {sequence} would be {width} bits wide at scale {scale}",
                self.subset, element.mnemonic
            )));
        }
        Ok(Body::Number { width: width as u32, scale, reference })
    }
}

/// `10^-scale` the way the library computes it: the exact power of ten,
/// and its reciprocal for a positive scale.
fn scale_factor(scale: i32) -> f64 {
    let power = 10f64.powi(scale.abs());
    if scale >= 0 {
        1.0 / power
    } else {
        power
    }
}

// ----------------------------------------------------------------- subset

#[derive(Debug, Clone, PartialEq)]
pub enum Datum {
    Number(f64),
    /// The raw bytes of a character element.
    Text(Vec<u8>),
    Missing,
}

impl Datum {
    pub fn number(&self) -> Option<f64> {
        match self {
            Datum::Number(v) => Some(*v),
            _ => None,
        }
    }

    /// Printable characters, trimmed.
    pub fn text(&self) -> Option<String> {
        match self {
            Datum::Text(bytes) => Some(ascii(bytes).trim().to_string()),
            _ => None,
        }
    }

    /// The bits of the double the library returns for this value: the
    /// number itself, 10E10 for missing, and for a string its first eight
    /// bytes (blank padded) laid in memory as a little-endian double.
    pub fn real8_bits(&self) -> u64 {
        match self {
            Datum::Number(v) => v.to_bits(),
            Datum::Missing => BMISS.to_bits(),
            Datum::Text(bytes) => {
                let mut cell = [b' '; 8];
                for (slot, &b) in cell.iter_mut().zip(bytes.iter()) {
                    *slot = b;
                }
                u64::from_le_bytes(cell)
            }
        }
    }
}

const MISSING: Datum = Datum::Missing;

#[derive(Debug, Clone, PartialEq)]
pub struct Leaf {
    pub node: u32,
    pub value: Datum,
}

/// One appearance of a replicated group in a subset: its node and the
/// leaves each replicate holds.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupInstance {
    pub node: u32,
    pub replicates: Vec<Range<u32>>,
}

/// One decoded subset: every value in the order it was read.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Subset {
    pub leaves: Vec<Leaf>,
    pub groups: Vec<GroupInstance>,
    /// The subset's stated byte count (the count's own two bytes included).
    pub byte_count: usize,
}

/// A string of eight bytes or fewer is missing when every bit is set, or
/// when it holds the 10E10 double older files stored for missing.
fn text_is_missing(bytes: &[u8]) -> bool {
    if bytes.len() > 8 {
        return false;
    }
    if (4..=8).contains(&bytes.len()) {
        let mut cell = [0u8; 8];
        cell[..bytes.len()].copy_from_slice(bytes);
        let value = u64::from_le_bytes(cell);
        let mask = if bytes.len() == 8 { u64::MAX } else { (1u64 << (8 * bytes.len())) - 1 };
        if value == 0x2020_20E0_7648_3742 & mask || value == 0x4237_4876_E800_0000 & mask {
            return true;
        }
    }
    !bytes.is_empty() && bytes.iter().all(|&b| b == 0xFF)
}

struct Walker<'a, 't> {
    template: &'t Template,
    reader: BitReader<'a>,
    what: &'t str,
    out: Subset,
}

impl Walker<'_, '_> {
    fn read(&mut self, width: u32, tag: &str) -> Result<u64, Box<dyn Error>> {
        if self.reader.pos + width as usize > self.reader.end {
            return Err(err(format!(
                "{}: mnemonic {tag} needs {width} bits at bit {} and the subset's data end at bit {}",
                self.what, self.reader.pos, self.reader.end
            )));
        }
        self.reader.read(width, self.what)
    }

    fn walk(&mut self, children: &[u32]) -> Result<(), Box<dyn Error>> {
        let template = self.template;
        for &index in children {
            let node = &template.nodes[index as usize];
            match &node.body {
                Body::Number { width, scale, reference } => {
                    let raw = self.read(*width, &node.tag)?;
                    let all_ones = if *width >= 64 { u64::MAX } else { (1u64 << width) - 1 };
                    let value = if raw < all_ones {
                        Datum::Number((raw as i64).wrapping_add(*reference) as f64 * scale_factor(*scale))
                    } else {
                        Datum::Missing
                    };
                    self.out.leaves.push(Leaf { node: index, value });
                }
                Body::Text { width } => {
                    let count = (*width / 8) as usize;
                    let mut bytes = Vec::with_capacity(count);
                    for _ in 0..count {
                        bytes.push(self.read(8, &node.tag)? as u8);
                    }
                    let value = if text_is_missing(&bytes) { Datum::Missing } else { Datum::Text(bytes) };
                    self.out.leaves.push(Leaf { node: index, value });
                }
                Body::Group { replication, children: kids, .. } => {
                    let count = match replication {
                        Replication::None => 1,
                        Replication::Fixed(n) => *n,
                        other => self.read(other.count_bits(), &node.tag)? as u32,
                    };
                    if *replication == Replication::None {
                        self.walk(kids)?;
                        continue;
                    }
                    let slot = self.out.groups.len();
                    self.out.groups.push(GroupInstance { node: index, replicates: Vec::with_capacity(count as usize) });
                    for _ in 0..count {
                        let start = self.out.leaves.len() as u32;
                        self.walk(kids)?;
                        let end = self.out.leaves.len() as u32;
                        self.out.groups[slot].replicates.push(start..end);
                    }
                }
            }
        }
        Ok(())
    }
}

// WMO compressed layout with the same file-provided DX template. Compression
// operates on one descriptor across all subsets. Delayed replication counts
// must agree across subsets (WMO FM94 compressed-data rule).
struct CompressedWalker<'a, 't> {
    template: &'t Template,
    reader: BitReader<'a>,
    what: &'t str,
    out: Vec<Subset>,
}
impl CompressedWalker<'_, '_> {
    fn numbers(&mut self, width: u32) -> Result<Vec<Option<u64>>, Box<dyn Error>> {
        let base = self.reader.read(width, self.what)?;
        let increment_width = self.reader.read(6, self.what)? as u32;
        let missing = if width == 64 { u64::MAX } else { (1u64 << width) - 1 };
        let increment_missing = (1u64 << increment_width) - 1;
        (0..self.out.len()).map(|_| {
            let delta = self.reader.read(increment_width, self.what)?;
            if base == missing || (increment_width > 0 && delta == increment_missing) {
                Ok(None)
            } else { base.checked_add(delta).map(Some).ok_or_else(|| err("compressed BUFR integer overflow")) }
        }).collect()
    }
    fn walk(&mut self, children: &[u32]) -> Result<(), Box<dyn Error>> {
        let template = self.template;
        for &index in children {
            let node = &template.nodes[index as usize];
            match &node.body {
                Body::Number { width, scale, reference } => {
                    let values = self.numbers(*width)?;
                    for (subset, raw) in self.out.iter_mut().zip(values) {
                        let value = raw.map(|v| Datum::Number((v as i64).wrapping_add(*reference) as f64 * scale_factor(*scale))).unwrap_or(Datum::Missing);
                        subset.leaves.push(Leaf { node: index, value });
                    }
                }
                Body::Text { width } => {
                    let base = self.reader.read_bytes((*width / 8) as usize, self.what)?;
                    let increment_bytes = self.reader.read(6, self.what)? as usize;
                    for subset in &mut self.out {
                        let bytes = if increment_bytes == 0 { base.clone() } else { self.reader.read_bytes(increment_bytes, self.what)? };
                        let value = if text_is_missing(&bytes) { Datum::Missing } else { Datum::Text(bytes) };
                        subset.leaves.push(Leaf { node: index, value });
                    }
                }
                Body::Group { replication, children: kids, .. } => {
                    let count = match replication {
                        Replication::None => 1,
                        Replication::Fixed(n) => *n,
                        other => {
                            let counts = self.numbers(other.count_bits())?;
                            let Some(Some(first)) = counts.first() else { return Err(err("missing compressed replication count")); };
                            if *first > 65535 || counts.iter().any(|n| *n != Some(*first)) {
                                return Err(err("compressed replication count differs between subsets or exceeds 65535"));
                            }
                            *first as u32
                        }
                    };
                    if *replication == Replication::None { self.walk(kids)?; continue; }
                    let slots: Vec<usize> = self.out.iter_mut().map(|sub| {
                        let slot = sub.groups.len();
                        sub.groups.push(GroupInstance { node: index, replicates: Vec::with_capacity(count as usize) }); slot
                    }).collect();
                    for _ in 0..count {
                        let starts: Vec<u32> = self.out.iter().map(|sub| sub.leaves.len() as u32).collect();
                        self.walk(kids)?;
                        for ((sub, slot), start) in self.out.iter_mut().zip(&slots).zip(starts) {
                            sub.groups[*slot].replicates.push(start..sub.leaves.len() as u32);
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Decode byte-counted PREPBUFR or WMO plain/compressed satellite subsets.
pub fn decode_subsets(message: &[u8], env: &Envelope, template: &Template, what: &str) -> Result<Vec<Subset>, Box<dyn Error>> {
    if env.compressed {
        let mut walker = CompressedWalker { template, reader: BitReader::new(message, env.data_start * 8, env.data_end * 8), what, out: vec![Subset::default(); env.subsets] };
        walker.walk(&template.root)?;
        return Ok(walker.out);
    }
    if env.descriptors.first() != Some(&BYTE_COUNT_DESCRIPTOR) {
        let mut bit = env.data_start * 8;
        let mut subsets = Vec::with_capacity(env.subsets);
        for _ in 0..env.subsets {
            let mut walker = Walker { template, reader: BitReader::new(message, bit, env.data_end * 8), what, out: Subset::default() };
            walker.walk(&template.root)?;
            bit = walker.reader.pos;
            subsets.push(walker.out);
        }
        return Ok(subsets);
    }
    let mut subsets = Vec::with_capacity(env.subsets);
    let mut at = env.data_start;
    for number in 0..env.subsets {
        let label = format!("{what} subset {}", number + 1);
        if at + 2 > env.data_end {
            return Err(err(format!("{label}: its byte count starts at byte {at}, past section 4's end at byte {}", env.data_end)));
        }
        let byte_count = usize::from(u16::from_be_bytes([message[at], message[at + 1]]));
        let mut walker = Walker {
            template,
            reader: BitReader::new(message, (at + 2) * 8, env.data_end * 8),
            what: &label,
            out: Subset { byte_count, ..Subset::default() },
        };
        walker.walk(&template.root)?;
        let used_bytes = (walker.reader.pos + 7) / 8 - at;
        let last = number + 1 == env.subsets;
        // A subset too long for the 16-bit counter travels alone in its
        // message, so the count is binding only where another subset follows.
        if !last && used_bytes > byte_count {
            return Err(err(format!(
                "{label}: the data ran {used_bytes} bytes and the subset states {byte_count}, so the next subset's start is unknown"
            )));
        }
        subsets.push(walker.out);
        at += byte_count;
    }
    Ok(subsets)
}

// ------------------------------------------------------------------ query

/// Where a list of mnemonics is read once per replicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Window {
    /// No mnemonic of the list is in the template: nothing is read.
    Absent,
    /// The first mnemonic sits in no 8-bit or 16-bit replicated sequence:
    /// one read over the whole subset.
    Whole,
    /// One read per replicate of this group node.
    Group(u32),
}

/// A list of mnemonics resolved against one template.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Query {
    pub nodes: Vec<Option<u32>>,
    pub window: Window,
}

impl Template {
    /// The first node carrying a mnemonic, in reading order.
    pub fn node_of(&self, mnemonic: &str) -> Option<u32> {
        self.first_by_tag.get(mnemonic).copied()
    }

    fn window_of(&self, node: u32) -> Window {
        let mut at = Some(node);
        while let Some(index) = at {
            let node = &self.nodes[index as usize];
            if let Body::Group { replication: Replication::Delayed8 | Replication::Delayed16, sequence: true, .. } = node.body {
                return Window::Group(index);
            }
            at = node.parent;
        }
        Window::Whole
    }

    /// Resolve a mnemonic list (a mnemonic the template lacks reads as
    /// missing).  The window is the first resolved mnemonic's.
    pub fn query(&self, mnemonics: &[&str]) -> Query {
        let nodes: Vec<Option<u32>> = mnemonics.iter().map(|m| self.node_of(m)).collect();
        let window = match nodes.iter().flatten().next() {
            Some(&first) => self.window_of(first),
            None => Window::Absent,
        };
        Query { nodes, window }
    }
}

impl Subset {
    /// The leaf ranges a query is read over, one per level.  For a group
    /// window that is the replicates of the group's first appearance in
    /// the subset, as the library walks them.
    pub fn windows(&self, query: &Query) -> Vec<Range<u32>> {
        match query.window {
            Window::Absent => Vec::new(),
            Window::Whole => vec![0..self.leaves.len() as u32],
            Window::Group(node) => self
                .groups
                .iter()
                .find(|group| group.node == node)
                .map(|group| group.replicates.clone())
                .unwrap_or_default(),
        }
    }

    /// The first occurrence of a node inside a window: the value in force.
    pub fn first(&self, window: &Range<u32>, node: Option<u32>) -> &Datum {
        let Some(node) = node else { return &MISSING };
        self.leaves[window.start as usize..window.end as usize]
            .iter()
            .find(|leaf| leaf.node == node)
            .map(|leaf| &leaf.value)
            .unwrap_or(&MISSING)
    }

    /// Every occurrence of a node inside a window, in order: an event
    /// stack from the newest event to the oldest.
    pub fn events<'a>(&'a self, window: &Range<u32>, node: Option<u32>) -> impl Iterator<Item = &'a Datum> + 'a {
        self.leaves[window.start as usize..window.end as usize]
            .iter()
            .filter(move |leaf| Some(leaf.node) == node)
            .map(|leaf| &leaf.value)
    }
}

// ------------------------------------------------------------------- file

/// One data message of a file: where it is and how to read it.  Its
/// subsets are decoded on demand (`NcepFile::subsets_of`), one message at
/// a time, so a pass over a file holds one message's values, not the
/// file's (a whole 12Z hour held at once peaked at 565 MB).
#[derive(Debug, Clone)]
pub struct DataMessage {
    /// The message's number in the file (1 is the first message).
    pub number: usize,
    pub offset: usize,
    /// The message's length in bytes.
    pub length: usize,
    pub envelope: Envelope,
    /// Which of the file's dictionaries and templates it was read with.
    pub dictionary: usize,
    pub template: usize,
}

/// A whole NCEP BUFR file: its dictionaries, templates and data messages.
#[derive(Debug, Clone, Default)]
pub struct NcepFile {
    pub framing: Framing,
    pub dictionaries: Vec<Dictionary>,
    /// `(dictionary index, template)`, in order of first use.
    pub templates: Vec<(usize, Template)>,
    pub messages: Vec<DataMessage>,
    /// Dictionary messages, the end markers included.
    pub dictionary_messages: usize,
    /// The file's bytes and its name for refusals.
    source: Vec<u8>,
    what: String,
}

impl NcepFile {
    pub fn template_of(&self, message: &DataMessage) -> &Template {
        &self.templates[message.template].1
    }

    pub fn dictionary_of(&self, message: &DataMessage) -> &Dictionary {
        &self.dictionaries[message.dictionary]
    }

    /// One data message's subsets, read with their byte counts.
    pub fn subsets_of(&self, message: &DataMessage) -> Result<Vec<Subset>, Box<dyn Error>> {
        let bytes = self
            .source
            .get(message.offset..message.offset + message.length)
            .ok_or_else(|| err(format!("{}: message {} lies outside the file", self.what, message.number)))?;
        let template = self.template_of(message);
        decode_subsets(
            bytes,
            &message.envelope,
            template,
            &format!("{} message {} (byte {}) type {}", self.what, message.number, message.offset, template.mnemonic),
        )
    }
}

/// Read a file: dictionary messages build the dictionary in force, and
/// each data message is indexed against it (its subsets are decoded by
/// `NcepFile::subsets_of`).
pub fn read_file(bytes: &[u8], what: &str) -> Result<NcepFile, Box<dyn Error>> {
    let (raw, framing) = split_messages(bytes, what)?;
    let mut file = NcepFile { framing, source: bytes.to_vec(), what: what.to_string(), ..NcepFile::default() };
    let mut in_dictionary_run = false;
    let mut template_index: HashMap<(usize, String), usize> = HashMap::new();
    for (index, message) in raw.iter().enumerate() {
        let number = index + 1;
        let label = format!("{what} message {number} (byte {})", message.offset);
        let env = envelope(message.bytes, &label)?;
        if env.is_dictionary() {
            if !in_dictionary_run {
                file.dictionaries.push(Dictionary::new());
                in_dictionary_run = true;
            }
            file.dictionary_messages += 1;
            let current = file.dictionaries.len() - 1;
            file.dictionaries[current].absorb(message.bytes, &env, &label)?;
            continue;
        }
        in_dictionary_run = false;
        let Some(dictionary_slot) = file.dictionaries.len().checked_sub(1) else {
            return Err(err(format!(
                "{label}: a data message of type {} precedes any dictionary message, and this reader takes its tables \
                 from the file alone",
                env.data_category
            )));
        };
        let dictionary = &file.dictionaries[dictionary_slot];
        let named = |position: usize| -> Option<&SequenceEntry> {
            env.descriptors
                .get(position)
                .and_then(|&code| dictionary.sequence(code))
                .filter(|sequence| dictionary.table_a_named(&sequence.mnemonic).is_some())
        };
        let sequence = match (named(1), named(0)) {
            (Some(sequence), _) if env.descriptors.first() == Some(&BYTE_COUNT_DESCRIPTOR) => sequence,
            (Some(sequence), _) => {
                return Err(err(format!(
                    "{label}: section 3 names {} second but opens with {} instead of the byte counter 063000",
                    sequence.mnemonic,
                    descriptor_text(env.descriptors[0])
                )))
            }
            (None, Some(sequence)) if env.descriptors.len() == 1 => sequence,
            (None, Some(_)) => return Err(err(format!("{label}: unexpected trailing section 3 descriptors"))),
            (None, None) => {
                let listed: Vec<String> = env.descriptors.iter().take(2).map(|&d| descriptor_text(d)).collect();
                return Err(err(format!(
                    "{label}: section 3 opens with {} and neither is a Table A sequence of the file's dictionary",
                    listed.join(" ")
                )));
            }
        };
        if env.compressed && env.descriptors.first() == Some(&BYTE_COUNT_DESCRIPTOR) {
            return Err(err(format!("{label}: byte-counted compressed layout is invalid")));
        }
        let entry = dictionary.table_a_named(&sequence.mnemonic).expect("filtered above");
        if u16::from(env.data_category) != entry.message_type {
            return Err(err(format!(
                "{label}: section 1 states message type {} and Table A gives {} the type {}",
                env.data_category, sequence.mnemonic, entry.message_type
            )));
        }
        let key = (dictionary_slot, sequence.mnemonic.clone());
        let template_slot = match template_index.get(&key) {
            Some(&slot) => slot,
            None => {
                let template = dictionary.template(&sequence.mnemonic).map_err(|e| err(format!("{label}: {e}")))?;
                file.templates.push((dictionary_slot, template));
                template_index.insert(key, file.templates.len() - 1);
                file.templates.len() - 1
            }
        };
        file.messages.push(DataMessage {
            number,
            offset: message.offset,
            length: message.bytes.len(),
            envelope: env,
            dictionary: dictionary_slot,
            template: template_slot,
        });
    }
    Ok(file)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::bufr::tests::Encoder;

    /// A test-only writer of version-1 dictionary messages and NCEP data
    /// messages, so the reader is held to bytes whose layout is known.
    pub fn pad(text: &str, width: usize) -> Vec<u8> {
        let mut bytes = text.as_bytes().to_vec();
        assert!(bytes.len() <= width, "{text:?} is wider than {width}");
        bytes.resize(width, b' ');
        bytes
    }

    fn section3(descriptors: &[u32], subsets: u16) -> Vec<u8> {
        let mut s3 = vec![0u8; 7];
        for &d in descriptors {
            let (f, x, y) = fxy(d);
            s3.extend_from_slice(&(((f as u16) << 14) | ((x as u16) << 8) | y as u16).to_be_bytes());
        }
        if s3.len() % 2 == 1 {
            s3.push(0);
        }
        let l3 = s3.len() as u32;
        s3[0..3].copy_from_slice(&l3.to_be_bytes()[1..]);
        s3[4..6].copy_from_slice(&subsets.to_be_bytes());
        s3[6] = 0x80;
        s3
    }

    /// An edition-3 message as NCEP writes it: an 18-byte section 1 with
    /// the century in its last byte.
    pub fn message(category: u8, subcategory: u8, date: (u16, u8, u8, u8), descriptors: &[u32], subsets: u16, data: &[u8]) -> Vec<u8> {
        let mut s1 = vec![0u8; 18];
        s1[2] = 18;
        s1[4] = 3;
        s1[5] = 7;
        s1[8] = category;
        s1[9] = subcategory;
        s1[10] = 36;
        if date.1 != 0 {
            s1[12] = (date.0 % 100) as u8;
            s1[17] = (date.0 / 100 + 1) as u8;
        }
        s1[13] = date.1;
        s1[14] = date.2;
        s1[15] = date.3;
        let s3 = section3(descriptors, subsets);
        let mut s4 = vec![0u8; 4];
        s4.extend_from_slice(data);
        if s4.len() % 2 == 1 {
            s4.push(0);
        }
        let l4 = s4.len() as u32;
        s4[0..3].copy_from_slice(&l4.to_be_bytes()[1..]);
        let total = 8 + s1.len() + s3.len() + s4.len() + 4;
        let mut out = b"BUFR".to_vec();
        out.extend_from_slice(&(total as u32).to_be_bytes()[1..]);
        out.push(3);
        out.extend(s1);
        out.extend(s3);
        out.extend(s4);
        out.extend_from_slice(b"7777");
        out
    }

    pub fn table_b_row(fxy: &str, mnemonic: &str, unit: &str, scale: i32, reference: i64, width: u32) -> Vec<u8> {
        let mut row = pad(fxy, 6);
        row.extend(pad(mnemonic, 8));
        row.push(b' ');
        row.extend(pad(&format!("{mnemonic} description"), 55));
        row.extend(pad(unit, 24));
        row.extend(pad(&format!("{}{:>3}", if scale < 0 { '-' } else { '+' }, scale.abs()), 4));
        row.extend(pad(&format!("{}{:>10}", if reference < 0 { '-' } else { '+' }, reference.abs()), 11));
        row.extend(pad(&format!("{width:>3}"), 3));
        assert_eq!(row.len(), TABLE_B_ROW);
        row
    }

    pub fn table_d_row(fxy: &str, mnemonic: &str, children: &[&str]) -> Vec<u8> {
        let mut row = pad(fxy, 6);
        row.extend(pad(mnemonic, 8));
        row.push(b' ');
        row.extend(pad(&format!("{mnemonic} description"), 55));
        row.push(children.len() as u8);
        for child in children {
            row.extend(pad(child, 6));
        }
        row
    }

    pub fn dictionary_message(table_a: &[(&str, &str)], table_b: &[Vec<u8>], table_d: &[Vec<u8>]) -> Vec<u8> {
        let mut data = vec![table_a.len() as u8];
        for (number, mnemonic) in table_a {
            let mut row = pad(number, 3);
            row.extend(pad(mnemonic, 8));
            row.push(b' ');
            row.extend(pad(&format!("{mnemonic} reports"), 55));
            assert_eq!(row.len(), TABLE_A_ROW);
            data.extend(row);
        }
        data.push(table_b.len() as u8);
        for row in table_b {
            data.extend(row);
        }
        data.push(table_d.len() as u8);
        for row in table_d {
            data.extend(row);
        }
        message(11, 1, (0, 0, 0, 0), &DX_DESCRIPTORS, 1, &data)
    }

    /// The small dictionary the tests of this module and of `prepbufr`
    /// share: one message type `OBSTYP` (number 120) whose subset is a
    /// header (station, position with operator 207 around it, type), then
    /// 8-bit replicated levels each holding a category, a 1-bit pressure
    /// group with an event stack, and a 1-bit temperature group with an
    /// event stack; and a second type `PLAIN` (121) with no levels.
    pub fn small_dictionary() -> Vec<u8> {
        dictionary_message(
            &[("120", "OBSTYP"), ("121", "PLAIN")],
            &[
                table_b_row("001194", "SID", "CCITT IA5", 0, 0, 64),
                table_b_row("006240", "XOB", "DEG E", 2, -18000, 16),
                table_b_row("005002", "YOB", "DEG N", 2, -9000, 15),
                table_b_row("055007", "TYP", "CODE TABLE", 0, 0, 9),
                table_b_row("008193", "CAT", "CODE TABLE", 0, 0, 6),
                table_b_row("007245", "POB", "MB", 1, 0, 14),
                table_b_row("007246", "PQM", "CODE TABLE", 0, 0, 5),
                table_b_row("007247", "PPC", "CODE TABLE", 0, 0, 5),
                table_b_row("012245", "TOB", "DEG C", 1, -2732, 14),
                table_b_row("012246", "TQM", "CODE TABLE", 0, 0, 5),
                table_b_row("012247", "TPC", "CODE TABLE", 0, 0, 5),
                table_b_row("010199", "ELV", "METER", 0, -1000, 17),
            ],
            &[
                table_d_row("348120", "OBSTYP", &["348001", "360002", "348002"]),
                table_d_row("348121", "PLAIN", &["348001"]),
                table_d_row("348001", "HEADR", &["001194", "207003", "006240", "005002", "207000", "055007", "010199"]),
                table_d_row("348002", "PRSLEVEL", &["008193", "360004", "348003", "360004", "348005"]),
                table_d_row("348003", "P___INFO", &["360003", "348004"]),
                table_d_row("348004", "P__EVENT", &["007245", "007246", "007247"]),
                table_d_row("348005", "T___INFO", &["360003", "348006"]),
                table_d_row("348006", "T__EVENT", &["012245", "012246", "012247"]),
                table_d_row("363008", "VIRTMP", &[]),
            ],
        )
    }

    /// One temperature or pressure event: value in tenths, mark, program.
    pub type Event = (Option<u32>, u32, u32);

    /// Pack one `OBSTYP` subset: the header, then the levels, each a
    /// category, pressure events and temperature events (raw integers).
    pub fn obstyp_subset(sid: &str, xob_raw: u64, yob_raw: u64, typ: u64, elv_raw: u64, levels: &[(u64, Vec<Event>, Vec<Event>)]) -> Vec<u8> {
        let mut e = Encoder::new();
        e.put_text(sid, 8);
        e.put(xob_raw, 26);
        e.put(yob_raw, 25);
        e.put(typ, 9);
        e.put(elv_raw, 17);
        e.put(levels.len() as u64, 8);
        for (cat, pressure, temperature) in levels {
            e.put(*cat, 6);
            for (stack, width) in [(pressure, 14u32), (temperature, 14u32)] {
                e.put(u64::from(!stack.is_empty()), 1);
                if stack.is_empty() {
                    continue;
                }
                e.put(stack.len() as u64, 8);
                for (value, mark, program) in stack.iter() {
                    match value {
                        Some(v) => e.put(u64::from(*v), width),
                        None => e.put_missing(width),
                    }
                    e.put(u64::from(*mark), 5);
                    e.put(u64::from(*program), 5);
                }
            }
        }
        let body = e.finish();
        let mut subset = ((body.len() + 2) as u16).to_be_bytes().to_vec();
        subset.extend(body);
        subset
    }

    pub fn data_message(category: u8, sequence: u32, date: (u16, u8, u8, u8), subsets: &[Vec<u8>]) -> Vec<u8> {
        let data: Vec<u8> = subsets.iter().flatten().copied().collect();
        message(category, 0, date, &[BYTE_COUNT_DESCRIPTOR, sequence, 102_000, 31_001, 206_001, 63_255], subsets.len() as u16, &data)
    }

    fn two_level_subset() -> Vec<u8> {
        obstyp_subset(
            "72365",
            7_338_000,
            12_504_000,
            120,
            2620,
            &[
                (0, vec![(Some(8350), 2, 1)], vec![(Some(2732 + 251), 2, 8), (Some(2732 + 248), 1, 1)]),
                (1, vec![(Some(5000), 2, 1)], vec![]),
            ],
        )
    }

    #[test]
    fn the_dictionary_parses_tables_replication_and_program_codes() {
        let bytes = small_dictionary();
        let env = envelope(&bytes, "dictionary").unwrap();
        assert!(env.is_dictionary() && env.edition == 3 && env.subsets == 1);
        let mut d = Dictionary::new();
        d.absorb(&bytes, &env, "dictionary").unwrap();
        assert_eq!((d.table_a.len(), d.elements.len(), d.sequences.len(), d.messages), (2, 12, 9, 1));
        assert_eq!(d.table_a[0], TableAEntry { message_type: 120, message_subtype: 0, mnemonic: "OBSTYP".into(), description: "OBSTYP reports".into() });
        let tob = d.element_named("TOB").unwrap();
        assert_eq!((tob.fxy, tob.scale, tob.reference, tob.width, tob.unit.as_str()), (12_245, 1, -2732, 14, "DEG C"));
        assert!(d.element_named("SID").unwrap().is_text());
        let obstyp = d.sequence_named("OBSTYP").unwrap();
        assert_eq!(obstyp.members, vec![
            Member { fxy: 348_001, replication: Replication::None },
            Member { fxy: 348_002, replication: Replication::Delayed8 },
        ]);
        let level = d.sequence_named("PRSLEVEL").unwrap();
        assert_eq!(level.members[1], Member { fxy: 348_003, replication: Replication::Bit1 });
        assert_eq!(d.sequence_named("P___INFO").unwrap().members, vec![Member { fxy: 348_004, replication: Replication::Stack8 }]);
        // operators stay in the member list as descriptors
        assert!(d.sequence_named("HEADR").unwrap().members.iter().any(|m| m.fxy == 207_003));
        assert_eq!(d.program_code("VIRTMP").unwrap(), 8);
        assert!(d.program_code("NOSUCH").unwrap_err().to_string().contains("NOSUCH"));
        assert!(d.program_code("HEADR").unwrap_err().to_string().contains("category 63"));
        // a second copy of the same message is a duplicate definition
        assert!(d.absorb(&bytes, &env, "again").unwrap_err().to_string().contains("defined twice"));
    }

    #[test]
    fn the_template_applies_operator_207_to_numbers_and_spares_code_tables() {
        let bytes = small_dictionary();
        let env = envelope(&bytes, "dictionary").unwrap();
        let mut d = Dictionary::new();
        d.absorb(&bytes, &env, "dictionary").unwrap();
        let t = d.template("OBSTYP").unwrap();
        let body = |m: &str| t.nodes[t.node_of(m).unwrap() as usize].body.clone();
        // 207003: scale + 3, reference x 1000, width + 10
        assert_eq!(body("XOB"), Body::Number { width: 26, scale: 5, reference: -18_000_000 });
        assert_eq!(body("YOB"), Body::Number { width: 25, scale: 5, reference: -9_000_000 });
        // cancelled by 207000 before TYP and ELV
        assert_eq!(body("TYP"), Body::Number { width: 9, scale: 0, reference: 0 });
        assert_eq!(body("ELV"), Body::Number { width: 17, scale: 0, reference: -1000 });
        assert_eq!(body("SID"), Body::Text { width: 64 });
        // POB reads once per level; a header mnemonic reads once per subset
        assert!(matches!(t.query(&["POB", "TOB"]).window, Window::Group(_)));
        assert_eq!(t.query(&["SID", "XOB"]).window, Window::Whole);
        assert_eq!(t.query(&["NUL", "QOB"]).window, Window::Absent);
        assert_eq!(t.query(&["NUL", "TPC"]).nodes[0], None);
    }

    #[test]
    fn subsets_walk_by_byte_count_and_an_event_stack_reads_newest_first() {
        let mut file = small_dictionary();
        let plain = {
            let mut e = Encoder::new();
            e.put_text("PLAIN1", 8);
            e.put(18_000_000, 26);
            e.put(9_000_000, 25);
            e.put(181, 9);
            e.put_missing(17);
            // four bytes of padding the count covers and the tree does not
            let mut body = e.finish();
            body.extend([0u8; 4]);
            let mut subset = ((body.len() + 2) as u16).to_be_bytes().to_vec();
            subset.extend(body);
            subset
        };
        file.extend(data_message(120, 348_120, (2026, 10, 3, 12), &[two_level_subset(), two_level_subset()]));
        file.extend([0u8; 6]);
        file.extend(data_message(121, 348_121, (2026, 10, 3, 12), &[plain.clone(), plain]));
        let read = read_file(&file, "test").unwrap();
        assert_eq!((read.dictionaries.len(), read.dictionary_messages, read.messages.len(), read.templates.len()), (1, 1, 2, 2));
        assert_eq!(read.framing, Framing { messages: 3, control_word_records: 0, padding_bytes: 6 });
        let m = &read.messages[0];
        let subsets = read.subsets_of(m).unwrap();
        assert_eq!((m.envelope.date10(), m.envelope.data_category, subsets.len()), (2_026_100_312, 120, 2));
        let t = read.template_of(m);
        assert_eq!(t.mnemonic, "OBSTYP");
        let header = t.query(&["SID", "XOB", "YOB", "TYP", "ELV", "SAID"]);
        let levels = t.query(&["POB", "TOB", "CAT"]);
        let marks = t.query(&["PQM", "NUL", "TQM"]);
        let programs = t.query(&["TPC"]);
        for s in &subsets {
            let whole = s.windows(&header);
            assert_eq!(whole.len(), 1);
            assert_eq!(s.first(&whole[0], header.nodes[0]).text().unwrap(), "72365");
            // the library's arithmetic: (raw + reference) * (1 / 10^5)
            assert_eq!(s.first(&whole[0], header.nodes[1]), &Datum::Number(-10_662_000.0 * (1.0 / 100_000.0)));
            assert_eq!(s.first(&whole[0], header.nodes[2]), &Datum::Number(3_504_000.0 * (1.0 / 100_000.0)));
            assert_eq!(s.first(&whole[0], header.nodes[3]), &Datum::Number(120.0));
            assert_eq!(s.first(&whole[0], header.nodes[4]), &Datum::Number(1620.0));
            assert_eq!(s.first(&whole[0], header.nodes[5]), &Datum::Missing);
            let w = s.windows(&levels);
            assert_eq!(w.len(), 2);
            assert_eq!(s.first(&w[0], levels.nodes[0]), &Datum::Number(8350.0 * (1.0 / 10.0)));
            // the value in force is the newest event, 25.1 C; the older 24.8 C is under it
            assert_eq!(s.first(&w[0], levels.nodes[1]), &Datum::Number(251.0 * (1.0 / 10.0)));
            let stack: Vec<f64> = s.events(&w[0], levels.nodes[1]).map(|d| d.number().unwrap()).collect();
            assert_eq!(stack, vec![251.0 * 0.1, 248.0 * 0.1]);
            let codes: Vec<f64> = s.events(&w[0], programs.nodes[0]).map(|d| d.number().unwrap()).collect();
            assert_eq!(codes, vec![8.0, 1.0]);
            // the second level has no temperature group: missing, and no events
            assert_eq!(s.first(&w[1], levels.nodes[1]), &Datum::Missing);
            assert_eq!(s.events(&w[1], programs.nodes[0]).count(), 0);
            assert_eq!(s.first(&w[1], levels.nodes[2]), &Datum::Number(1.0));
            assert_eq!(s.first(&w[0], marks.nodes[0]), &Datum::Number(2.0));
            assert_eq!(s.first(&w[0], marks.nodes[1]), &Datum::Missing);
            assert_eq!(s.windows(&marks).len(), 2);
        }
        // the second message's subsets are found through their byte counts, padding and all
        let p = &read.messages[1];
        let subsets = read.subsets_of(p).unwrap();
        assert_eq!(subsets.len(), 2);
        let t = read.template_of(p);
        let q = t.query(&["SID", "ELV", "POB"]);
        for s in &subsets {
            let w = s.windows(&q);
            assert_eq!(s.first(&w[0], q.nodes[0]).text().unwrap(), "PLAIN1");
            assert_eq!(s.first(&w[0], q.nodes[1]), &Datum::Missing);
            assert_eq!(s.first(&w[0], q.nodes[2]), &Datum::Missing);
            assert_eq!(s.byte_count, 24);
        }
        // a level query on a type without levels reads nothing
        assert_eq!(subsets[0].windows(&t.query(&["POB"])).len(), 0);
    }

    #[test]
    fn real8_bits_match_the_library_for_numbers_strings_and_missing() {
        assert_eq!(Datum::Missing.real8_bits(), 0x4237_4876_E800_0000);
        assert_eq!(Datum::Number(1.5).real8_bits(), 1.5f64.to_bits());
        assert_eq!(Datum::Text(b"KOKC".to_vec()).real8_bits(), u64::from_le_bytes(*b"KOKC    "));
        assert!(text_is_missing(&[0xFF; 8]));
        assert!(text_is_missing(&0x4237_4876_E800_0000u64.to_le_bytes()));
        assert!(!text_is_missing(b"72365   "));
        assert!(!text_is_missing(&[0xFF; 12]));
    }

    #[test]
    fn record_control_words_are_read_in_either_byte_order() {
        let dictionary = small_dictionary();
        let data = data_message(120, 348_120, (2026, 10, 3, 12), &[two_level_subset()]);
        for big_endian in [false, true] {
            let mut file = Vec::new();
            for message in [&dictionary, &data] {
                let padded = (message.len() + 7) / 8 * 8;
                let word = if big_endian { (padded as u32).to_be_bytes() } else { (padded as u32).to_le_bytes() };
                file.extend(word);
                file.extend(message.iter());
                file.extend(vec![0u8; padded - message.len()]);
                file.extend(word);
            }
            let read = read_file(&file, "blocked").unwrap();
            assert_eq!(read.framing.control_word_records, 2);
            assert_eq!(read.subsets_of(&read.messages[0]).unwrap().len(), 1);
        }
    }

    /// Index a file and decode every message, as a whole pass does.
    fn read_all(bytes: &[u8], what: &str) -> Result<(), Box<dyn Error>> {
        let file = read_file(bytes, what)?;
        for message in &file.messages {
            file.subsets_of(message)?;
        }
        Ok(())
    }

    #[test]
    fn refusals_name_the_mnemonic_the_descriptor_and_the_subset() {
        let dictionary = small_dictionary();
        // a subset cut short: the count says 12 bytes, the tree needs more
        let mut short = two_level_subset();
        short.truncate(12);
        short[0..2].copy_from_slice(&12u16.to_be_bytes());
        let mut file = dictionary.clone();
        file.extend(data_message(120, 348_120, (2026, 10, 3, 12), &[short]));
        let error = read_all(&file, "short").unwrap_err().to_string();
        assert!(error.contains("subset 1") && error.contains("mnemonic") && error.contains("OBSTYP"), "{error}");
        // a subset whose data run past its count while another follows
        let mut lying = two_level_subset();
        lying[0..2].copy_from_slice(&10u16.to_be_bytes());
        let mut file = dictionary.clone();
        file.extend(data_message(120, 348_120, (2026, 10, 3, 12), &[lying, two_level_subset()]));
        let error = read_all(&file, "lying").unwrap_err().to_string();
        assert!(error.contains("subset 1") && error.contains("states 10"), "{error}");
        // a data message before any dictionary
        let error = read_file(&data_message(120, 348_120, (2026, 10, 3, 12), &[two_level_subset()]), "bare").unwrap_err().to_string();
        assert!(error.contains("precedes any dictionary"), "{error}");
        // a Table D child in neither table, named with its sequence
        let broken = dictionary_message(&[("120", "OBSTYP")], &[], &[table_d_row("348120", "OBSTYP", &["001194"])]);
        let mut file = broken;
        file.extend(data_message(120, 348_120, (2026, 10, 3, 12), &[two_level_subset()]));
        let error = read_file(&file, "broken").unwrap_err().to_string();
        assert!(error.contains("001194") && error.contains("OBSTYP") && error.contains("neither Table B nor Table D"), "{error}");
        // an operator this reader does not carry, named with its sequence
        let operator = dictionary_message(
            &[("120", "OBSTYP")],
            &[table_b_row("001194", "SID", "CCITT IA5", 0, 0, 64)],
            &[table_d_row("348120", "OBSTYP", &["203014", "001194"])],
        );
        let mut file = operator;
        file.extend(data_message(120, 348_120, (2026, 10, 3, 12), &[two_level_subset()]));
        let error = read_file(&file, "operator").unwrap_err().to_string();
        assert!(error.contains("203014") && error.contains("OBSTYP"), "{error}");
        // bytes between messages that are not padding
        let mut file = dictionary.clone();
        file.extend(b"junk");
        let error = read_file(&file, "junk").unwrap_err().to_string();
        assert!(error.contains("neither a BUFR message, zero padding nor a record control word"), "{error}");

    }
    fn simple_template() -> Template {
        Template { mnemonic: "TEST".into(), message_type: 21,
            nodes: vec![Node { tag: "X".into(), parent: None, body: Body::Number { width: 8, scale: 1, reference: -10 } }],
            root: vec![0], first_by_tag: HashMap::from([("X".into(),0)]) }
    }
    #[test]
    fn compressed_values_missing_and_scale() {
        let t = simple_template();
        let mut e = Encoder::new();
        e.put(20,8); e.put(3,6); e.put(0,3); e.put(3,3); e.put(7,3);
        let bytes=e.finish();
        let mut w=CompressedWalker {template:&t,reader:BitReader::new(&bytes,0,bytes.len()*8),what:"test",out:vec![Subset::default();3]};
        w.walk(&t.root).unwrap();
        assert_eq!(w.out[0].leaves[0].value,Datum::Number(1.));
        assert_eq!(w.out[1].leaves[0].value,Datum::Number(1.3));
        assert_eq!(w.out[2].leaves[0].value,Datum::Missing);
    }
    #[test]
    fn compressed_constant_and_all_missing() {
        for (raw,expected) in [(20,Datum::Number(1.)),(255,Datum::Missing)] {
            let t=simple_template();let mut e=Encoder::new();e.put(raw,8);e.put(0,6);let bytes=e.finish();
            let mut w=CompressedWalker {template:&t,reader:BitReader::new(&bytes,0,bytes.len()*8),what:"test",out:vec![Subset::default();3]};
            w.walk(&t.root).unwrap();assert!(w.out.iter().all(|s|s.leaves[0].value==expected));
        }
    }
    #[test]
    fn wmo_uncompressed_subsets_are_bit_contiguous() {
        let mut t=simple_template();t.nodes[0].body=Body::Number {width:3,scale:0,reference:0};
        let mut e=Encoder::new();e.put(2,3);e.put(5,3);let data=e.finish();
        let bytes=message(21,0,(2024,8,6,18),&[348_120],2,&data);
        let env=envelope(&bytes,"test").unwrap();let out=decode_subsets(&bytes,&env,&t,"test").unwrap();
        assert_eq!(out[0].leaves[0].value,Datum::Number(2.));assert_eq!(out[1].leaves[0].value,Datum::Number(5.));
    }
    #[test]
    fn compressed_replication_counts_must_agree() {
        let mut t=simple_template();t.nodes.push(Node {tag:"GROUP".into(),parent:None,body:Body::Group {replication:Replication::Delayed8,sequence:true,children:vec![0]}});t.root=vec![1];
        let mut e=Encoder::new();e.put(1,8);e.put(2,6);e.put(0,2);e.put(1,2);let bytes=e.finish();
        let mut w=CompressedWalker {template:&t,reader:BitReader::new(&bytes,0,bytes.len()*8),what:"test",out:vec![Subset::default();2]};
        assert!(w.walk(&t.root).unwrap_err().to_string().contains("differs"));
    }

}
