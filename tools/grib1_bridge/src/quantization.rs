//! Quantization-aware bound handling shared by the GRIB2 bridges.
//!
//! Every scaled GRIB2 data-representation template reconstructs a value
//! as `Y = (R + X * 2^E) * 10^-D`.  The representable values therefore
//! sit on a grid whose spacing is `2^E * 10^-D`, and the encoder picks
//! `X` by rounding, so a quantity that is physically AT a limit --
//! saturated soil, ice-free ocean, snow-free ground -- can decode to a
//! value a fraction of one step on the wrong side of that limit.  That
//! is a statement about the packing, not about the data.
//!
//! So a bound that real data legitimately sits on is declared
//! [`BoundKind::Physical`], and a decoded value that overshoots it by no
//! more than the record's own decode quantum is clamped back onto the
//! bound, counted, and reported.  Everything else refuses exactly as
//! before: a slack plausibility range is never widened, and an
//! excursion larger than the derived tolerance is still data corruption.
//!
//! The tolerance cannot be talked upward by a record's own header.  It
//! is confined to a band that is physically negligible against the
//! field's declared range ([`MIN_RELATIVE_TOLERANCE`] ..
//! [`MAX_RELATIVE_TOLERANCE`]), so a coarsely packed record -- one whose
//! quantum is a sizeable fraction of the field's range, or one whose
//! template carries no scale at all -- buys no extra room.

/// GRIB2 data-representation template 5.4 stores IEEE-754 values
/// verbatim.  There is no reference/scale reconstruction, so there is no
/// quantization grid and the `binary_scale`/`decimal_scale` slots a
/// parser reports for it are placeholders rather than packing facts.
pub const IEEE_FLOAT_TEMPLATE: u16 = 4;

/// Smallest fraction of THE BOUND'S OWN MAGNITUDE the derived tolerance
/// may fall below.  This floor covers the templates that carry no scale
/// and the round-off of the reconstruction arithmetic itself, both of
/// which scale with the value being reconstructed.  A bound of zero
/// therefore earns no floor -- reconstructing zero is exact -- and leans
/// entirely on the record's own quantum.  Deliberately not anchored to
/// the declared range: a field with a slack sanity ceiling must not buy
/// extra room at its physical floor.
pub const MIN_RELATIVE_TOLERANCE: f64 = 1.0e-6;

/// Largest fraction of a field's declared range the derived tolerance
/// may reach.  A quantum coarser than this says the record cannot
/// resolve the bound in the first place, and the bridge would rather
/// refuse than accept a wide excursion on the encoder's word.  This one
/// is range-anchored because it only ever narrows.
///
/// 1e-3, not 1e-4: NCEP's operational GFS 0.25-degree product packs
/// SOILW at decimal scale 3, binary scale 0 (one step 0.001) with a
/// non-integer reference value, so on some records 1.0 is not on the
/// grid at all and a saturated (land-ice) cell decodes to the nearest
/// step above it.  The 2026-06-14 12Z cycle's 10-40 cm record has
/// reference 23.3813: its steps near the bound are 0.9993813 and
/// 1.0003813, and the encoder's round-to-nearest put 1.0 on the second,
/// 0.00038 past the bound and inside one step.  A 1e-4 ceiling refused
/// that operational record on every f000..f010 object, which is the
/// breakage this value removes.  What the ceiling still prevents is
/// unchanged in kind: an integer-packed unit field (one step = the
/// whole range) buys 0.001, never a wide gate.
pub const MAX_RELATIVE_TOLERANCE: f64 = 1.0e-3;

/// What kind of limit a bound is.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoundKind {
    /// A saturating physical limit that real cells sit exactly on, so
    /// the packing quantum can push them a hair past it.  Eligible for
    /// clamping.
    Physical,
    /// A slack plausibility range no real value approaches.  An
    /// excursion here is evidence about the data, never about the
    /// packing, so it always refuses.
    Sanity,
}

/// One field's declared range together with the nature of each end.
///
/// A side may be unbounded (`±f64::INFINITY`), which is why the ceiling
/// anchor is carried explicitly rather than computed from the range: a
/// field declared only "non-negative" still needs a finite magnitude to
/// measure a negligible fraction of.
#[derive(Clone, Copy, Debug)]
pub struct Bounds {
    pub minimum: f64,
    pub maximum: f64,
    pub minimum_kind: BoundKind,
    pub maximum_kind: BoundKind,
    /// The finite magnitude [`MAX_RELATIVE_TOLERANCE`] is a fraction of.
    /// A field with two declared ends uses its own span; a half-open one
    /// uses the extent its data actually occupies.
    pub scale: f64,
}

impl Bounds {
    /// A closed range, anchoring the ceiling on its own span.
    pub fn closed(
        minimum: f64,
        maximum: f64,
        minimum_kind: BoundKind,
        maximum_kind: BoundKind,
    ) -> Self {
        Self {
            minimum,
            maximum,
            minimum_kind,
            maximum_kind,
            scale: maximum - minimum,
        }
    }
}

/// A decoded value's verdict against one field's bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BoundVerdict {
    /// Inside the declared range: pass through unchanged.
    Inside,
    /// Outside a physical bound by no more than the tolerance: emit
    /// `value` (the bound itself) and record that it moved `excursion`.
    Clamped { value: f64, excursion: f64 },
    /// Outside by more than the tolerance, or outside a bound that is
    /// not physical.  Carries the tolerance that WAS on offer at that
    /// bound so the refusal can show its own arithmetic; a `Sanity` end
    /// offers zero.
    Refuse { excursion: f64, tolerance: f64 },
}

/// The decode quantum of one record: the smallest nonzero step its
/// packing can represent, `2^E * 10^-D`.
///
/// Simple (5.0), complex and spatially differenced (5.2/5.3), and the
/// image-compressed templates (5.40/5.41/5.42) all reconstruct through
/// the same reference-plus-scale expression, so they share this grid.
/// Template 5.4 has no grid and reports zero; the relative floor then
/// carries the (vanishing) tolerance.
pub fn decode_quantum(template: u16, binary_scale: i16, decimal_scale: i16) -> f64 {
    if template == IEEE_FLOAT_TEMPLATE {
        return 0.0;
    }
    let quantum =
        2.0f64.powi(i32::from(binary_scale)) * 10.0f64.powi(-i32::from(decimal_scale));
    if quantum.is_finite() && quantum > 0.0 {
        quantum
    } else {
        0.0
    }
}

impl Bounds {
    /// How far past ONE named bound a record with this quantum may sit
    /// and still be read as quantization rather than corruption.
    ///
    /// The offer is the record's own quantum, raised only to the
    /// round-off floor of the bound's own magnitude, and capped at a
    /// fraction of the declared range so a coarse packing cannot buy a
    /// wide gate.  A bound that is not physical is offered nothing.
    pub fn tolerance_at(&self, bound: f64, kind: BoundKind, quantum: f64) -> f64 {
        if kind != BoundKind::Physical {
            return 0.0;
        }
        if !self.scale.is_finite() || self.scale <= 0.0 {
            return 0.0;
        }
        let floor = MIN_RELATIVE_TOLERANCE * bound.abs();
        let ceiling = MAX_RELATIVE_TOLERANCE * self.scale;
        let offer = if !quantum.is_finite() || quantum < floor {
            floor
        } else {
            quantum
        };
        if offer > ceiling {
            ceiling
        } else {
            offer
        }
    }

    /// The tolerance offered at the lower bound.
    pub fn low_tolerance(&self, quantum: f64) -> f64 {
        self.tolerance_at(self.minimum, self.minimum_kind, quantum)
    }

    /// The tolerance offered at the upper bound.
    pub fn high_tolerance(&self, quantum: f64) -> f64 {
        self.tolerance_at(self.maximum, self.maximum_kind, quantum)
    }

    /// Judge one finite decoded value against this record's packing.
    pub fn check(&self, value: f64, quantum: f64) -> BoundVerdict {
        if value < self.minimum {
            let excursion = self.minimum - value;
            let tolerance = self.low_tolerance(quantum);
            // Subtraction at a nonzero physical bound can round an exact
            // one-quantum excursion above its tolerance. Allow only arithmetic
            // roundoff, and never grant it to a zero-tolerance/sanity bound.
            let roundoff = 4.0 * f64::EPSILON * value.abs().max(self.minimum.abs());
            if excursion <= tolerance || (tolerance > 0.0 && excursion - tolerance <= roundoff) {
                return BoundVerdict::Clamped {
                    value: self.minimum,
                    excursion,
                };
            }
            return BoundVerdict::Refuse {
                excursion,
                tolerance,
            };
        }
        if value > self.maximum {
            let excursion = value - self.maximum;
            let tolerance = self.high_tolerance(quantum);
            let roundoff = 4.0 * f64::EPSILON * value.abs().max(self.maximum.abs());
            if excursion <= tolerance || (tolerance > 0.0 && excursion - tolerance <= roundoff) {
                return BoundVerdict::Clamped {
                    value: self.maximum,
                    excursion,
                };
            }
            return BoundVerdict::Refuse {
                excursion,
                tolerance,
            };
        }
        BoundVerdict::Inside
    }
}

/// Running clamp tally for one field, carried into the receipts.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ClampTally {
    pub clamps: usize,
    pub max_excursion: f64,
}

impl ClampTally {
    pub fn record(&mut self, excursion: f64) {
        self.clamps += 1;
        if excursion > self.max_excursion {
            self.max_excursion = excursion;
        }
    }

    pub fn merge(&mut self, other: ClampTally) {
        self.clamps += other.clamps;
        if other.max_excursion > self.max_excursion {
            self.max_excursion = other.max_excursion;
        }
    }
}

/// Whether a decoded value is the coded level `target` up to the same
/// quantization tolerance, for fields whose contract is a small set of
/// exact codes rather than a range.
pub fn is_coded_value(value: f64, target: f64, tolerance: f64) -> bool {
    (value - target).abs() <= tolerance
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The packing the field report arrived with: binary scale -19,
    /// decimal scale 3.  One step of that grid is 1.9073486328125e-9.
    const REPORTED_BINARY_SCALE: i16 = -19;
    const REPORTED_DECIMAL_SCALE: i16 = 3;

    fn reported_quantum() -> f64 {
        decode_quantum(0, REPORTED_BINARY_SCALE, REPORTED_DECIMAL_SCALE)
    }

    fn unit_fraction() -> Bounds {
        Bounds::closed(0.0, 1.0, BoundKind::Physical, BoundKind::Physical)
    }

    #[test]
    fn one_quantum_roundoff_clamps_but_two_quanta_and_sanity_refuse() {
        let bounds = unit_fraction();
        assert!(matches!(bounds.check(1.0010000000000001, 0.001),
            BoundVerdict::Clamped { value: 1.0, .. }));
        assert!(matches!(bounds.check(1.002, 0.001), BoundVerdict::Refuse { .. }));
        let negative = Bounds::closed(-1.0, 0.0, BoundKind::Physical, BoundKind::Physical);
        assert!(matches!(negative.check(-1.0010000000000001, 0.001),
            BoundVerdict::Clamped { value: -1.0, .. }));
        let sanity = Bounds::closed(0.0, 1.0, BoundKind::Sanity, BoundKind::Sanity);
        assert!(matches!(sanity.check(1.0 + f64::EPSILON, 0.001), BoundVerdict::Refuse { .. }));
        assert!(matches!(bounds.check(-f64::MIN_POSITIVE, 0.0), BoundVerdict::Refuse { .. }));
    }

    #[test]
    fn the_quantum_is_the_packing_step() {
        assert_eq!(reported_quantum(), 1.9073486328125e-9);
        // The reported excursion is exactly one step of that grid, which
        // is what makes it a packing artefact rather than a data fault.
        assert_eq!(1.0f64 + reported_quantum(), 1.0000000019073487);
    }

    #[test]
    fn an_ieee_float_record_has_no_grid() {
        // Template 5.4 stores values verbatim; its scale slots are not
        // packing facts and must not be raised to a quantum of 1.0.
        assert_eq!(decode_quantum(IEEE_FLOAT_TEMPLATE, 0, 0), 0.0);
        // And a zero bound under that template is exact, so it is offered
        // nothing at all -- the floor is the bound's own magnitude.
        assert_eq!(unit_fraction().low_tolerance(0.0), 0.0);
    }

    #[test]
    fn a_coarse_quantum_cannot_buy_a_wide_gate() {
        let bounds = unit_fraction();
        // Integer packing of a [0,1] field: one step is the whole range.
        let coarse = decode_quantum(0, 0, 0);
        assert_eq!(coarse, 1.0);
        assert_eq!(bounds.high_tolerance(coarse), MAX_RELATIVE_TOLERANCE);
        assert_eq!(bounds.low_tolerance(coarse), MAX_RELATIVE_TOLERANCE);
    }

    #[test]
    fn a_slack_ceiling_does_not_loosen_the_physical_floor() {
        // Snow water equivalent: zero is physical, and 100000 kg/m2 is a
        // plausibility ceiling.  The floor's offer must come from the
        // record's own step, not from the size of that ceiling.
        let snow = Bounds::closed(0.0, 100_000.0, BoundKind::Physical, BoundKind::Sanity);
        assert_eq!(snow.low_tolerance(reported_quantum()), reported_quantum());
        assert_eq!(snow.high_tolerance(reported_quantum()), 0.0);
    }

    #[test]
    fn saturated_soil_clamps_and_a_wet_impossibility_refuses() {
        let bounds = unit_fraction();
        let quantum = reported_quantum();
        match bounds.check(1.0000000019073487, quantum) {
            BoundVerdict::Clamped { value, excursion } => {
                assert_eq!(value, 1.0);
                assert!(excursion > 0.0, "{excursion}");
                assert!(excursion <= bounds.high_tolerance(quantum), "{excursion}");
            }
            other => panic!("saturated soil must clamp, got {other:?}"),
        }
        assert!(matches!(
            bounds.check(1.05, quantum),
            BoundVerdict::Refuse { .. }
        ));
        assert_eq!(bounds.check(0.5, quantum), BoundVerdict::Inside);
    }

    #[test]
    fn a_zero_bound_kissed_from_below_clamps_by_exactly_one_step() {
        let bounds = unit_fraction();
        let quantum = reported_quantum();
        // An encoder whose reference value rounds a hair below zero puts
        // every dry cell just outside the bound.
        match bounds.check(-quantum, quantum) {
            BoundVerdict::Clamped { value, excursion } => {
                assert_eq!(value, 0.0);
                assert_eq!(excursion, quantum);
            }
            other => panic!("a dry cell must clamp, got {other:?}"),
        }
        // One step is the whole offer at a zero bound: two is corruption.
        assert!(matches!(
            bounds.check(-2.0 * quantum, quantum),
            BoundVerdict::Refuse { .. }
        ));
    }

    #[test]
    fn a_sanity_bound_is_never_widened() {
        // Relative humidity: zero is physical, the 110 % ceiling is a
        // plausibility range and stays exact.
        let bounds = Bounds::closed(0.0, 110.0, BoundKind::Physical, BoundKind::Sanity);
        let quantum = reported_quantum();
        match bounds.check(110.0 + quantum / 2.0, quantum) {
            BoundVerdict::Refuse { tolerance, .. } => assert_eq!(tolerance, 0.0),
            other => panic!("a sanity ceiling must refuse, got {other:?}"),
        }
        assert!(matches!(
            bounds.check(-quantum / 2.0, quantum),
            BoundVerdict::Clamped { value: 0.0, .. }
        ));
    }

    #[test]
    fn a_refusal_reports_the_offer_it_exceeded() {
        let bounds = unit_fraction();
        match bounds.check(1.05, reported_quantum()) {
            BoundVerdict::Refuse {
                excursion,
                tolerance,
            } => {
                assert!(excursion > tolerance);
                assert_eq!(tolerance, MIN_RELATIVE_TOLERANCE);
            }
            other => panic!("1.05 must refuse, got {other:?}"),
        }
    }

    #[test]
    fn the_tally_keeps_the_worst_excursion() {
        let mut tally = ClampTally::default();
        tally.record(1.0e-9);
        tally.record(4.0e-9);
        tally.record(2.0e-9);
        assert_eq!(tally.clamps, 3);
        assert_eq!(tally.max_excursion, 4.0e-9);
        let mut total = ClampTally::default();
        total.merge(tally);
        total.merge(ClampTally {
            clamps: 1,
            max_excursion: 9.0e-9,
        });
        assert_eq!(total.clamps, 4);
        assert_eq!(total.max_excursion, 9.0e-9);
    }

    #[test]
    fn operational_gfs_soilw_packing_clamps_a_saturated_cell() {
        // gfs.t12z.pgrb2.0p25 of 2026-06-14, SOILW 10-40 cm: template
        // 5.3, E = 0, D = 3, R = 23.3813.  The grid near the bound is
        // (23.3813 + X) / 1000, so 1.0 encodes as X = 977 and decodes to
        // 1.0003813 -- the refusal that stopped every GFS start.
        let bounds = unit_fraction();
        let quantum = decode_quantum(3, 0, 3);
        assert_eq!(quantum, 1.0e-3);
        let decoded = (23.3813f64 + 977.0) / 1000.0;
        match bounds.check(decoded, quantum) {
            BoundVerdict::Clamped { value, excursion } => {
                assert_eq!(value, 1.0);
                assert!(excursion < quantum / 2.0, "{excursion}");
            }
            other => panic!("a saturated GFS cell must clamp, got {other:?}"),
        }
        // Two steps past the bound is not packing, and still refuses.
        assert!(matches!(
            bounds.check(1.0 + 2.0 * quantum, quantum),
            BoundVerdict::Refuse { .. }
        ));
    }

    #[test]
    fn a_degenerate_range_grants_nothing() {
        let point = Bounds::closed(1.0, 1.0, BoundKind::Physical, BoundKind::Physical);
        assert_eq!(point.high_tolerance(1.0e-9), 0.0);
        assert!(matches!(
            point.check(1.0 + f64::EPSILON, 1.0e-9),
            BoundVerdict::Refuse { .. }
        ));
    }
}
