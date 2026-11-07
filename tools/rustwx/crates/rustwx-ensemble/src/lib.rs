//! Ensemble product mathematics: mean, spread, member maximum,
//! neighbourhood exceedance probability (NMEP), probability-matched
//! mean, and the missingness
//! bookkeeping every one of them publishes.
//!
//! This is a transcription of `gpuwm/da/enprod.py`'s reduction layer, whose
//! two policy documents (`NAN_POLICIES`, `PMM_TIE_RULES`) are reproduced
//! here verbatim in substance because they ARE the contract -- each one was
//! written after the implicit behaviour it replaces produced a specific
//! wrong number, and each of those wrong numbers is a test below.
//!
//! No I/O, no rendering, no model or case identity.  A grid is `ny * nx`
//! row-major `f64`; a member stack is `n` such grids plus the members' own
//! NUMBERS (not their positions), because colour and blame both key on the
//! number.
//!
//! ## The masking policy in one sentence
//!
//! A non-finite value -- NaN and both infinities alike -- is not data: it
//! is excluded from every reduction at that grid point AND the denominator
//! shrinks with it, and the denominator that was used is published
//! ([`MissingnessReport`]) so a masked statistic cannot hide how much of it
//! was masked.

use std::fmt;

/// What to do with a non-finite member value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NanPolicy {
    /// Masked AND counted: excluded from the reduction at that point, with
    /// the denominator shrinking to match, and the coverage published.
    #[default]
    Mask,
    /// Fail closed: any non-finite value anywhere refuses the product,
    /// naming the members at fault.
    Refuse,
}

impl NanPolicy {
    pub fn parse(value: &str) -> Result<Self, EnsembleError> {
        match value {
            "mask" => Ok(Self::Mask),
            "refuse" => Ok(Self::Refuse),
            other => Err(EnsembleError::Policy(format!(
                "unknown nan policy {other:?}; choose from mask, refuse"
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mask => "mask",
            Self::Refuse => "refuse",
        }
    }
}

/// How the probability-matched mean breaks ties in the ensemble mean.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PmmTieRule {
    /// Ebert's algorithm exactly: the pooled intensity distribution is
    /// preserved value for value and tied points receive their pooled
    /// values in flat-index order.  Deterministic, and the order carries
    /// no information -- which is why [`pmm_tie_report`] exists.
    #[default]
    FlatIndex,
    /// Every point in a tie group receives the mean of that group's pooled
    /// values.  No artificial gradient, at the cost of the exact pooled
    /// distribution.
    Average,
}

impl PmmTieRule {
    pub fn parse(value: &str) -> Result<Self, EnsembleError> {
        match value {
            "flat-index" => Ok(Self::FlatIndex),
            "average" => Ok(Self::Average),
            other => Err(EnsembleError::Policy(format!(
                "unknown pmm tie rule {other:?}; choose from flat-index, average"
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::FlatIndex => "flat-index",
            Self::Average => "average",
        }
    }
}

/// A fail-closed refusal, or a malformed request.  Every message names the
/// members at fault: "some members are missing" is not a diagnosis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnsembleError {
    /// The stack itself is not what was claimed (ragged, empty, mis-shaped).
    Shape(String),
    /// A policy name this build does not implement.
    Policy(String),
    /// `--nan-policy refuse` and the stack has non-finite values.
    Refused(String),
}

impl fmt::Display for EnsembleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(message) | Self::Policy(message) | Self::Refused(message) => {
                f.write_str(message)
            }
        }
    }
}

impl std::error::Error for EnsembleError {}

/// `n` member grids on one `ny * nx` mesh, each carrying its own member
/// NUMBER.
///
/// The numbers are stored rather than derived from position because
/// [`member_color`] keys on them: a paintball plot of members 1..30 and one
/// of members 3, 7, 11 must give member 7 the same colour, or comparing two
/// paintball plots is actively misleading.
#[derive(Debug, Clone, PartialEq)]
pub struct MemberStack {
    ny: usize,
    nx: usize,
    numbers: Vec<u32>,
    values: Vec<Vec<f64>>,
}

impl MemberStack {
    pub fn new(ny: usize, nx: usize, members: Vec<(u32, Vec<f64>)>) -> Result<Self, EnsembleError> {
        if ny == 0 || nx == 0 {
            return Err(EnsembleError::Shape(format!(
                "a member stack needs a non-empty grid; got {ny}x{nx}"
            )));
        }
        if members.is_empty() {
            return Err(EnsembleError::Shape(
                "a member stack needs at least one member".to_string(),
            ));
        }
        let points = ny * nx;
        let mut numbers = Vec::with_capacity(members.len());
        let mut values = Vec::with_capacity(members.len());
        for (number, grid) in members {
            if grid.len() != points {
                return Err(EnsembleError::Shape(format!(
                    "member {number} carries {} value(s); the grid is {ny}x{nx} = {points}",
                    grid.len()
                )));
            }
            numbers.push(number);
            values.push(grid);
        }
        Ok(Self {
            ny,
            nx,
            numbers,
            values,
        })
    }

    pub fn ny(&self) -> usize {
        self.ny
    }

    pub fn nx(&self) -> usize {
        self.nx
    }

    pub fn points(&self) -> usize {
        self.ny * self.nx
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn numbers(&self) -> &[u32] {
        &self.numbers
    }

    pub fn member(&self, index: usize) -> &[f64] {
        &self.values[index]
    }

    pub fn members(&self) -> impl Iterator<Item = (u32, &[f64])> {
        self.numbers
            .iter()
            .copied()
            .zip(self.values.iter().map(Vec::as_slice))
    }

    /// Apply the nan policy's gate.  Under [`NanPolicy::Refuse`] this is
    /// where the refusal happens, once, before any reduction runs.
    fn gate(&self, policy: NanPolicy) -> Result<(), EnsembleError> {
        if policy != NanPolicy::Refuse {
            return Ok(());
        }
        let mut bad = 0usize;
        let mut named: Vec<String> = Vec::new();
        for (number, grid) in self.members() {
            let count = grid.iter().filter(|value| !value.is_finite()).count();
            if count > 0 {
                bad += count;
                named.push(format!("member {number}: {count}"));
            }
        }
        if bad == 0 {
            return Ok(());
        }
        Err(EnsembleError::Refused(format!(
            "--nan-policy refuse: {bad} non-finite value(s) in the member stack \
             ({}). Nothing was drawn. Use --nan-policy mask to exclude them from \
             the reductions and stamp the coverage on the panel instead.",
            named.join(", ")
        )))
    }
}

/// What is missing from a member stack, and how much is left.
///
/// A masked reduction that does not publish its denominator is exactly the
/// thing the propagate-NaN policy was right to refuse; publishing it is
/// what makes masking accurate.
#[derive(Debug, Clone, PartialEq)]
pub struct MissingnessReport {
    pub policy: &'static str,
    pub members: usize,
    pub grid_points: usize,
    pub nonfinite_values: usize,
    pub members_affected: Vec<u32>,
    pub min_finite_members: usize,
    pub fully_missing_points: usize,
    /// Fraction of the (members x points) cells that carried data.
    pub coverage: f64,
}

impl MissingnessReport {
    /// One line for the panel, or `None` when nothing was missing.
    pub fn caption(&self) -> Option<String> {
        if self.nonfinite_values == 0 {
            return None;
        }
        let shown = self
            .members_affected
            .iter()
            .take(6)
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let shown = if self.members_affected.len() > 6 {
            format!("{shown}, +{} more", self.members_affected.len() - 6)
        } else {
            shown
        };
        let tail = if self.fully_missing_points == 0 {
            String::new()
        } else {
            format!(
                "; {} point(s) blank (no finite member)",
                self.fully_missing_points
            )
        };
        Some(format!(
            "MASKED: {} non-finite value(s) from member(s) {shown}; coverage \
             {:.1}% of member-points, min {} of {} members per point{tail}",
            self.nonfinite_values,
            100.0 * self.coverage,
            self.min_finite_members,
            self.members,
        ))
    }
}

pub fn missingness_report(stack: &MemberStack, policy: NanPolicy) -> Result<MissingnessReport, EnsembleError> {
    stack.gate(policy)?;
    let points = stack.points();
    let mut per_point = vec![0usize; points];
    let mut affected = Vec::new();
    let mut finite_total = 0usize;
    for (number, grid) in stack.members() {
        let mut complete = true;
        for (index, value) in grid.iter().enumerate() {
            if value.is_finite() {
                per_point[index] += 1;
                finite_total += 1;
            } else {
                complete = false;
            }
        }
        if !complete {
            affected.push(number);
        }
    }
    let total = points * stack.len();
    Ok(MissingnessReport {
        policy: policy.as_str(),
        members: stack.len(),
        grid_points: points,
        nonfinite_values: total - finite_total,
        members_affected: affected,
        min_finite_members: per_point.iter().copied().min().unwrap_or(0),
        fully_missing_points: per_point.iter().filter(|count| **count == 0).count(),
        coverage: if total == 0 {
            0.0
        } else {
            finite_total as f64 / total as f64
        },
    })
}

/// Arithmetic mean over the member axis, under the stated NaN policy.
///
/// `Mask` averages the finite members at each point and leaves points with
/// no finite member NaN.  EVERY non-finite value is excluded, infinities
/// included: `nansum` ignores NaN and PROPAGATES infinity, so a stack of
/// `[1.0, 2.0, +Inf]` reported two-thirds coverage on a panel whose mean
/// was `+Inf`.  A product that contradicts its own coverage stamp is worse
/// than one that has none.
pub fn ensemble_mean(stack: &MemberStack, policy: NanPolicy) -> Result<Vec<f64>, EnsembleError> {
    stack.gate(policy)?;
    let mut out = vec![f64::NAN; stack.points()];
    for index in 0..stack.points() {
        let mut sum = 0.0;
        let mut count = 0usize;
        for grid in &stack.values {
            let value = grid[index];
            if value.is_finite() {
                sum += value;
                count += 1;
            }
        }
        out[index] = if count > 0 {
            sum / count as f64
        } else {
            f64::NAN
        };
    }
    Ok(out)
}

/// Member maximum at each grid point, under the stated NaN policy.
///
/// The "ensemble max" map: the strongest value any member has at each
/// point (max composite reflectivity, max updraft helicity).  `Mask`
/// takes the maximum over the finite members only -- an infinity is no
/// more data here than in the mean -- and leaves a point no member has
/// NaN, never `-inf` and never zero.
pub fn ensemble_max(stack: &MemberStack, policy: NanPolicy) -> Result<Vec<f64>, EnsembleError> {
    stack.gate(policy)?;
    let mut out = vec![f64::NAN; stack.points()];
    for (index, slot) in out.iter_mut().enumerate() {
        for grid in &stack.values {
            let value = grid[index];
            if value.is_finite() && (slot.is_nan() || value > *slot) {
                *slot = value;
            }
        }
    }
    Ok(out)
}

/// Ensemble standard deviation -- the SAMPLE one (`ddof = 1`) by default.
///
/// The ensemble is a sample from the forecast distribution, not the
/// population.  Under `Mask` the `n > ddof` minimum is applied PER POINT: a
/// point with one finite member is NaN, not zero, because reporting 0.0
/// there would read as perfect agreement.
pub fn ensemble_spread(
    stack: &MemberStack,
    ddof: usize,
    policy: NanPolicy,
) -> Result<Vec<f64>, EnsembleError> {
    stack.gate(policy)?;
    if stack.len() <= ddof {
        return Err(EnsembleError::Shape(format!(
            "ensemble spread with ddof={ddof} needs more than {ddof} member(s); \
             this ensemble has {}",
            stack.len()
        )));
    }
    let mean = ensemble_mean(stack, policy)?;
    let mut out = vec![f64::NAN; stack.points()];
    for index in 0..stack.points() {
        let mut count = 0usize;
        let mut squares = 0.0;
        for grid in &stack.values {
            let value = grid[index];
            if value.is_finite() {
                count += 1;
                let deviation = value - mean[index];
                squares += deviation * deviation;
            }
        }
        out[index] = if count > ddof {
            (squares / (count - ddof) as f64).sqrt()
        } else {
            f64::NAN
        };
    }
    Ok(out)
}

/// Integer `(dy, dx)` offsets inside a disc of the given radius, in cells.
///
/// Nested by construction: every offset inside radius r is inside radius
/// R >= r.  That nesting is what makes neighbourhood probability monotone
/// in the radius, so it is a property of this function and not a
/// coincidence of the caller.
pub fn disc_offsets(radius_cells: f64) -> Result<Vec<(i64, i64)>, EnsembleError> {
    if !radius_cells.is_finite() || radius_cells < 0.0 {
        return Err(EnsembleError::Shape(format!(
            "radius must be finite and >= 0, got {radius_cells}"
        )));
    }
    let limit = radius_cells.floor() as i64;
    let squared = radius_cells * radius_cells;
    let mut offsets = Vec::new();
    for dy in -limit..=limit {
        for dx in -limit..=limit {
            if (dy * dy + dx * dx) as f64 <= squared {
                offsets.push((dy, dx));
            }
        }
    }
    Ok(offsets)
}

/// Disc offsets that can reach any in-domain cell of an `ny * nx` grid.
///
/// A radius larger than the domain is a legitimate request (it means "the
/// whole domain"), so the disc is CLIPPED rather than the request refused.
fn in_domain_offsets(radius_cells: f64, ny: usize, nx: usize) -> Result<Vec<(i64, i64)>, EnsembleError> {
    Ok(disc_offsets(radius_cells)?
        .into_iter()
        .filter(|(dy, dx)| dy.unsigned_abs() < ny as u64 && dx.unsigned_abs() < nx as u64)
        .collect())
}

/// How many IN-DOMAIN cells each point's disc actually covers.
///
/// Exists so "the maximum is over the clipped disc" is a checkable
/// structural claim and not only an assertion about output values: for a
/// maximum operator, replicating an edge value outward can never change an
/// in-domain result, so no comparison of maxima can tell clipping from
/// replicate padding.  Counting the footprint can.
pub fn neighborhood_footprint(
    ny: usize,
    nx: usize,
    radius_cells: f64,
) -> Result<Vec<usize>, EnsembleError> {
    let mut out = vec![0usize; ny * nx];
    if radius_cells <= 0.0 {
        return Ok(vec![1usize; ny * nx]);
    }
    for (dy, dx) in in_domain_offsets(radius_cells, ny, nx)? {
        for_each_shifted(ny, nx, dy, dx, |dst, _src| out[dst] += 1);
    }
    Ok(out)
}

/// Walk the overlapping destination/source index pairs for one disc offset.
fn for_each_shifted<F: FnMut(usize, usize)>(ny: usize, nx: usize, dy: i64, dx: i64, mut body: F) {
    let y_lo = (-dy).max(0) as usize;
    let y_hi = ny - dy.max(0) as usize;
    let x_lo = (-dx).max(0) as usize;
    let x_hi = nx - dx.max(0) as usize;
    for y in y_lo..y_hi {
        let sy = (y as i64 + dy) as usize;
        for x in x_lo..x_hi {
            let sx = (x as i64 + dx) as usize;
            body(y * nx + x, sy * nx + sx);
        }
    }
}

/// Maximum over a disc of `radius_cells` around every grid point.
///
/// Beyond the domain edge there is no data, so the maximum is taken over
/// the in-domain part of the disc only -- NOT over edge values replicated
/// outward, which would invent exceedances in the boundary rows.
///
/// Under `Mask` a non-finite cell contributes nothing to its neighbours'
/// maxima and a point whose whole clipped disc is non-finite comes back
/// NaN.  Letting one NaN propagate across a neighbourhood, as a plain
/// elementwise maximum does, turned a hit into a miss at every point that
/// could see the bad cell.
pub fn neighborhood_max(
    field: &[f64],
    ny: usize,
    nx: usize,
    radius_cells: f64,
    policy: NanPolicy,
) -> Result<Vec<f64>, EnsembleError> {
    if field.len() != ny * nx {
        return Err(EnsembleError::Shape(format!(
            "field carries {} value(s); the grid is {ny}x{nx}",
            field.len()
        )));
    }
    if radius_cells <= 0.0 {
        return Ok(field.to_vec());
    }
    let masking = policy != NanPolicy::Refuse;
    let source: Vec<f64> = if masking {
        field
            .iter()
            .map(|value| if value.is_finite() { *value } else { f64::NEG_INFINITY })
            .collect()
    } else {
        field.to_vec()
    };
    let mut out = vec![f64::NEG_INFINITY; field.len()];
    for (dy, dx) in in_domain_offsets(radius_cells, ny, nx)? {
        for_each_shifted(ny, nx, dy, dx, |dst, src| {
            if source[src] > out[dst] {
                out[dst] = source[src];
            }
        });
    }
    if masking {
        // NEG_INFINITY survives only where every in-disc cell was
        // non-finite, and that is genuinely "no data here", not "very
        // small".
        for value in &mut out {
            if *value == f64::NEG_INFINITY {
                *value = f64::NAN;
            }
        }
    }
    Ok(out)
}

/// Strict below-threshold probability. Missing members mask the cell instead
/// of silently shrinking the roster. A neighborhood is a per-member minimum.
pub fn below_threshold_probability(
    stack: &MemberStack,
    threshold: f64,
    radius_cells: f64,
    policy: NanPolicy,
) -> Result<Vec<f64>, EnsembleError> {
    let negative = MemberStack::new(
        stack.ny, stack.nx,
        stack.numbers.iter().copied().zip(stack.values.iter()
            .map(|v| v.iter().map(|x| -x).collect())).collect(),
    )?;
    let mut result = exceedance_probability(&negative, -threshold, radius_cells, policy)?;
    for (i, v) in result.iter_mut().enumerate() {
        if stack.values.iter().any(|grid| !grid[i].is_finite()) {
            *v = f64::NAN;
        }
    }
    Ok(result)
}

/// Fraction of members exceeding `threshold`; strictly greater.
///
/// With a radius this is the neighbourhood-maximum ensemble probability
/// (NMEP): each member is FIRST reduced to its neighbourhood maximum, and
/// the ensemble fraction is taken of THAT.  The order matters -- taking the
/// neighbourhood maximum of the probability field instead would let one
/// member's hit be reported at full ensemble confidence.
///
/// Under `Mask` the DENOMINATOR at a point is the number of members with a
/// finite value AT THAT POINT -- its own value, not its neighbourhood's.
/// Fixing the voting roster to the point rather than to the disc keeps the
/// denominator independent of the radius, which is what makes the
/// advertised monotonicity in the radius EXACT rather than approximate.
pub fn exceedance_probability(
    stack: &MemberStack,
    threshold: f64,
    radius_cells: f64,
    policy: NanPolicy,
) -> Result<Vec<f64>, EnsembleError> {
    stack.gate(policy)?;
    let points = stack.points();
    let mut hits = vec![0usize; points];
    let mut votes = vec![0usize; points];
    for grid in &stack.values {
        let reduced = if radius_cells > 0.0 {
            neighborhood_max(grid, stack.ny, stack.nx, radius_cells, policy)?
        } else {
            grid.clone()
        };
        for index in 0..points {
            if policy == NanPolicy::Refuse {
                votes[index] += 1;
                if reduced[index] > threshold {
                    hits[index] += 1;
                }
            } else if grid[index].is_finite() {
                votes[index] += 1;
                if reduced[index].is_finite() && reduced[index] > threshold {
                    hits[index] += 1;
                }
            }
        }
    }
    Ok((0..points)
        .map(|index| {
            if votes[index] > 0 {
                hits[index] as f64 / votes[index] as f64
            } else {
                f64::NAN
            }
        })
        .collect())
}

/// Exact probability-matched mean (Ebert 2001).
///
/// 1. the ensemble mean supplies the spatial PATTERN;
/// 2. every member value is pooled and sorted, and evenly spaced draws are
///    taken, yielding one value per assignable grid point drawn from the
///    pooled intensity distribution -- which averaging destroys;
/// 3. the pooled values are laid onto the grid in the mean's rank order.
///
/// **Missingness stays where it is.**  A point with no finite member is NaN
/// in the output and takes no pooled value.  Sorting NaNs into the pool
/// instead assigned the pooled NaN to the highest finite mean point: the
/// two-member probe `[[100, 1, NaN], [90, 2, 0]]` came back `[NaN, 90, 1]`,
/// erasing the panel's strongest feature and filling the actually-invalid
/// point with a finite number.
pub fn probability_matched_mean(
    stack: &MemberStack,
    policy: NanPolicy,
    tie_rule: PmmTieRule,
) -> Result<Vec<f64>, EnsembleError> {
    stack.gate(policy)?;
    let mean = ensemble_mean(stack, policy)?;
    let mut out = vec![f64::NAN; mean.len()];
    let finite_index: Vec<usize> = (0..mean.len()).filter(|i| mean[*i].is_finite()).collect();
    let n_assign = finite_index.len();
    if n_assign == 0 {
        return Ok(out);
    }
    let mut pool: Vec<f64> = stack
        .values
        .iter()
        .flat_map(|grid| grid.iter().copied())
        .filter(|value| value.is_finite())
        .collect();
    if pool.is_empty() {
        return Ok(out);
    }
    // Descending.  `total_cmp` rather than `partial_cmp().unwrap()`: the
    // pool is already filtered to finite values, and a total order cannot
    // panic on one that slips through a future edit.
    pool.sort_by(|a, b| b.total_cmp(a));

    // Evenly spaced draws from the pooled distribution: with a complete
    // stack this is exactly indices 0, M, 2M, ... of the descending pool.
    let stride = pool.len() as f64 / n_assign as f64;
    let mut picked: Vec<f64> = (0..n_assign)
        .map(|rank| {
            let pick = (rank as f64 * stride).floor() as usize;
            pool[pick.min(pool.len() - 1)]
        })
        .collect();

    // The mean's descending rank order, stable so equal means keep flat
    // index order (that IS the flat-index tie rule).
    let mut order: Vec<usize> = (0..n_assign).collect();
    order.sort_by(|a, b| {
        mean[finite_index[*b]]
            .total_cmp(&mean[finite_index[*a]])
            .then(a.cmp(b))
    });

    if tie_rule == PmmTieRule::Average {
        let ranked: Vec<f64> = order.iter().map(|i| mean[finite_index[*i]]).collect();
        let mut start = 0usize;
        while start < ranked.len() {
            let mut end = start + 1;
            while end < ranked.len() && ranked[end] == ranked[start] {
                end += 1;
            }
            if end - start > 1 {
                let mean_of_group: f64 =
                    picked[start..end].iter().sum::<f64>() / (end - start) as f64;
                for slot in &mut picked[start..end] {
                    *slot = mean_of_group;
                }
            }
            start = end;
        }
    }

    for (rank, position) in order.into_iter().enumerate() {
        out[finite_index[position]] = picked[rank];
    }
    Ok(out)
}

/// How much of the mean field is a plateau, and how big the worst one is.
///
/// A flat-index tie break is deterministic but paints an artificial
/// row-major gradient across every plateau, so a reader needs to know
/// whether the panel has one.  Reported rather than silently tolerated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PmmTieReport {
    pub tied_points: usize,
    pub largest_tie_group: usize,
    pub tied_fraction: f64,
}

pub fn pmm_tie_report(stack: &MemberStack, policy: NanPolicy) -> Result<PmmTieReport, EnsembleError> {
    let mean = ensemble_mean(stack, policy)?;
    let mut finite: Vec<f64> = mean.into_iter().filter(|value| value.is_finite()).collect();
    if finite.is_empty() {
        return Ok(PmmTieReport {
            tied_points: 0,
            largest_tie_group: 0,
            tied_fraction: 0.0,
        });
    }
    let total = finite.len();
    finite.sort_by(f64::total_cmp);
    let mut tied = 0usize;
    let mut largest = 1usize;
    let mut start = 0usize;
    while start < finite.len() {
        let mut end = start + 1;
        while end < finite.len() && finite[end] == finite[start] {
            end += 1;
        }
        let size = end - start;
        if size > 1 {
            tied += size;
        }
        largest = largest.max(size);
        start = end;
    }
    Ok(PmmTieReport {
        tied_points: tied,
        largest_tie_group: largest,
        tied_fraction: tied as f64 / total as f64,
    })
}

/// Paintball member colours: 40 entries, `#rrggbb`.
///
/// Held as literals so member -> colour assignment is testable without any
/// plotting library installed and cannot shift under a library release.
pub const PAINTBALL_PALETTE: [&str; 40] = [
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896",
    "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7",
    "#bcbd22", "#dbdb8d", "#17becf", "#9edae5", "#393b79", "#5254a3", "#6b6ecf", "#9c9ede",
    "#637939", "#8ca252", "#b5cf6b", "#cedb9c", "#8c6d31", "#bd9e39", "#e7ba52", "#e7cb94",
    "#843c39", "#ad494a", "#d6616b", "#e7969c", "#7b4173", "#a55194", "#ce6dbd", "#de9ed6",
];

/// The paintball colour for a member NUMBER -- not for its position.
pub fn member_color(member_number: u32) -> &'static str {
    PAINTBALL_PALETTE[(member_number as usize) % PAINTBALL_PALETTE.len()]
}

/// Probability shading ladder: 10% steps, with zero left transparent so the
/// plot shows where the ensemble said something, not where it did not.
pub fn probability_levels() -> Vec<f64> {
    (1..=10).map(|step| f64::from(step) / 10.0).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stack(ny: usize, nx: usize, members: &[&[f64]]) -> MemberStack {
        MemberStack::new(
            ny,
            nx,
            members
                .iter()
                .enumerate()
                .map(|(index, grid)| (index as u32 + 1, grid.to_vec()))
                .collect(),
        )
        .expect("well-formed probe stack")
    }

    // ---- shape/refusal -----------------------------------------------

    #[test]
    fn a_ragged_stack_is_refused_by_the_member_that_is_wrong() {
        let error = MemberStack::new(1, 3, vec![(1, vec![1.0, 2.0, 3.0]), (7, vec![1.0])])
            .expect_err("a two-value member on a three-point grid is not a stack");
        let message = error.to_string();
        assert!(message.contains("member 7"), "{message}");
        assert!(message.contains("1x3"), "{message}");
    }

    #[test]
    fn refuse_names_every_member_at_fault_and_the_way_out() {
        let probe = stack(1, 3, &[&[1.0, f64::NAN, 3.0], &[1.0, 2.0, 3.0], &[f64::INFINITY, 2.0, 3.0]]);
        let error = ensemble_mean(&probe, NanPolicy::Refuse).expect_err("must refuse");
        let message = error.to_string();
        assert!(message.contains("member 1: 1"), "{message}");
        assert!(message.contains("member 3: 1"), "{message}");
        assert!(!message.contains("member 2"), "{message}");
        assert!(message.contains("--nan-policy mask"), "{message}");
    }

    // ---- mean --------------------------------------------------------

    #[test]
    fn the_mean_excludes_infinity_as_well_as_nan() {
        // The exact stack the docstring names: coverage said two thirds
        // while the mean it stamped that on was +Inf.
        let probe = stack(1, 1, &[&[1.0], &[2.0], &[f64::INFINITY]]);
        let mean = ensemble_mean(&probe, NanPolicy::Mask).unwrap();
        assert_eq!(mean, vec![1.5]);
        let report = missingness_report(&probe, NanPolicy::Mask).unwrap();
        assert_eq!(report.nonfinite_values, 1);
        assert_eq!(report.min_finite_members, 2);
    }

    #[test]
    fn a_point_no_member_has_is_nan_not_zero() {
        let probe = stack(1, 2, &[&[1.0, f64::NAN], &[3.0, f64::NAN]]);
        let mean = ensemble_mean(&probe, NanPolicy::Mask).unwrap();
        assert_eq!(mean[0], 2.0);
        assert!(mean[1].is_nan());
        let report = missingness_report(&probe, NanPolicy::Mask).unwrap();
        assert_eq!(report.fully_missing_points, 1);
        assert_eq!(report.members_affected, vec![1, 2]);
        assert!(report.caption().unwrap().contains("1 point(s) blank"));
    }

    // ---- max ---------------------------------------------------------

    #[test]
    fn the_max_skips_a_nan_member_and_leaves_an_empty_point_nan() {
        // Member 2 is missing at point 0 and member 3 is +Inf there; the
        // max is the finite members' max, not NaN and not infinity.
        let probe = stack(
            1,
            3,
            &[
                &[10.0, -5.0, f64::NAN],
                &[f64::NAN, -2.0, f64::NAN],
                &[f64::INFINITY, -7.0, f64::NAN],
            ],
        );
        let max = ensemble_max(&probe, NanPolicy::Mask).unwrap();
        assert_eq!(max[0], 10.0);
        assert_eq!(max[1], -2.0, "an all-negative point keeps its sign");
        assert!(max[2].is_nan(), "a point no member has is NaN");
        let refused = ensemble_max(&probe, NanPolicy::Refuse).unwrap_err();
        assert!(matches!(refused, EnsembleError::Refused(_)), "{refused}");
    }

    // ---- spread ------------------------------------------------------

    #[test]
    fn spread_is_the_sample_deviation_and_needs_more_members_than_ddof() {
        let probe = stack(1, 1, &[&[1.0], &[3.0]]);
        let spread = ensemble_spread(&probe, 1, NanPolicy::Mask).unwrap();
        // ddof=1 over {1,3}: mean 2, sum sq 2, /1 -> sqrt(2)
        assert!((spread[0] - 2.0f64.sqrt()).abs() < 1e-12, "{spread:?}");
        let one = stack(1, 1, &[&[1.0]]);
        assert!(ensemble_spread(&one, 1, NanPolicy::Mask).is_err());
    }

    #[test]
    fn a_point_with_one_finite_member_has_no_spread_rather_than_zero() {
        let probe = stack(1, 2, &[&[1.0, 5.0], &[3.0, f64::NAN]]);
        let spread = ensemble_spread(&probe, 1, NanPolicy::Mask).unwrap();
        assert!((spread[0] - 2.0f64.sqrt()).abs() < 1e-12);
        assert!(spread[1].is_nan(), "one finite member is not agreement");
    }

    #[test]
    fn spread_does_not_go_infinite_where_coverage_says_partial() {
        // (+Inf - mean)^2 is +Inf, and a nan-ignoring sum keeps it.
        let probe = stack(1, 1, &[&[1.0], &[3.0], &[f64::INFINITY]]);
        let spread = ensemble_spread(&probe, 1, NanPolicy::Mask).unwrap();
        assert!(spread[0].is_finite(), "{spread:?}");
        assert!((spread[0] - 2.0f64.sqrt()).abs() < 1e-12);
    }

    // ---- neighbourhood ------------------------------------------------

    #[test]
    fn disc_offsets_nest_as_the_radius_grows() {
        for (small, large) in [(0.0, 1.0), (1.0, 1.5), (1.5, 3.0), (3.0, 4.2)] {
            let inner = disc_offsets(small).unwrap();
            let outer = disc_offsets(large).unwrap();
            for offset in &inner {
                assert!(outer.contains(offset), "{offset:?} left the disc at {large}");
            }
        }
        assert!(disc_offsets(f64::NAN).is_err());
        assert!(disc_offsets(-1.0).is_err());
    }

    #[test]
    fn the_footprint_proves_the_disc_is_clipped_not_replicated() {
        // A maximum cannot distinguish clipping from edge replication, so
        // the structural claim is checked by counting instead.
        let footprint = neighborhood_footprint(3, 3, 1.0).unwrap();
        // radius 1 disc is 5 cells; a corner sees 3, an edge 4, the centre 5.
        assert_eq!(footprint[0], 3);
        assert_eq!(footprint[1], 4);
        assert_eq!(footprint[4], 5);
    }

    #[test]
    fn a_radius_wider_than_the_domain_means_the_whole_domain() {
        // The broadcast crash: an offset further out than the grid is wide.
        let footprint = neighborhood_footprint(2, 5, 9.0).unwrap();
        assert!(footprint.iter().all(|count| *count == 10), "{footprint:?}");
    }

    #[test]
    fn one_bad_cell_does_not_erase_its_neighbours_maxima() {
        let field = vec![10.0, f64::NAN, 0.0];
        let out = neighborhood_max(&field, 1, 3, 1.0, NanPolicy::Mask).unwrap();
        assert_eq!(out, vec![10.0, 10.0, 0.0]);
    }

    // ---- NMEP ----------------------------------------------------------

    #[test]
    fn the_documented_probe_stays_a_hit_when_the_radius_grows() {
        // [10, NaN, 0] at threshold 5 gave [1,0,0] at radius 0 and
        // [0,0,0] at radius 1 -- the defect this policy exists for.
        let probe = stack(1, 3, &[&[10.0, f64::NAN, 0.0]]);
        let r0 = exceedance_probability(&probe, 5.0, 0.0, NanPolicy::Mask).unwrap();
        assert_eq!(r0[0], 1.0);
        assert!(r0[1].is_nan(), "no member votes here");
        assert_eq!(r0[2], 0.0);
        let r1 = exceedance_probability(&probe, 5.0, 1.0, NanPolicy::Mask).unwrap();
        assert_eq!(r1[0], 1.0, "the hit survived the radius; it used to be 0");
        assert!(r1[1].is_nan());
        // Cell 2's radius-1 disc reaches only the NaN at cell 1, never the
        // 10 two cells away, so it stays a miss.  Asserting a hit here
        // would be asserting that masking INVENTS exceedances, which is
        // the opposite error to the one this probe is about.
        assert_eq!(r1[2], 0.0);
        let r2 = exceedance_probability(&probe, 5.0, 2.0, NanPolicy::Mask).unwrap();
        assert_eq!(r2[2], 1.0, "at radius 2 the disc does reach the 10");
    }

    #[test]
    fn nmep_is_monotone_in_the_radius() {
        let probe = stack(
            3,
            3,
            &[
                &[0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0],
                &[9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, f64::NAN],
                &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        );
        let mut previous = exceedance_probability(&probe, 5.0, 0.0, NanPolicy::Mask).unwrap();
        for radius in [1.0, 1.5, 2.0, 3.0] {
            let next = exceedance_probability(&probe, 5.0, radius, NanPolicy::Mask).unwrap();
            for (index, (before, after)) in previous.iter().zip(next.iter()).enumerate() {
                if before.is_nan() {
                    assert!(after.is_nan(), "point {index} gained a voter with the radius");
                    continue;
                }
                assert!(
                    after + 1e-12 >= *before,
                    "point {index} fell from {before} to {after} at radius {radius}"
                );
            }
            previous = next;
        }
    }

    #[test]
    fn nmep_takes_the_member_maximum_first_not_the_probability_maximum() {
        // One member hits at cell 0; the other two never do.  Order-swapped,
        // cell 1 would read 1.0 (one member's hit at full confidence).
        let probe = stack(
            1,
            3,
            &[&[9.0, 0.0, 0.0], &[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]],
        );
        let out = exceedance_probability(&probe, 5.0, 1.0, NanPolicy::Mask).unwrap();
        assert!((out[1] - 1.0 / 3.0).abs() < 1e-12, "{out:?}");
    }

    /// The fail-closed policy's own arm of the exceedance count.
    ///
    /// Every other NMEP test here runs `Mask`, so the `Refuse` branch --
    /// the one a forecast asks for when it wants a missing member to
    /// stop the product rather than be worked around -- had no test at
    /// all.  Inverting its comparison turns every exceedance probability
    /// into its complement, and it survived the whole suite.
    ///
    /// The probe separates all four ways that comparison can be wrong.
    /// Point 0 has three of four members above the threshold, so the
    /// answer is 0.75: an inverted comparison reads 0.25 and an equality
    /// reads 0.  Point 1 has two members sitting EXACTLY on the
    /// threshold, so the answer is 0: `>=` reads 0.5.  The hit and vote
    /// accumulators are unsigned, so turning either `+=` into `-=`
    /// underflows rather than answering.
    #[test]
    fn exceedance_under_refuse_counts_what_is_above_the_threshold() {
        let probe = stack(
            1,
            3,
            &[
                &[9.0, 5.0, 0.0],
                &[9.0, 5.0, 0.0],
                &[9.0, 0.0, 0.0],
                &[0.0, 0.0, 0.0],
            ],
        );
        let out = exceedance_probability(&probe, 5.0, 0.0, NanPolicy::Refuse).unwrap();
        assert_eq!(out[0], 0.75, "three of four members exceed 5 here: {out:?}");
        assert_eq!(
            out[1], 0.0,
            "a member sitting exactly on the threshold has not exceeded it: {out:?}"
        );
        assert_eq!(out[2], 0.0, "{out:?}");

        // The same threshold rule under Mask, which has its own
        // comparison a few lines down and its own way of being wrong.
        let masked = exceedance_probability(&probe, 5.0, 0.0, NanPolicy::Mask).unwrap();
        assert_eq!(masked[0], 0.75, "{masked:?}");
        assert_eq!(masked[1], 0.0, "{masked:?}");
    }

    #[test]
    fn the_denominator_is_the_point_roster_not_the_disc_roster() {
        let probe = stack(1, 3, &[&[0.0, f64::NAN, 0.0], &[9.0, 9.0, 0.0]]);
        for radius in [0.0, 1.0, 2.0] {
            let out = exceedance_probability(&probe, 5.0, radius, NanPolicy::Mask).unwrap();
            // Point 1: only member 2 votes, and it exceeds at every radius.
            assert_eq!(out[1], 1.0, "radius {radius}");
        }
    }

    // ---- PMM -----------------------------------------------------------

    #[test]
    fn pmm_keeps_the_strongest_feature_where_the_mean_put_it() {
        // The exact probe the docstring names: it used to return
        // [NaN, 90, 1].
        let probe = stack(1, 3, &[&[100.0, 1.0, f64::NAN], &[90.0, 2.0, 0.0]]);
        let out = probability_matched_mean(&probe, NanPolicy::Mask, PmmTieRule::FlatIndex).unwrap();
        assert_eq!(out, vec![100.0, 90.0, 1.0]);
    }

    #[test]
    fn pmm_preserves_the_pooled_distribution_on_a_complete_stack() {
        // Complete stack, M members: the picks are exactly 0, M, 2M, ...
        let probe = stack(
            1,
            4,
            &[&[1.0, 4.0, 3.0, 2.0], &[8.0, 5.0, 6.0, 7.0]],
        );
        let out = probability_matched_mean(&probe, NanPolicy::Mask, PmmTieRule::FlatIndex).unwrap();
        // pool desc = 8 7 6 5 4 3 2 1 ; picks 0,2,4,6 -> 8 6 4 2
        // mean = 4.5 4.5 4.5 4.5 -> all tied, flat index order
        assert_eq!(out, vec![8.0, 6.0, 4.0, 2.0]);
        // ... and the ensemble maximum survives.
        assert_eq!(out.iter().cloned().fold(f64::MIN, f64::max), 8.0);
    }

    #[test]
    fn the_average_tie_rule_flattens_the_row_major_artifact() {
        let probe = stack(
            1,
            4,
            &[&[1.0, 4.0, 3.0, 2.0], &[8.0, 5.0, 6.0, 7.0]],
        );
        let out = probability_matched_mean(&probe, NanPolicy::Mask, PmmTieRule::Average).unwrap();
        assert_eq!(out, vec![5.0, 5.0, 5.0, 5.0]);
        let report = pmm_tie_report(&probe, NanPolicy::Mask).unwrap();
        assert_eq!(report.tied_points, 4);
        assert_eq!(report.largest_tie_group, 4);
        assert!((report.tied_fraction - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_point_with_no_finite_member_takes_no_pooled_value() {
        let probe = stack(1, 3, &[&[5.0, f64::NAN, 1.0], &[6.0, f64::NAN, 2.0]]);
        let out = probability_matched_mean(&probe, NanPolicy::Mask, PmmTieRule::FlatIndex).unwrap();
        assert!(out[1].is_nan());
        assert!(out[0].is_finite() && out[2].is_finite());
    }

    // ---- colours --------------------------------------------------------

    #[test]
    fn a_member_keeps_its_colour_when_the_roster_is_filtered() {
        let full: Vec<&str> = (1..=30).map(member_color).collect();
        let filtered: Vec<&str> = [3u32, 7, 11].iter().copied().map(member_color).collect();
        assert_eq!(filtered[1], full[6]);
        assert_eq!(member_color(0), PAINTBALL_PALETTE[0]);
        assert_eq!(member_color(40), PAINTBALL_PALETTE[0]);
    }

    #[test]
    fn probability_levels_start_above_zero() {
        let levels = probability_levels();
        assert_eq!(levels.len(), 10);
        assert!((levels[0] - 0.1).abs() < 1e-12);
        assert!((levels[9] - 1.0).abs() < 1e-12);
    }
}

#[cfg(test)] mod visibility_probability_tests {
 use super::*;
 #[test] fn strict_below_uses_complete_roster() {
  let s=MemberStack::new(1,3,vec![(1,vec![399.0,1000.0,f64::NAN]),(2,vec![400.0,999.0,0.0])]).unwrap();
  let p=below_threshold_probability(&s,400.0,0.0,NanPolicy::Mask).unwrap();
  assert_eq!(&p[..2],&[0.5,0.0]); assert!(p[2].is_nan());
  let p=below_threshold_probability(&s,1000.0,0.0,NanPolicy::Mask).unwrap();
  assert_eq!(&p[..2],&[1.0,0.5]); assert!(p[2].is_nan());
  assert!(below_threshold_probability(&s,1000.0,0.0,NanPolicy::Refuse).is_err());
 }
}
