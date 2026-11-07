//! LANE 2.  WPS point interpolators (`gpuwm/static/build.py`).
//!
//! `vals`: 2-D f64 window with NaN missing; `xi, yi`: fractional
//! 1-based source coordinates; `x0, y0`: window origin.  Each returns
//! NaN where the option does not apply (fall-through), and
//! [`interp_seq`] applies options in order, each filling remaining
//! NaNs -- exactly `_interp_seq`.  Because every option is pure per
//! point, per-point fall-through equals the Python's option-major
//! masking, and points parallelize with rayon without touching any
//! accumulation order.
//!
//! `search_nearest` is WPS `search`: breadth-first from
//! `nint(x), nint(y)` with the 1200-depth cap, then the
//! Euclidean-closest of the finite frontier (NOT an unrestricted
//! nearest-neighbour search; the distinction matters in sizeable
//! holes).  The Python's deque order is the defined behaviour; kept.

use std::collections::HashSet;
use std::collections::VecDeque;

use rayon::prelude::*;

use crate::error::{Result, StaticError};

/// WPS interpolation arithmetic for default-REAL orographic fields.
/// The input values are integer words converted to f32. WPS applies the
/// index scale only to the interpolated result.
pub(crate) fn orographic_real(
    tile: &crate::geog::GeogWindow, vals: &[f32], x: f32, y: f32, op: InterpOp,
) -> f32 {
    if let InterpOp::SearchDepth(depth) = op {
        return search_real(tile, vals, x, y, depth);
    }
    let (sx,sy,ex,ey)=(tile.x0,tile.y0,tile.x1(),tile.y1());
    // WPS v4.6.0 interp_module.F:nearest_neighbor, default REAL/NINT.
    if op == InterpOp::NearestNeighbor {
        let (i, j) = (x.round() as i64, y.round() as i64);
        return if i < sx || i > ex || j < sy || j > ey { f32::NAN }
            else { vals[(j-sy) as usize*tile.nx+(i-sx) as usize] };
    }
    let (mut ix,mut jx,mut iy,mut jy)=(x.floor() as i64,x.ceil() as i64,
                                    y.floor() as i64,y.ceil() as i64);
    if op == InterpOp::Average4Pt {
        if x > sx as f32-0.5 && ix < sx { ix=sx; jx=sx; }
        else if x < ex as f32+0.5 && jx > ex { ix=ex; jx=ex; }
        if y > sy as f32-0.5 && iy < sy { iy=sy; jy=sy; }
        else if y < ey as f32+0.5 && jy > ey { iy=ey; jy=ey; }
    }
    if ix < sx || jx > ex || iy < sy || jy > ey { return f32::NAN; }
    let at=|i:i64,j:i64| vals[(j-sy) as usize*tile.nx+(i-sx) as usize];
    let (a,b,c,d)=(at(ix,iy),at(ix,jy),at(jx,iy),at(jx,jy));
    match op {
        InterpOp::Average4Pt => {
            let mut sum=0.0f32;
            let mut count=0.0f32;
            for v in [a,b,c,d] { if !v.is_nan() { sum+=v; count+=1.0; } }
            if count > 0.0 {sum/count} else {f32::NAN}
        }
        InterpOp::FourPt => {
            if [a,b,c,d].iter().any(|v| v.is_nan()) { return f32::NAN; }
            if ix == jx {
                if iy == jy {a} else {a*(jy as f32-y)+b*(y-iy as f32)}
            } else if iy == jy {a*(jx as f32-x)+c*(x-ix as f32)}
            else {(y-iy as f32)*(b*(jx as f32-x)+d*(x-ix as f32))+
                  (jy as f32-y)*(a*(jx as f32-x)+c*(x-ix as f32))}
        }
        _ => f32::NAN,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpOp {
    NearestNeighbor,
    FourPt,
    Average4Pt,
    Average16Pt,
    SixteenPt,
    Search,
    SearchDepth(u32),
}

/// WPS v4.6.0 interp_module.F:search_extrap, default REAL, no source mask.
/// The queue depth is updated between sibling inserts in the Fortran body;
/// retaining that order matters for bounded search(n). The upstream notice
/// is retained in licenses/LICENSE-WRF-public-domain.txt.
pub(crate) fn search_real(
    tile: &crate::geog::GeogWindow, vals: &[f32], x: f32, y: f32, depth_limit: u32,
) -> f32 {
    let (sx, sy, ex, ey) = (tile.x0, tile.y0, tile.x1(), tile.y1());
    let (ic, jc) = (x.round() as i64, y.round() as i64);
    if ic < sx || ic > ex || jc < sy || jc > ey { return f32::NAN; }
    let offset = |i: i64, j: i64| (j-sy) as usize*tile.nx+(i-sx) as usize;
    let valid = |i: i64, j: i64| !vals[offset(i,j)].is_nan();
    let mut queue = VecDeque::from([(ic, jc, 0u32)]);
    let mut seen = HashSet::from([(ic, jc)]);
    let mut found = None;
    while found.is_none() {
        let Some((i, j, mut depth)) = queue.pop_front() else { break; };
        if valid(i,j) { found = Some((i,j)); }
        for (ii,jj) in [(i-1,j),(i+1,j),(i,j-1),(i,j+1)] {
            if ii >= sx && ii <= ex && jj >= sy && jj <= ey
                && !seen.contains(&(ii,jj)) && depth < depth_limit {
                depth += 1;
                queue.push_back((ii,jj,depth));
                seen.insert((ii,jj));
            }
        }
    }
    let Some((mut bi, mut bj)) = found else { return f32::NAN; };
    let distance = |i: i64, j: i64| {
        let dx = i as f32-x;
        let dy = j as f32-y;
        dx*dx+dy*dy
    };
    let mut best = distance(bi,bj);
    for (i,j,_) in queue {
        if valid(i,j) {
            let candidate = distance(i,j);
            if candidate < best { best=candidate; bi=i; bj=j; }
        }
    }
    vals[offset(bi,bj)]
}

/// Window view for the interpolators (borrowed, row-major).
pub struct WindowView<'a> {
    pub ny: usize,
    pub nx: usize,
    pub vals: &'a [f64],
    /// Window origin in 1-based source coordinates.
    pub x0: f64,
    pub y0: f64,
}

impl WindowView<'_> {
    #[inline]
    fn at(&self, j: i64, i: i64) -> f64 {
        self.vals[j as usize * self.nx + i as usize]
    }
}

#[inline]
fn clip(v: i64, lo: i64, hi: i64) -> i64 {
    v.max(lo).min(hi)
}

/// `np.nansum` over a small fixed stack: NaNs contribute +0.0 to the
/// sequential sum, in stack order, starting from the first element (the
/// exact numpy small-stack accumulation).
#[inline]
fn nansum_count(stack: &[f64]) -> (f64, i64) {
    let mut cnt = 0i64;
    let mut acc: Option<f64> = None;
    for &v in stack {
        let term = if v.is_nan() {
            0.0
        } else {
            cnt += 1;
            v
        };
        acc = Some(match acc {
            None => term,
            Some(a) => a + term,
        });
    }
    (acc.unwrap_or(0.0), cnt)
}

/// Bilinear on the 4 surrounding pixels; NaN if any is missing
/// (`four_pt`).
pub fn four_pt(win: &WindowView<'_>, xi: f64, yi: f64) -> f64 {
    let fx = xi - win.x0;
    let fy = yi - win.y0;
    let (ny, nx) = (win.ny as i64, win.nx as i64);
    let i0 = fx.floor() as i64;
    let i1 = fx.ceil() as i64;
    let j0 = fy.floor() as i64;
    let j1 = fy.ceil() as i64;
    let ok = i0 >= 0 && i1 < nx && j0 >= 0 && j1 < ny;
    let i0 = clip(i0, 0, nx - 1);
    let i1 = clip(i1, 0, nx - 1);
    let j0 = clip(j0, 0, ny - 1);
    let j1 = clip(j1, 0, ny - 1);
    let wx = fx - i0 as f64;
    let wy = fy - j0 as f64;
    let v00 = win.at(j0, i0);
    let v01 = win.at(j0, i1);
    let v10 = win.at(j1, i0);
    let v11 = win.at(j1, i1);
    let res = (1.0 - wy) * ((1.0 - wx) * v00 + wx * v01)
        + wy * ((1.0 - wx) * v10 + wx * v11);
    if ok { res } else { f64::NAN }
}

/// Mean of the valid pixels among the surrounding 2x2, permitted up to
/// half a source cell beyond the window with edge clamping
/// (`average_4pt`).
pub fn average_4pt(win: &WindowView<'_>, xi: f64, yi: f64) -> f64 {
    let (ny, nx) = (win.ny as i64, win.nx as i64);
    let (x0, y0) = (win.x0, win.y0);
    let x1 = x0 + (nx - 1) as f64;
    let y1 = y0 + (ny - 1) as f64;
    let (x0i, y0i) = (x0 as i64, y0 as i64);
    let (x1i, y1i) = (x1 as i64, y1 as i64);
    let mut i0 = xi.floor() as i64;
    let mut i1 = xi.ceil() as i64;
    let mut j0 = yi.floor() as i64;
    let mut j1 = yi.ceil() as i64;

    // WPS permits average_4pt up to half a source cell beyond a loaded
    // tile, clamping the pair of indices to the edge point
    // (interp_module.F).
    let lo = xi > x0 - 0.5 && i0 < x0i;
    let hi = xi < x1 + 0.5 && i1 > x1i;
    if lo {
        i0 = x0i;
        i1 = x0i;
    } else if hi {
        i0 = x1i;
        i1 = x1i;
    }
    let lo = yi > y0 - 0.5 && j0 < y0i;
    let hi = yi < y1 + 0.5 && j1 > y1i;
    if lo {
        j0 = y0i;
        j1 = y0i;
    } else if hi {
        j0 = y1i;
        j1 = y1i;
    }
    let ok = i0 >= x0i && i1 <= x1i && j0 >= y0i && j1 <= y1i;
    let i0 = clip(i0 - x0i, 0, nx - 1);
    let i1 = clip(i1 - x0i, 0, nx - 1);
    let j0 = clip(j0 - y0i, 0, ny - 1);
    let j1 = clip(j1 - y0i, 0, ny - 1);
    let stack = [
        win.at(j0, i0),
        win.at(j0, i1),
        win.at(j1, i0),
        win.at(j1, i1),
    ];
    let (sum, cnt) = nansum_count(&stack);
    if ok && cnt > 0 {
        sum / cnt.max(1) as f64
    } else {
        f64::NAN
    }
}

/// WPS `average_16pt`: mean of valid values in the surrounding 4x4.
pub fn average_16pt(win: &WindowView<'_>, xi: f64, yi: f64) -> f64 {
    let fx = xi - win.x0;
    let fy = yi - win.y0;
    let (ny, nx) = (win.ny as i64, win.nx as i64);
    let i0 = fx.floor() as i64;
    let j0 = fy.floor() as i64;
    let ok = i0 >= 1 && i0 <= nx - 3 && j0 >= 1 && j0 <= ny - 3;
    let i0c = clip(i0, 1, (nx - 3).max(1));
    let j0c = clip(j0, 1, (ny - 3).max(1));
    if i0c + 2 >= nx || j0c + 2 >= ny {
        // A window narrower than the 4x4 stencil cannot answer (the
        // Python raises IndexError before its `ok` mask applies; the
        // defined behaviour is fall-through).
        return f64::NAN;
    }
    let mut stack = [0.0f64; 16];
    let mut k = 0;
    for dj in [-1i64, 0, 1, 2] {
        for di in [-1i64, 0, 1, 2] {
            stack[k] = win.at(j0c + dj, i0c + di);
            k += 1;
        }
    }
    let (sum, cnt) = nansum_count(&stack);
    if ok && cnt > 0 {
        sum / cnt.max(1) as f64
    } else {
        f64::NAN
    }
}

/// MM5/WPS overlapping parabolic in 1-D (`_oned`), exact expression
/// order.
#[inline]
fn oned(x: f64, a: f64, b: f64, c: f64, d: f64) -> f64 {
    (1.0 - x) * (b + x * (0.5 * (c - a) + x * (0.5 * (c + a) - b)))
        + x * (c
            + (1.0 - x)
                * (0.5 * (b - d) + (1.0 - x) * (0.5 * (b + d) - c)))
}

/// Overlapping parabolic on the surrounding 4x4; NaN if any missing
/// (`sixteen_pt` -- NaN operands fall through `_oned` naturally).
pub fn sixteen_pt(win: &WindowView<'_>, xi: f64, yi: f64) -> f64 {
    let fx = xi - win.x0;
    let fy = yi - win.y0;
    let (ny, nx) = (win.ny as i64, win.nx as i64);
    let i0 = fx.floor() as i64;
    let j0 = fy.floor() as i64;
    let ok = i0 >= 1 && i0 <= nx - 3 && j0 >= 1 && j0 <= ny - 3;
    let i0c = clip(i0, 1, (nx - 3).max(1));
    let j0c = clip(j0, 1, (ny - 3).max(1));
    if i0c + 2 >= nx || j0c + 2 >= ny {
        return f64::NAN;
    }
    let x = fx - i0c as f64;
    let y = fy - j0c as f64;
    let mut rows = [0.0f64; 4];
    for (l, row) in rows.iter_mut().enumerate() {
        let j = j0c + l as i64 - 1;
        *row = oned(
            x,
            win.at(j, i0c - 1),
            win.at(j, i0c),
            win.at(j, i0c + 1),
            win.at(j, i0c + 2),
        );
    }
    let res = oned(y, rows[0], rows[1], rows[2], rows[3]);
    if ok { res } else { f64::NAN }
}

/// WPS `search`: breadth-first extrapolation from `nint(x), nint(y)`
/// with the 1200-depth cap; the winner is the Euclidean-closest valid
/// point of the finite frontier (`search_nearest`).
pub fn search_nearest(win: &WindowView<'_>, xi: f64, yi: f64) -> f64 {
    let fx = xi - win.x0;
    let fy = yi - win.y0;
    let (ny, nx) = (win.ny as i64, win.nx as i64);
    let valid = |j: i64, i: i64| !win.at(j, i).is_nan();
    if !win.vals.iter().any(|v| !v.is_nan()) {
        return f64::NAN;
    }
    let ic = (fx + 0.5).floor() as i64;
    let jc = (fy + 0.5).floor() as i64;
    if ic < 0 || ic >= nx || jc < 0 || jc >= ny {
        return f64::NAN;
    }
    let mut q: VecDeque<(i64, i64, u32)> = VecDeque::new();
    q.push_back((ic, jc, 0));
    let mut seen: HashSet<i64> = HashSet::new();
    seen.insert(jc * nx + ic);
    let mut found: Option<(i64, i64)> = None;
    while let Some((i, j, depth)) = {
        if found.is_some() { None } else { q.pop_front() }
    } {
        if valid(j, i) {
            found = Some((i, j));
        }
        if depth < 1200 {
            // WPS default maximum search depth
            for (ii, jj) in
                [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
            {
                let key = jj * nx + ii;
                if ii >= 0
                    && ii < nx
                    && jj >= 0
                    && jj < ny
                    && seen.insert(key)
                {
                    q.push_back((ii, jj, depth + 1));
                }
            }
        }
    }
    let Some((mut bi, mut bj)) = found else {
        return f64::NAN;
    };
    let mut best_d2 =
        (bi as f64 - fx).powi(2) + (bj as f64 - fy).powi(2);
    for &(i, j, _) in &q {
        if valid(j, i) {
            let d2 =
                (i as f64 - fx).powi(2) + (j as f64 - fy).powi(2);
            if d2 < best_d2 {
                bi = i;
                bj = j;
                best_d2 = d2;
            }
        }
    }
    win.at(bj, bi)
}

fn apply(op: InterpOp, win: &WindowView<'_>, xi: f64, yi: f64) -> f64 {
    match op {
        InterpOp::NearestNeighbor => {
            let (i,j)=((xi.round()-win.x0) as i64,(yi.round()-win.y0) as i64);
            if i < 0 || j < 0 || i >= win.nx as i64 || j >= win.ny as i64 {
                f64::NAN
            } else { win.at(j,i) }
        },
        InterpOp::FourPt => four_pt(win, xi, yi),
        InterpOp::Average4Pt => average_4pt(win, xi, yi),
        InterpOp::Average16Pt => average_16pt(win, xi, yi),
        InterpOp::SixteenPt => sixteen_pt(win, xi, yi),
        InterpOp::Search => search_nearest(win, xi, yi),
        InterpOp::SearchDepth(_) => f64::NAN,
    }
}

/// One interpolation option over many points.  LANE 2 (rayon over
/// points; each point is independent so the result is order-free).
pub fn interp_one(
    op: InterpOp,
    win: &WindowView<'_>,
    xi: &[f64],
    yi: &[f64],
    out: &mut [f64],
) -> Result<()> {
    if xi.len() != yi.len() || xi.len() != out.len() {
        return Err(StaticError::Invalid(format!(
            "interp_one length mismatch: xi={}, yi={}, out={}",
            xi.len(),
            yi.len(),
            out.len()
        )));
    }
    out.par_iter_mut()
        .zip(xi.par_iter().zip(yi.par_iter()))
        .for_each(|(slot, (&x, &y))| *slot = apply(op, win, x, y));
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn nearest_preserves_half_ties_missing_and_bounds() {
        use crate::geog::{GeogIndex,GeogWindow};
        let index=GeogIndex::parse(&std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("golden/orographic/index")).unwrap();
        let tile=GeogWindow{index,x0:1,y0:1,first_plane:0,nz:1,ny:2,nx:3,
            raw:vec![0;6],coverage:None};
        let v=[11.0,12.0,f32::NAN,21.0,22.0,23.0];
        for (x,y,want) in [(0.5,1.0,11.0),(1.5,1.0,12.0),(3.49,2.0,23.0),
            (1.0,1.5,21.0),(3.0,1.0,f32::NAN),(0.49,1.0,f32::NAN),
            (3.5,2.0,f32::NAN),(1.0,2.5,f32::NAN)] {
            let got=super::orographic_real(&tile,&v,x,y,super::InterpOp::NearestNeighbor);
            if want.is_nan(){assert!(got.is_nan());}else{assert_eq!(got,want);}
        }
    }

    #[test]
    fn bounded_search_matches_unmodified_wps_fortran() {
        use crate::geog::{GeogIndex,GeogWindow};
        let index=GeogIndex::parse(&std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("golden/orographic/index")).unwrap();
        let mut tile=GeogWindow{index,x0:1,y0:1,first_plane:0,nz:1,ny:11,nx:11,
            raw:vec![0;121],coverage:None};
        tile.index.tile_x=11;
        tile.index.tile_y=11;
        let expected=include_bytes!("../golden/orographic/search-real.bin");
        let x=[6.0,6.4,5.5,1.0,11.0,0.49,6.0,6.0];
        let y=[6.0,6.4,6.5,1.0,11.0,6.0,0.49,11.51];
        let mut n=0;
        for pattern in 0..8 {
            let mut values=vec![f32::NAN;121];
            let mut put=|i:usize,j:usize,v:f32| {values[(j-1)*11+i-1]=v;};
            match pattern {
                0 => for j in 1..=11 {for i in 1..=11 {put(i,j,(100*j+i) as f32);}},
                1 => put(7,6,17.0),
                2 => put(6,7,27.0),
                3 => put(6,8,37.0),
                4 => put(4,6,47.0),
                5 => {put(5,6,15.0);put(7,6,17.0);put(6,5,25.0);put(6,7,27.0);},
                7 => {put(5,5,11.0);put(7,7,33.0);put(9,6,22.0);},
                _ => (),
            }
            for depth in [0,1,2,3,4,5,8] {
                for k in 0..x.len() {
                    let want=f32::from_le_bytes(expected[n*4..(n+1)*4].try_into().unwrap());
                    let got=super::search_real(&tile,&values,x[k],y[k],depth);
                    if want == -9999.0 {assert!(got.is_nan(),"pattern{pattern} depth{depth} point{k}");}
                    else {assert_eq!(got.to_bits(),want.to_bits(),"pattern{pattern} depth{depth} point{k}");}
                    n+=1;
                }
            }
        }
        assert_eq!(n*4,expected.len());
    }
    #[test]
    fn orographic_interpolation_matches_wps_real_oracle() {
        use crate::geog::{GeogIndex,GeogWindow};
        use super::{orographic_real,InterpOp};
        let index=GeogIndex::parse(&std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("golden/orographic/index")).unwrap();
        let mut tile=GeogWindow{index,x0:1,y0:1,first_plane:0,nz:1,ny:2,nx:2,
            raw:vec![31,80,245,301],coverage:None};
        let data=include_bytes!("../golden/orographic/interp-real.bin");
        let x=[1.25,1.0,1.25,1.0,0.75,1.99999988,1.25,1.75];
        let y=[1.75,1.75,1.0,1.0,1.25,1.00000012,1.25,1.25];
        let values=tile.values_real(0);
        for k in 0..x.len() {
            for (m,op) in [InterpOp::Average4Pt,InterpOp::FourPt].iter().enumerate() {
                let value=orographic_real(&tile,&values,x[k],y[k],*op);
                let expected=f32::from_le_bytes(data[(k*2+m)*4..(k*2+m+1)*4].try_into().unwrap());
                if expected == -99999.0 {assert!(value.is_nan());}
                else {assert_eq!((value*tile.index.scale_factor as f32).to_bits(),expected.to_bits(),"point{k} {op:?}");}
            }
        }
        for pattern in 0..2 {
            if pattern==0 {tile.raw[0]=65535;} else {tile.raw.fill(65535);}
            let values=tile.values_real(0);
            for k in 0..4 {
                for (m,op) in [InterpOp::Average4Pt,InterpOp::FourPt].iter().enumerate() {
                    let value=orographic_real(&tile,&values,x[k],y[k],*op);
                    let word=16+pattern*8+k*2+m;
                    let expected=f32::from_le_bytes(data[word*4..(word+1)*4].try_into().unwrap());
                    if expected==65535.0 {assert!(value.is_nan());}
                    else {assert_eq!((value*tile.index.scale_factor as f32).to_bits(),expected.to_bits(),"missing pattern{pattern} point{k} {op:?}");}
                }
            }
        }
    }
    use super::*;
    use crate::testsupport::{
        assert_bits_f64, golden_dir, hex_f64_vec, json, read_f64,
    };

    fn view<'a>(
        dims: &[usize],
        vals: &'a [f64],
        x0: f64,
        y0: f64,
    ) -> WindowView<'a> {
        WindowView {
            ny: dims[0],
            nx: dims[1],
            vals,
            x0,
            y0,
        }
    }

    #[test]
    fn interpolators_match_the_python_reference_bitwise() {
        let dir = golden_dir().join("interp");
        let spec = json(&dir.join("goldens.json"));
        let (dims, vals) =
            read_f64(&dir.join(spec["vals"].as_str().unwrap()));
        let win = view(
            &dims,
            &vals,
            spec["x0"].as_f64().unwrap(),
            spec["y0"].as_f64().unwrap(),
        );
        let xi = hex_f64_vec(&spec["xi"]);
        let yi = hex_f64_vec(&spec["yi"]);
        for (name, op) in [
            ("four_pt", InterpOp::FourPt),
            ("average_4pt", InterpOp::Average4Pt),
            ("average_16pt", InterpOp::Average16Pt),
            ("sixteen_pt", InterpOp::SixteenPt),
            ("search", InterpOp::Search),
        ] {
            let want = hex_f64_vec(&spec["ops"][name]);
            let mut got = vec![0.0; xi.len()];
            interp_one(op, &win, &xi, &yi, &mut got).unwrap();
            assert_bits_f64(&got, &want, name);
        }
        let seq = [
            InterpOp::FourPt,
            InterpOp::Average4Pt,
            InterpOp::Average16Pt,
            InterpOp::Search,
        ];
        let want = hex_f64_vec(&spec["seq"]["out"]);
        let got = interp_seq(&seq, &win, &xi, &yi).unwrap();
        assert_bits_f64(&got, &want, "interp_seq");
    }

    #[test]
    fn search_frontier_rule_matches_the_python_hole_case() {
        let dir = golden_dir().join("interp");
        let spec = json(&dir.join("goldens.json"));
        let hole = &spec["search_hole"];
        let (dims, vals) =
            read_f64(&dir.join(hole["vals"].as_str().unwrap()));
        let win = view(
            &dims,
            &vals,
            hole["x0"].as_f64().unwrap(),
            hole["y0"].as_f64().unwrap(),
        );
        let xi = hex_f64_vec(&hole["xi"]);
        let yi = hex_f64_vec(&hole["yi"]);
        let want = hex_f64_vec(&hole["out"]);
        let mut got = vec![0.0; xi.len()];
        interp_one(InterpOp::Search, &win, &xi, &yi, &mut got).unwrap();
        assert_bits_f64(&got, &want, "search_hole");
    }
}

/// Apply a sequence, each op filling remaining NaNs (`_interp_seq`).
/// Per point the first op producing a non-NaN answer wins, which is
/// exactly the option-major masking the Python performs.
pub fn interp_seq(
    seq: &[InterpOp],
    win: &WindowView<'_>,
    xi: &[f64],
    yi: &[f64],
) -> Result<Vec<f64>> {
    if xi.len() != yi.len() {
        return Err(StaticError::Invalid(format!(
            "interp_seq length mismatch: xi={}, yi={}",
            xi.len(),
            yi.len()
        )));
    }
    Ok(xi
        .par_iter()
        .zip(yi.par_iter())
        .map(|(&x, &y)| {
            for &op in seq {
                let got = apply(op, win, x, y);
                if !got.is_nan() {
                    return got;
                }
            }
            f64::NAN
        })
        .collect())
}
