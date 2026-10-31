//! All scalar/vector parameters the full SPC window displays, computed once
//! from a [`crate::Profile`]. Values that `sharprs` provides directly are
//! wired here; the rest are ported from `sharpmod.sharptab.derived` /
//! vendored `sharppy.sharptab.params`. NaN = unavailable (renders as `--`).
//!
//! Golden reference for every field: `testdata/golden_full.json` (generated
//! by running the actual Python stack on the bundled example sounding).

use sharprs::params::cape::{
    self, LiftedParcelLevel, ParcelResult, ParcelType as SharprsParcelType,
};
use sharprs::params::{composites, indices};
use sharprs::profile::comp2vec;
use sharprs::thermo;
use sharprs::utils::kts2ms;
use sharprs::winds;

use crate::Profile;
use crate::utils::qc;

/// `(u, v)` kts.
pub type Comp = (f64, f64);
/// `(wdir deg, wspd kt)`.
pub type Vect = (f64, f64);

/// Every value the bottom index board / insets / hodograph annotations need.
#[derive(Clone, Debug)]
pub struct DerivedParams {
    // --- thermo block ---
    pub pwat: f64,
    pub k_idx: f64,
    pub tei: f64,
    pub esp: f64,
    pub mmp: f64,
    pub wndg: f64,
    pub dcp: f64,
    pub mburst: f64,
    pub ship: f64,
    pub right_scp: f64,
    pub left_scp: f64,
    pub stp_cin: f64,
    pub stp_fixed: f64,
    pub sweat: f64,
    pub sig_severe: f64,
    pub dcape: f64,
    pub drush_f: f64,
    pub mean_mixr: f64,
    pub low_rh: f64,
    pub mid_rh: f64,
    pub totals_totals: f64,
    pub conv_t_f: f64,
    pub max_t_f: f64,
    pub thetae_diff: f64,
    // --- lapse rates (C/km) ---
    pub lapserate_3km: f64,
    pub lapserate_3_6km: f64,
    pub lapserate_850_500: f64,
    pub lapserate_700_500: f64,
    pub lapserate_sfc_500m: f64,
    pub lapserate_sfc_1km: f64,
    // --- kinematics ---
    pub srh500: f64,
    pub srh1km: f64,
    pub srh3km: f64,
    pub right_esrh: f64,
    pub sfc_500m_shear: Comp,
    pub sfc_1km_shear: Comp,
    pub sfc_3km_shear: Comp,
    pub sfc_6km_shear: Comp,
    pub sfc_8km_shear: Comp,
    pub eff_shear: Comp,
    pub ebwd: Comp,
    pub lcl_el_shear: Comp,
    pub mean_wind_sfc_500m: Comp,
    pub mean_1km: Vect,
    pub mean_3km: Vect,
    pub mean_6km: Vect,
    pub mean_8km: Vect,
    pub mean_eff: Comp,
    pub mean_ebw: Comp,
    pub mean_lcl_el: Vect,
    pub srw_sfc_500m: Comp,
    pub srw_1km: Vect,
    pub srw_3km: Vect,
    pub srw_6km: Vect,
    pub srw_8km: Vect,
    pub srw_4_5km: Vect,
    pub srw_eff: Comp,
    pub srw_ebw: Comp,
    pub srw_lcl_el: Vect,
    pub wind1km: Vect,
    pub wind6km: Vect,
    /// Corfidi vectors: (upshear (u,v), downshear (u,v)).
    pub corfidi_up: Comp,
    pub corfidi_dn: Comp,
    pub right_critical_angle: f64,
    pub brnshear: f64,
    // --- SHARPpy-Reimagined derived composites ---
    pub ehi_0_1km: f64,
    pub ehi_0_3km: f64,
    pub vgp: f64,
    pub peskov: f64,
    pub mcs_index: f64,
    pub ncape: f64,
    pub lrghail: f64,
    pub lscp: f64,
    pub nstp: f64,
    pub hgz_cape: f64,
    pub wbz_height: f64,
    pub ecape: f64,
    /// MOSHE as displayed: NaN (`--`) when the profile has no effective
    /// inflow layer, because its EBWD term is then undefined.
    pub modified_sherbe: f64,
    /// sharpmod's MOSHE, kept only for parity proofs: with no effective
    /// layer it feeds the literal MISSING (-9999 kt) ebwspd into the formula
    /// and returns a large meaningless number. Never displayed.
    pub modified_sherbe_sharpmod: f64,
    pub cape_0_3km: f64,
    pub cape_0_6km: f64,
    // --- strips / insets data ---
    /// Inferred temperature advection: (C/hr per layer, (pbot, ptop) bounds).
    pub temp_adv: Vec<f64>,
    pub temp_adv_bounds: Vec<(f64, f64)>,
    /// Storm slinky trajectory ((x, y) meters, updraft tilt deg) for the
    /// displayed parcel, right-mover storm motion.
    pub slinky_traj: Vec<(f64, f64)>,
    pub slinky_tilt: f64,
}

impl DerivedParams {
    pub fn nan() -> DerivedParams {
        DerivedParams {
            pwat: f64::NAN,
            k_idx: f64::NAN,
            tei: f64::NAN,
            esp: f64::NAN,
            mmp: f64::NAN,
            wndg: f64::NAN,
            dcp: f64::NAN,
            mburst: f64::NAN,
            ship: f64::NAN,
            right_scp: f64::NAN,
            left_scp: f64::NAN,
            stp_cin: f64::NAN,
            stp_fixed: f64::NAN,
            sweat: f64::NAN,
            sig_severe: f64::NAN,
            dcape: f64::NAN,
            drush_f: f64::NAN,
            mean_mixr: f64::NAN,
            low_rh: f64::NAN,
            mid_rh: f64::NAN,
            totals_totals: f64::NAN,
            conv_t_f: f64::NAN,
            max_t_f: f64::NAN,
            thetae_diff: f64::NAN,
            lapserate_3km: f64::NAN,
            lapserate_3_6km: f64::NAN,
            lapserate_850_500: f64::NAN,
            lapserate_700_500: f64::NAN,
            lapserate_sfc_500m: f64::NAN,
            lapserate_sfc_1km: f64::NAN,
            srh500: f64::NAN,
            srh1km: f64::NAN,
            srh3km: f64::NAN,
            right_esrh: f64::NAN,
            sfc_500m_shear: (f64::NAN, f64::NAN),
            sfc_1km_shear: (f64::NAN, f64::NAN),
            sfc_3km_shear: (f64::NAN, f64::NAN),
            sfc_6km_shear: (f64::NAN, f64::NAN),
            sfc_8km_shear: (f64::NAN, f64::NAN),
            eff_shear: (f64::NAN, f64::NAN),
            ebwd: (f64::NAN, f64::NAN),
            lcl_el_shear: (f64::NAN, f64::NAN),
            mean_wind_sfc_500m: (f64::NAN, f64::NAN),
            mean_1km: (f64::NAN, f64::NAN),
            mean_3km: (f64::NAN, f64::NAN),
            mean_6km: (f64::NAN, f64::NAN),
            mean_8km: (f64::NAN, f64::NAN),
            mean_eff: (f64::NAN, f64::NAN),
            mean_ebw: (f64::NAN, f64::NAN),
            mean_lcl_el: (f64::NAN, f64::NAN),
            srw_sfc_500m: (f64::NAN, f64::NAN),
            srw_1km: (f64::NAN, f64::NAN),
            srw_3km: (f64::NAN, f64::NAN),
            srw_6km: (f64::NAN, f64::NAN),
            srw_8km: (f64::NAN, f64::NAN),
            srw_4_5km: (f64::NAN, f64::NAN),
            srw_eff: (f64::NAN, f64::NAN),
            srw_ebw: (f64::NAN, f64::NAN),
            srw_lcl_el: (f64::NAN, f64::NAN),
            wind1km: (f64::NAN, f64::NAN),
            wind6km: (f64::NAN, f64::NAN),
            corfidi_up: (f64::NAN, f64::NAN),
            corfidi_dn: (f64::NAN, f64::NAN),
            right_critical_angle: f64::NAN,
            brnshear: f64::NAN,
            ehi_0_1km: f64::NAN,
            ehi_0_3km: f64::NAN,
            vgp: f64::NAN,
            peskov: f64::NAN,
            mcs_index: f64::NAN,
            ncape: f64::NAN,
            lrghail: f64::NAN,
            lscp: f64::NAN,
            nstp: f64::NAN,
            hgz_cape: f64::NAN,
            wbz_height: f64::NAN,
            ecape: f64::NAN,
            modified_sherbe: f64::NAN,
            modified_sherbe_sharpmod: f64::NAN,
            cape_0_3km: f64::NAN,
            cape_0_6km: f64::NAN,
            temp_adv: Vec::new(),
            temp_adv_bounds: Vec::new(),
            slinky_traj: Vec::new(),
            slinky_tilt: f64::NAN,
        }
    }

    /// Compute everything from an analyzed profile.
    ///
    /// Values `sharprs` provides are wired directly; the rest are ports of
    /// vendored `sharppy.sharptab.params`/`winds` and
    /// `sharpmod.sharptab.{derived,ecape,params,winds}`: see
    /// `tests/golden_full.rs` for the field-by-field reference values.
    pub fn compute(prof: &Profile) -> DerivedParams {
        let mut d = DerivedParams::nan();
        let inner = &prof.inner;
        if inner.num_levels() < 3 {
            return d;
        }
        let cape_prof = cape::Profile::new(
            inner.pres.clone(),
            inner.hght.clone(),
            inner.tmpc.clone(),
            inner.dwpc.clone(),
            inner.sfc,
        );
        let mupcl = &prof.mupcl;
        let sfcpcl = &prof.sfcpcl;
        let mlpcl = &prof.mlpcl;
        let (rstu, rstv, lstu, lstv) = prof.srwind;
        let sfc_pres = inner.sfc_pressure();

        // Pressures of the fixed AGL levels used throughout (log-p interp by
        // height, the same convention as SHARPpy `interp.pres(to_msl(h))`).
        let p_at = |h_agl: f64| inner.pres_at_height(inner.to_msl(h_agl));
        let p500m = p_at(500.0);
        let p1km = p_at(1000.0);
        let p1_5km = p_at(1500.0);
        let p3km = p_at(3000.0);
        let p3_5km = p_at(3500.0);
        let p4km = p_at(4000.0);
        let p5km = p_at(5000.0);
        let p6km = p_at(6000.0);
        let p8km = p_at(8000.0);
        let p12km = p_at(12000.0);

        // --- kinematics: shears -------------------------------------------
        d.sfc_1km_shear = shear(inner, sfc_pres, p1km);
        d.sfc_3km_shear = shear(inner, sfc_pres, p3km);
        d.sfc_6km_shear = shear(inner, sfc_pres, p6km);
        d.sfc_8km_shear = shear(inner, sfc_pres, p8km);
        d.lcl_el_shear = shear(inner, mupcl.lclpres, mupcl.elpres);
        // sharpmod `sfc_500m_kinematics` interpolates u/v linearly in height
        // (not log-p) for the 0-500 m bulk shear.
        {
            let sfc_h = inner.sfc_height();
            let u0 = interp_h(&inner.hght, &inner.u, sfc_h);
            let v0 = interp_h(&inner.hght, &inner.v, sfc_h);
            let u5 = interp_h(&inner.hght, &inner.u, sfc_h + 500.0);
            let v5 = interp_h(&inner.hght, &inner.v, sfc_h + 500.0);
            d.sfc_500m_shear = (u5 - u0, v5 - v0);
        }

        // --- effective layer kinematics -----------------------------------
        let mut ebwspd = f64::NAN;
        if qc(prof.ebottom) && qc(prof.etop) {
            d.eff_shear = shear(inner, prof.ebottom, prof.etop);
            d.mean_eff = mean_wind(inner, prof.ebottom, prof.etop, 0.0, 0.0);
            d.srw_eff = mean_wind(inner, prof.ebottom, prof.etop, rstu, rstv);
            if qc(prof.ebotm) && qc(mupcl.elhght) {
                let depth = (mupcl.elhght - prof.ebotm) / 2.0;
                let elh = p_at(prof.ebotm + depth);
                d.ebwd = shear(inner, prof.ebottom, elh);
                ebwspd = mag(d.ebwd.0, d.ebwd.1);
                d.mean_ebw = mean_wind(inner, prof.ebottom, elh, 0.0, 0.0);
                d.srw_ebw = mean_wind(inner, prof.ebottom, elh, rstu, rstv);
            }
        }

        // --- mean / storm-relative winds ----------------------------------
        d.mean_1km = vect(mean_wind(inner, sfc_pres, p1km, 0.0, 0.0));
        d.mean_3km = vect(mean_wind(inner, sfc_pres, p3km, 0.0, 0.0));
        let mean_6km = mean_wind(inner, sfc_pres, p6km, 0.0, 0.0);
        d.mean_6km = vect(mean_6km);
        d.mean_8km = vect(mean_wind(inner, sfc_pres, p8km, 0.0, 0.0));
        d.mean_lcl_el = vect(mean_wind(inner, mupcl.lclpres, mupcl.elpres, 0.0, 0.0));
        d.mean_wind_sfc_500m = mean_wind(inner, sfc_pres, p500m, 0.0, 0.0);
        d.srw_1km = vect(mean_wind(inner, sfc_pres, p1km, rstu, rstv));
        d.srw_3km = vect(mean_wind(inner, sfc_pres, p3km, rstu, rstv));
        d.srw_6km = vect(mean_wind(inner, sfc_pres, p6km, rstu, rstv));
        d.srw_8km = vect(mean_wind(inner, sfc_pres, p8km, rstu, rstv));
        d.srw_4_5km = vect(mean_wind(inner, p4km, p5km, rstu, rstv));
        d.srw_lcl_el = vect(mean_wind(inner, mupcl.lclpres, mupcl.elpres, rstu, rstv));
        d.wind1km = inner.interp_vec(p1km);
        d.wind6km = inner.interp_vec(p6km);

        // Full ConvectiveProfile semantics use the profile's active Bunkers
        // storm motion for both SFC-500 m storm-relative diagnostics.
        d.srw_sfc_500m = mean_wind(inner, sfc_pres, p500m, rstu, rstv);

        // --- helicity ------------------------------------------------------
        d.srh500 = helicity(inner, 0.0, 500.0, rstu, rstv);
        d.srh1km = helicity(inner, 0.0, 1000.0, rstu, rstv);
        d.srh3km = helicity(inner, 0.0, 3000.0, rstu, rstv);
        d.right_esrh = prof.right_esrh;
        let left_esrh = if qc(prof.ebotm) && qc(prof.etopm) {
            helicity(inner, prof.ebotm, prof.etopm, lstu, lstv)
        } else {
            f64::NAN
        };

        // --- corfidi / critical angle / BRN shear -------------------------
        if let Ok((upu, upv, dnu, dnv)) = winds::corfidi_mcs_motion(inner) {
            d.corfidi_up = (upu, upv);
            d.corfidi_dn = (dnu, dnv);
        }
        // ConvectiveProfile publishes critical angle only when an effective
        // inflow layer exists; the same storm motion remains valid for the
        // fixed-layer diagnostics above when no effective layer is found.
        if qc(prof.ebottom) && qc(prof.etop) {
            d.right_critical_angle = winds::critical_angle(inner, rstu, rstv).unwrap_or(f64::NAN);
        }
        // Port of `params.bulk_rich` (MU parcel branch: sfc..6 km layer).
        {
            let pblw = inner.pres_at_height(inner.sfc_height() + 500.0);
            let (mnlu, mnlv) = mean_wind(inner, sfc_pres, pblw, 0.0, 0.0);
            let (mnuu, mnuv) = mean_wind(inner, sfc_pres, p6km, 0.0, 0.0);
            let s = kts2ms(mag(mnuu - mnlu, mnuv - mnlv));
            d.brnshear = s * s / 2.0;
        }

        // --- thermo block ---------------------------------------------------
        d.pwat = indices::precip_water(inner, None, None).unwrap_or(f64::NAN);
        d.k_idx = indices::k_index(inner).unwrap_or(f64::NAN);
        d.mean_mixr = indices::mean_mixratio(inner, None, None).unwrap_or(f64::NAN);
        d.low_rh = indices::mean_relh(inner, None, None).unwrap_or(f64::NAN);
        d.mid_rh = indices::mean_relh(inner, Some(sfc_pres - 150.0), Some(sfc_pres - 350.0))
            .unwrap_or(f64::NAN);
        d.totals_totals = indices::t_totals(inner).unwrap_or(f64::NAN);
        d.max_t_f = thermo::ctof(indices::max_temp(inner, None).unwrap_or(f64::NAN));
        d.conv_t_f = thermo::ctof(convective_temp(&cape_prof, inner));

        // Theta-e per level with SHARPpy's formulation (`thermo.thetae`);
        // sharprs's Profile.thetae array uses a different saturated lift.
        let thetae: Vec<f64> = (0..inner.num_levels())
            .map(|i| {
                let (p, t, td) = (inner.pres[i], inner.tmpc[i], inner.dwpc[i]);
                if qc(p) && qc(t) && qc(td) {
                    thermo::thetae(p, t, td)
                } else {
                    f64::NAN
                }
            })
            .collect();
        // TEI: max minus min theta-e in the lowest 400 hPa (params.tei).
        {
            let (mut te_max, mut te_min) = (f64::NAN, f64::NAN);
            for i in 0..inner.num_levels() {
                if !qc(inner.pres[i]) || inner.pres[i] < sfc_pres - 400.0 || !thetae[i].is_finite()
                {
                    continue;
                }
                if te_max.is_nan() || thetae[i] > te_max {
                    te_max = thetae[i];
                }
                if te_min.is_nan() || thetae[i] < te_min {
                    te_min = thetae[i];
                }
            }
            d.tei = te_max - te_min;
        }
        // Theta-e difference in the lowest 3 km (params.thetae_diff).
        {
            let (mut te_max, mut te_min) = (f64::NAN, f64::NAN);
            let (mut p_max, mut p_min) = (f64::NAN, f64::NAN);
            for i in 0..inner.num_levels() {
                if !qc(inner.hght[i]) || inner.to_agl(inner.hght[i]) > 3000.0 {
                    continue;
                }
                if !thetae[i].is_finite() {
                    continue;
                }
                if te_max.is_nan() || thetae[i] > te_max {
                    te_max = thetae[i];
                    p_max = inner.pres[i];
                }
                if te_min.is_nan() || thetae[i] < te_min {
                    te_min = thetae[i];
                    p_min = inner.pres[i];
                }
            }
            if te_max.is_finite() && te_min.is_finite() {
                d.thetae_diff = if p_max < p_min { 0.0 } else { te_max - te_min };
            }
        }

        // --- lapse rates ----------------------------------------------------
        d.lapserate_3km = crate::extras::lapse_rate_agl(inner, 0.0, 3000.0);
        d.lapserate_3_6km = crate::extras::lapse_rate_agl(inner, 3000.0, 6000.0);
        d.lapserate_850_500 = indices::lapse_rate(inner, 850.0, 500.0, true).unwrap_or(f64::NAN);
        d.lapserate_700_500 = indices::lapse_rate(inner, 700.0, 500.0, true).unwrap_or(f64::NAN);
        // The local sharpmod companion established these two shallow rows
        // with plain temperature rather than SHARPpy's general vtmp lapse
        // rate. Preserve that public display contract.
        d.lapserate_sfc_500m = crate::extras::temperature_lapse_rate_agl(inner, 0.0, 500.0);
        d.lapserate_sfc_1km = crate::extras::temperature_lapse_rate_agl(inner, 0.0, 1000.0);

        // --- DCAPE / downrush -----------------------------------------------
        // Display convention is SHARPpy's (positive J/kg).
        d.dcape = if qc(prof.dcape) {
            prof.dcape.abs()
        } else {
            f64::NAN
        };
        // Downrush temp = downdraft parcel temperature at the surface (the
        // highest-pressure point of the trace), in Fahrenheit.
        {
            let mut best = f64::NAN;
            let mut best_p = f64::NEG_INFINITY;
            for (t, p) in prof.dpcl_ttrace.iter().zip(prof.dpcl_ptrace.iter()) {
                if qc(*p) && qc(*t) && *p > best_p {
                    best_p = *p;
                    best = *t;
                }
            }
            d.drush_f = thermo::ctof(best);
        }

        // --- composite indices ----------------------------------------------
        let shr06_kt = mag(d.sfc_6km_shear.0, d.sfc_6km_shear.1);
        let shr06_ms = kts2ms(shr06_kt);
        d.esp = composites::esp(mlpcl.b3km, d.lapserate_3km, mlpcl.bplus).unwrap_or(f64::NAN);
        d.wndg = {
            let mw = kts2ms(mag_c(mean_wind(inner, p1km, p3_5km, 0.0, 0.0)));
            composites::wndg(mlpcl.bplus, d.lapserate_3km, mw, mlpcl.bminus).unwrap_or(f64::NAN)
        };
        d.dcp =
            composites::dcp(d.dcape, mupcl.bplus, shr06_kt, mag_c(mean_6km)).unwrap_or(f64::NAN);
        d.sig_severe = composites::sig_severe(mlpcl.bplus, shr06_ms).unwrap_or(f64::NAN);
        d.ship = {
            let mumr = thermo::mixratio(mupcl.pres, mupcl.dwpc);
            // SHARPpy passes the freezing level in m MSL (interp.hght of the
            // 0 C level), not AGL.
            let frz_lvl = indices::temp_lvl(inner, 0.0, false)
                .map(|p| inner.interp_hght(p))
                .unwrap_or(f64::NAN);
            let h5 = inner.interp_tmpc(500.0);
            composites::ship(
                mupcl.bplus,
                mumr,
                d.lapserate_700_500,
                h5,
                shr06_ms,
                frz_lvl,
            )
            .unwrap_or(f64::NAN)
        };
        if qc(prof.ebottom) && qc(prof.etop) {
            let ebwd_ms = kts2ms(ebwspd);
            d.right_scp = composites::scp(mupcl.bplus, d.right_esrh, ebwd_ms).unwrap_or(f64::NAN);
            d.left_scp = composites::scp(mupcl.bplus, left_esrh, ebwd_ms).unwrap_or(f64::NAN);
            let stp_esrh = if prof.latitude() < 0.0 {
                -left_esrh
            } else {
                d.right_esrh
            };
            d.stp_cin =
                composites::stp_cin(mlpcl.bplus, stp_esrh, ebwd_ms, mlpcl.lclhght, mlpcl.bminus)
                    .unwrap_or(f64::NAN);
            if prof.latitude() < 0.0 {
                d.stp_cin = -d.stp_cin;
            }
        } else {
            d.right_scp = 0.0;
            d.left_scp = 0.0;
            d.stp_cin = 0.0;
        }
        let fixed_srh = if prof.latitude() < 0.0 {
            helicity(inner, 0.0, 1000.0, lstu, lstv)
        } else {
            d.srh1km
        };
        d.stp_fixed = composites::stp_fixed(sfcpcl.bplus, sfcpcl.lclhght, fixed_srh, shr06_ms)
            .unwrap_or(f64::NAN);
        // With a constant wind column, Bunkers' deviation vector is
        // undefined (0/0). NumPy's legacy helicity path sums the resulting
        // empty positive/negative layer selections to zero, which makes all
        // shear/SRH severe composites exactly zero. Preserve that externally
        // visible result while leaving genuinely absent winds missing.
        let constant_wind = crate::extras::has_constant_wind(inner);
        if constant_wind {
            d.right_scp = 0.0;
            d.left_scp = 0.0;
            d.stp_cin = if prof.latitude() < 0.0 { -0.0 } else { 0.0 };
            d.stp_fixed = 0.0;
        }
        d.sweat = {
            let td850 = inner.interp_dwpc(850.0);
            let (dir850, spd850) = inner.interp_vec(850.0);
            let (dir500, spd500) = inner.interp_vec(500.0);
            composites::sweat(td850, d.totals_totals, dir850, spd850, dir500, spd500)
                .unwrap_or(f64::NAN)
        };
        d.mburst = {
            let vt = indices::v_totals(inner).unwrap_or(f64::NAN);
            let sfc_te = thermo::thetae(sfcpcl.pres, sfcpcl.tmpc, sfcpcl.dwpc);
            composites::mburst(
                sfcpcl.bplus,
                sfcpcl.li5,
                d.lapserate_3km,
                vt,
                d.dcape,
                d.pwat,
                d.thetae_diff,
                sfc_te,
            )
            .map(|v| v as f64)
            .unwrap_or(f64::NAN)
        };
        // MMP replicates SHARPpy `params.mmp` exactly, including its
        // upper-triangle skip (`if b < t: continue`) over the candidate
        // low-level / 6-10 km level pairs.
        let lr38 = indices::lapse_rate(inner, 3000.0, 8000.0, false).unwrap_or(f64::NAN);
        let mnwind_3_12_ms = kts2ms(mag_c(mean_wind(inner, p3km, p12km, 0.0, 0.0)));
        let (mmp_shear_ms, mcs_shear_ms) = max_bulk_shear(inner);
        d.mmp = if qc(mupcl.bplus) && mupcl.bplus < 100.0 {
            0.0
        } else {
            composites::mmp(mupcl.bplus, mmp_shear_ms, lr38, mnwind_3_12_ms).unwrap_or(f64::NAN)
        };
        // MCS index = the Coniglio regression's linear predictor (sharpmod
        // `derived.mcs_index`), using the max shear over *all* level pairs.
        // A LOW value is the MCS end (MMP = 1 / (1 + e^index)); the board
        // colours it that way. A missing MU parcel is `--`, not the -9999
        // sentinel run through the sum (+6.2, a quiet-looking junk number).
        d.mcs_index = if qc(mupcl.bplus) {
            13.0 + (-4.59e-2 * mcs_shear_ms)
                + (-1.16 * lr38)
                + (-6.17e-4 * mupcl.bplus)
                + (-0.17 * mnwind_3_12_ms)
        } else {
            f64::NAN
        };

        // --- SHARPpy-Reimagined derived composites --------------------------
        // LRGHAIL's lightweight compatibility oracle explicitly uses
        // non-parcel Bunkers motion. EHI, however, reuses ``prof.srwind`` when
        // a full ConvectiveProfile has already established parcel Bunkers
        // motion, so its authoritative full-profile result uses rstu/rstv.
        let npb = winds::non_parcel_bunkers_motion(inner).unwrap_or((
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
        ));
        // EHI (sharpmod `derived.ehi`): SBCAPE x SRH with the active cached
        // Bunkers right mover.
        d.ehi_0_1km = composites::ehi(sfcpcl.bplus, helicity(inner, 0.0, 1000.0, rstu, rstv))
            .unwrap_or(f64::NAN);
        d.ehi_0_3km = composites::ehi(sfcpcl.bplus, helicity(inner, 0.0, 3000.0, rstu, rstv))
            .unwrap_or(f64::NAN);
        // VGP (sharpmod): sqrt(SBCAPE) * (0-4 km shear [m/s] / 4000 m).
        d.vgp = {
            let sh = kts2ms(mag_c(shear(inner, sfc_pres, p4km)));
            if !qc(sfcpcl.bplus) || !sh.is_finite() || sfcpcl.bplus < 0.0 {
                f64::NAN
            } else if sfcpcl.bplus == 0.0 {
                0.0
            } else {
                sfcpcl.bplus.sqrt() * sh / 4000.0
            }
        };
        // Peskov (sharpmod): K-index + SBCAPE/1000 - DD700/5.
        d.peskov = {
            let t700 = inner.interp_tmpc(700.0);
            let td700 = inner.interp_dwpc(700.0);
            d.k_idx + sfcpcl.bplus / 1000.0 - (t700 - td700) / 5.0
        };
        // NCAPE (sharpmod, Blanchard 1998): MUCAPE / (EL - LFC) depth.
        d.ncape = {
            let depth = mupcl.elhght - mupcl.lfchght;
            if qc(mupcl.bplus) && depth.is_finite() && depth > 0.0 {
                mupcl.bplus / depth
            } else {
                f64::NAN
            }
        };
        // LRGHAIL (params.lhp); its oracle profile carries the *non-parcel*
        // Bunkers motion, so the SR wind directions use `npb` here.
        d.lrghail = {
            let (zbot, ztop) = {
                let pb = indices::temp_lvl(inner, -10.0, false).unwrap_or(sfc_pres);
                let pt = indices::temp_lvl(inner, -30.0, false).unwrap_or(sfc_pres);
                (inner.interp_hght(pb), inner.interp_hght(pt))
            };
            let thk_hgz = ztop - zbot;
            let shear_el = kts2ms(mag_c(shear(inner, sfc_pres, mupcl.elpres)));
            let grw_el_dir = inner.interp_vec(mupcl.elpres).0;
            let grw_36_dir = vect(mean_wind(inner, p3km, p6km, 0.0, 0.0)).0;
            let srw_01_dir = vect(mean_wind(inner, sfc_pres, p1km, npb.0, npb.1)).0;
            let srw_36_dir = vect(mean_wind(inner, p3km, p6km, npb.0, npb.1)).0;
            indices::lhp(
                mupcl.bplus,
                shr06_ms,
                d.lapserate_700_500,
                thk_hgz,
                shear_el,
                grw_el_dir - grw_36_dir,
                srw_36_dir - srw_01_dir,
            )
        };
        // LSCP (sharpmod `derived.left_supercell_composite`).
        d.lscp = if constant_wind {
            0.0
        } else if !qc(prof.ebottom) || !qc(prof.etop) {
            0.0
        } else if !qc(mupcl.bplus)
            || !qc(mupcl.bminus)
            || !left_esrh.is_finite()
            || !ebwspd.is_finite()
        {
            f64::NAN
        } else if mupcl.bplus <= 0.0 {
            0.0
        } else {
            let mut ebwd_ms = ebwspd / KTS_PER_MS;
            if ebwd_ms > 20.0 {
                ebwd_ms = 20.0;
            } else if ebwd_ms < 10.0 {
                ebwd_ms = 0.0;
            }
            let mucin_term = if mupcl.bminus > -40.0 {
                1.0
            } else {
                -40.0 / mupcl.bminus
            };
            (mupcl.bplus / 1000.0) * (left_esrh / 50.0) * (ebwd_ms / 20.0) * mucin_term
        };
        // NSTP requires an ambient surface vorticity supplied by the data
        // source; a bare sounding has none, so it stays NaN (matches the
        // reference stack, which reports it missing for this input).
        d.nstp = f64::NAN;
        // Layer CAPE integrals (sharpmod `params.layer_cape_*`): a surface
        // parcel's positive buoyancy accumulated over the bounded layer.
        d.cape_0_3km = layer_cape(&cape_prof, inner, sfc_pres, p3km);
        d.cape_0_6km = layer_cape(&cape_prof, inner, sfc_pres, p6km);
        d.hgz_cape = {
            let pbot = indices::temp_lvl(inner, -10.0, false).unwrap_or(f64::NAN);
            let ptop = indices::temp_lvl(inner, -30.0, false).unwrap_or(f64::NAN);
            if !qc(pbot) || !qc(ptop) || pbot <= ptop || sfc_pres <= ptop {
                f64::NAN
            } else {
                let cape_to_top = layer_cape(&cape_prof, inner, sfc_pres, ptop);
                let cape_to_bot = if sfc_pres > pbot {
                    layer_cape(&cape_prof, inner, sfc_pres, pbot)
                } else {
                    0.0
                };
                let hgz = cape_to_top - cape_to_bot;
                if hgz.is_finite() {
                    hgz.max(0.0)
                } else {
                    f64::NAN
                }
            }
        };
        d.wbz_height = indices::wet_bulb_zero(inner).unwrap_or(f64::NAN);
        d.ecape = ecape(inner, mupcl);
        // MOSHE's EBWD term needs an effective inflow layer. Without one it is
        // unavailable (`--`), like every other no-layer effective field.
        // sharpmod instead reads ConvectiveProfile's literal MISSING (-9999 kt)
        // ebwspd, so (EBWD - 8) / 10 is about -515 and the product is a large
        // junk number of either sign (hundreds to thousands on a CAPE-0 day).
        // That sharpmod value is kept only in `modified_sherbe_sharpmod` for
        // parity proofs.
        let no_effective_layer = !qc(prof.ebottom) || !qc(prof.etop);
        d.modified_sherbe_sharpmod = modified_sherbe(
            inner,
            d.lapserate_3km,
            p1_5km,
            if no_effective_layer { -9999.0 } else { ebwspd },
        );
        d.modified_sherbe = if no_effective_layer {
            f64::NAN
        } else {
            d.modified_sherbe_sharpmod
        };

        // --- strips / insets -------------------------------------------------
        let (temp_adv, bounds) = inferred_temp_adv(inner);
        d.temp_adv = temp_adv;
        d.temp_adv_bounds = bounds;
        let (traj, tilt) = parcel_traj(inner, mupcl, rstu, rstv);
        d.slinky_traj = traj;
        d.slinky_tilt = tilt;

        d
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Small kinematic wrappers (NaN-on-error views of `sharprs::winds`)
// ═══════════════════════════════════════════════════════════════════════════

const KTS_PER_MS: f64 = 1.943_844_492_440_604_6;

#[inline]
fn mag(u: f64, v: f64) -> f64 {
    (u * u + v * v).sqrt()
}

#[inline]
fn mag_c(c: Comp) -> f64 {
    mag(c.0, c.1)
}

#[inline]
fn vect(c: Comp) -> Vect {
    comp2vec(c.0, c.1)
}

fn mean_wind(inner: &sharprs::Profile, pbot: f64, ptop: f64, stu: f64, stv: f64) -> Comp {
    if !qc(pbot) || !qc(ptop) || !qc(stu) || !qc(stv) {
        return (f64::NAN, f64::NAN);
    }
    // SHARPpy samples with `np.arange(pbot, ptop - 1, -1)`.  When an
    // effective-inflow layer contains exactly one pressure level, that grid
    // contains only `pbot`; sharprs's inclusive loop also sampled `ptop - 1`
    // and biased the mean (and both storm-relative means) toward the next
    // level above it.
    if (pbot - ptop).abs() <= 1.0e-9 {
        let (u, v) = inner.interp_wind(pbot);
        return (u - stu, v - stv);
    }
    winds::mean_wind(inner, pbot, ptop, -1.0, stu, stv).unwrap_or((f64::NAN, f64::NAN))
}

fn shear(inner: &sharprs::Profile, pbot: f64, ptop: f64) -> Comp {
    if !qc(pbot) || !qc(ptop) {
        return (f64::NAN, f64::NAN);
    }
    winds::wind_shear(inner, pbot, ptop).unwrap_or((f64::NAN, f64::NAN))
}

fn helicity(inner: &sharprs::Profile, lower_agl: f64, upper_agl: f64, stu: f64, stv: f64) -> f64 {
    crate::extras::helicity(inner, lower_agl, upper_agl, stu, stv).0
}

/// Linear interpolation of `field` against `xs` (both may contain NaN; pairs
/// with either value missing are dropped). NaN outside the valid range:
/// the same behaviour as sharpmod's `generic_interp_hght`.
fn interp_h(xs: &[f64], field: &[f64], target: f64) -> f64 {
    let mut pts: Vec<(f64, f64)> = xs
        .iter()
        .zip(field.iter())
        .filter(|(x, f)| x.is_finite() && f.is_finite())
        .map(|(x, f)| (*x, *f))
        .collect();
    if pts.len() < 2 || !target.is_finite() {
        return f64::NAN;
    }
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    if target < pts[0].0 || target > pts[pts.len() - 1].0 {
        return f64::NAN;
    }
    for w in pts.windows(2) {
        if target >= w[0].0 && target <= w[1].0 {
            let (x0, y0) = w[0];
            let (x1, y1) = w[1];
            if (x1 - x0).abs() < 1e-12 {
                return y0;
            }
            return y0 + (y1 - y0) * (target - x0) / (x1 - x0);
        }
    }
    f64::NAN
}

fn user_lpl(pres: f64, tmpc: f64, dwpc: f64) -> LiftedParcelLevel {
    LiftedParcelLevel {
        pres,
        tmpc,
        dwpc,
        parcel_type: SharprsParcelType::UserDefined { pres, tmpc, dwpc },
    }
}

/// Positive buoyancy of the surface parcel integrated between `pbot`/`ptop`
/// (port of sharpmod `params._layer_cape`, which anchors SHARPpy's `cape()`
/// at the surface with its default user parcel).
fn layer_cape(cape_prof: &cape::Profile, inner: &sharprs::Profile, pbot: f64, ptop: f64) -> f64 {
    if !qc(pbot) || !qc(ptop) || pbot <= ptop {
        return f64::NAN;
    }
    let tmpc = inner.tmpc[inner.sfc];
    let dwpc = inner.dwpc[inner.sfc];
    if !qc(tmpc) || !qc(dwpc) {
        return f64::NAN;
    }
    let lpl = user_lpl(inner.sfc_pressure(), tmpc, dwpc);
    let _ = cape_prof; // retained in the signature for the other parcel paths
    crate::extras::cape_bounded_sharppy(inner, lpl.pres, lpl.tmpc, lpl.dwpc, pbot, ptop).0
}

/// Port of SHARPpy `params.convective_temp` (mincinh = 0): iteratively warm
/// the surface until the lifted parcel's CIN vanishes.
fn convective_temp(_cape_prof: &cape::Profile, inner: &sharprs::Profile) -> f64 {
    let mincinh = 0.0;
    let surface_pressure = inner.sfc_pressure();
    let mmr = match crate::extras::sharppy_mean_mixratio(
        inner,
        surface_pressure,
        surface_pressure - 100.0,
    ) {
        Some(v) if v.is_finite() => v,
        _ => return f64::NAN,
    };
    let pres = surface_pressure;
    let mut tmpc = inner.tmpc[inner.sfc];
    let dwpc = thermo::temp_at_mixrat(mmr, pres);
    if !qc(pres) || !qc(tmpc) || !dwpc.is_finite() {
        return f64::NAN;
    }
    let lift =
        |t: f64| -> (f64, f64) { crate::extras::cape_truncated_sharppy(inner, pres, t, dwpc) };
    // Quick viability check: if 25 C of heating cannot remove the cap, bail.
    let (bp, bm) = lift(tmpc + 25.0);
    if bp == 0.0 || !qc(bm) || bm < mincinh {
        return f64::NAN;
    }
    let excess = dwpc - tmpc;
    if excess > 0.0 {
        tmpc = tmpc + excess + 4.0;
    }
    let (mut bp, mut bm) = lift(tmpc);
    if bp == 0.0 || !qc(bm) {
        bm = f64::NAN;
    }
    let mut iters = 0;
    while !qc(bm) || bm < mincinh {
        if qc(bm) && bm < -100.0 {
            tmpc += 2.0;
        } else {
            tmpc += 0.5;
        }
        let r = lift(tmpc);
        bp = r.0;
        bm = r.1;
        if bp == 0.0 {
            bm = f64::NAN;
        }
        iters += 1;
        if iters > 200 {
            return f64::NAN;
        }
    }
    tmpc
}

/// Max bulk shear between the lowest-1 km levels and the 6-10 km levels.
/// Returns `(restricted, all_pairs)` in m/s: SHARPpy's `params.mmp` skips
/// pairs with `b < t` (a quirk kept for parity), while sharpmod's
/// `mcs_index` scans every pair.
fn max_bulk_shear(inner: &sharprs::Profile) -> (f64, f64) {
    let mut low: Vec<usize> = Vec::new();
    let mut high: Vec<usize> = Vec::new();
    for i in 0..inner.num_levels() {
        if !qc(inner.hght[i]) || !qc(inner.pres[i]) {
            continue;
        }
        let agl = inner.to_agl(inner.hght[i]);
        if agl <= 1000.0 {
            low.push(i);
        } else if (6000.0..10000.0).contains(&agl) {
            high.push(i);
        }
    }
    if low.is_empty() || high.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mut restricted = f64::NAN;
    let mut all_pairs = f64::NAN;
    for (b, &bi) in low.iter().enumerate() {
        for (t, &ti) in high.iter().enumerate() {
            let s = match winds::wind_shear(inner, inner.pres[bi], inner.pres[ti]) {
                Ok((u, v)) => mag(u, v),
                Err(_) => continue,
            };
            if !s.is_finite() {
                continue;
            }
            if all_pairs.is_nan() || s > all_pairs {
                all_pairs = s;
            }
            if b >= t && (restricted.is_nan() || s > restricted) {
                restricted = s;
            }
        }
    }
    (kts2ms(restricted), kts2ms(all_pairs))
}

// ═══════════════════════════════════════════════════════════════════════════
// Modified SHERBE (sharpmod `derived.modified_sherbe`: the SPC MOSHE)
// ═══════════════════════════════════════════════════════════════════════════

fn modified_sherbe(inner: &sharprs::Profile, lllr: f64, p1_5km: f64, ebwspd_kt: f64) -> f64 {
    let s15 = kts2ms(mag_c(shear(inner, inner.sfc_pressure(), p1_5km)));
    let eshr = ebwspd_kt / KTS_PER_MS;
    let maxtevv = max_thetae_vertical_velocity(inner);
    if !lllr.is_finite() || !s15.is_finite() || !eshr.is_finite() || !maxtevv.is_finite() {
        return f64::NAN;
    }
    ((lllr - 4.0).powi(2) / 4.0)
        * ((s15 - 8.0) / 10.0)
        * ((eshr - 8.0) / 10.0)
        * ((maxtevv + 10.0) / 9.0)
}

/// MOSHE's MAXTEVV term: max theta-e decrease x upward motion over 2-km-deep
/// layers with tops from 2 to 6 km AGL every 500 m.
fn max_thetae_vertical_velocity(inner: &sharprs::Profile) -> f64 {
    // Omega column with the -9999 sentinel masked (sharpmod masks <= -9000).
    let omeg: Vec<f64> = inner
        .omeg
        .iter()
        .map(|&o| {
            if o.is_finite() && o > -9000.0 {
                o
            } else {
                f64::NAN
            }
        })
        .collect();
    if omeg.iter().all(|o| o.is_nan()) {
        return f64::NAN;
    }
    let thetae_at_agl = |h: f64| -> f64 {
        let msl = inner.to_msl(h);
        let p = inner.pres_at_height(msl);
        let t = interp_h(&inner.hght, &inner.tmpc, msl);
        let td = interp_h(&inner.hght, &inner.dwpc, msl);
        if p.is_finite() && t.is_finite() && td.is_finite() {
            thermo::thetae(p, t, td)
        } else {
            f64::NAN
        }
    };
    let mut best = f64::NAN;
    for k in 0..=8 {
        let top = 2000.0 + 500.0 * k as f64;
        let bottom = top - 2000.0;
        let te_b = thetae_at_agl(bottom);
        let te_t = thetae_at_agl(top);
        let om_t = interp_h(&inner.hght, &omeg, inner.to_msl(top));
        if !te_b.is_finite() || !te_t.is_finite() || !om_t.is_finite() {
            continue;
        }
        let value = (te_b - te_t) / 2.0 * -om_t;
        if value.is_finite() && (best.is_nan() || value > best) {
            best = value;
        }
    }
    best
}

// ═══════════════════════════════════════════════════════════════════════════
// Analytic ECAPE, port of the `ecape-rs` solver the reference stack ships
// (`rw_ecape_analytic`: Peters et al. 2023 analytic formula), with the parcel
// quantities (CAPE, LFC, EL) taken from the sharprs MU parcel.
// ═══════════════════════════════════════════════════════════════════════════

const EC_RD: f64 = 287.04;
const EC_RV: f64 = 461.5;
const EC_PHI: f64 = EC_RD / EC_RV;
const EC_CPD: f64 = 1005.0;
const EC_CPV: f64 = 1870.0;
const EC_CPL: f64 = 4190.0;
const EC_G: f64 = 9.81;
const EC_LV: f64 = 2_501_000.0;
const EC_TTRIP: f64 = 273.15;
const EC_VPR: f64 = 611.2;
const EC_KTS_TO_MS: f64 = 0.514_444_444_444_444_5;

fn ec_spec_hum(p_pa: f64, td_c: f64) -> f64 {
    let vp = 611.2 * ((17.67 * td_c) / (td_c + 243.5)).exp();
    0.62197 * vp / (p_pa - 0.37803 * vp)
}

/// Saturation mixing ratio over liquid (ecape-rs `r_sat`, ice_flag = 0).
fn ec_r_sat(t_k: f64, p_pa: f64) -> f64 {
    let term1 = (EC_CPV - EC_CPL) / EC_RV;
    let term2 = (EC_LV - EC_TTRIP * (EC_CPV - EC_CPL)) / EC_RV;
    let esl =
        ((t_k - EC_TTRIP) * term2 / (t_k * EC_TTRIP)).exp() * EC_VPR * (t_k / EC_TTRIP).powf(term1);
    EC_PHI * esl / (p_pa - esl).max(1e-9)
}

fn ec_mse(z_m: f64, t_k: f64, qv: f64) -> f64 {
    EC_CPD * t_k + EC_G * z_m + EC_LV * qv
}

/// Log-pressure interpolation with end clamping (ecape-rs `interp_log_pressure`).
fn ec_interp_logp(target: f64, pres: &[f64], vals: &[f64]) -> f64 {
    let n = pres.len();
    if n == 0 {
        return 0.0;
    }
    if target >= pres[0] {
        return vals[0];
    }
    if target <= pres[n - 1] {
        return vals[n - 1];
    }
    for i in 1..n {
        if pres[i] <= target {
            let frac = (target.ln() - pres[i - 1].ln()) / (pres[i].ln() - pres[i - 1].ln());
            return vals[i - 1] + frac * (vals[i] - vals[i - 1]);
        }
    }
    vals[n - 1]
}

/// Pressure at height, linear in pressure, clamped (ecape-rs `metpy_pressure_at_height`).
fn ec_p_at_h(pres: &[f64], hgts: &[f64], z: f64) -> f64 {
    let n = hgts.len();
    if z <= hgts[0] {
        return pres[0];
    }
    if z >= hgts[n - 1] {
        return pres[n - 1];
    }
    for i in 1..n {
        if z <= hgts[i] {
            let f = (z - hgts[i - 1]) / (hgts[i] - hgts[i - 1]);
            return pres[i - 1] + f * (pres[i] - pres[i - 1]);
        }
    }
    pres[n - 1]
}

fn ec_close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-8 + 1e-5 * b.abs()
}

/// MetPy-style pressure-weighted continuous layer average over an AGL height
/// layer (ecape-rs `metpy_weighted_continuous_average_height`).
fn ec_wavg(pres: &[f64], hagl: &[f64], vals: &[f64], bottom: f64, depth: f64) -> f64 {
    let top = bottom + depth;
    let bp = ec_p_at_h(pres, hagl, bottom);
    let tp = ec_p_at_h(pres, hagl, top);
    let mut layer: Vec<f64> = pres
        .iter()
        .copied()
        .filter(|&p| (p < bp || ec_close(p, bp)) && (p > tp || ec_close(p, tp)))
        .collect();
    if !layer.iter().any(|&p| ec_close(p, bp)) {
        layer.push(bp);
    }
    if !layer.iter().any(|&p| ec_close(p, tp)) {
        layer.push(tp);
    }
    layer.sort_by(|a, b| b.partial_cmp(a).unwrap());
    if layer.len() < 2 {
        return vals[0];
    }
    let lv: Vec<f64> = layer
        .iter()
        .map(|&p| ec_interp_logp(p, pres, vals))
        .collect();
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 1..layer.len() {
        let dp = layer[i] - layer[i - 1];
        num += 0.5 * (lv[i] + lv[i - 1]) * dp;
        den += dp;
    }
    if den.abs() > 1e-12 { num / den } else { lv[0] }
}

/// MetPy-parity Bunkers right mover (m/s): ecape-rs `bunkers_storm_motion`.
fn ec_bunkers_rm(pres: &[f64], hagl: &[f64], u: &[f64], v: &[f64]) -> (f64, f64) {
    let mean = (
        ec_wavg(pres, hagl, u, 0.0, 6000.0),
        ec_wavg(pres, hagl, v, 0.0, 6000.0),
    );
    let w500 = (
        ec_wavg(pres, hagl, u, 0.0, 500.0),
        ec_wavg(pres, hagl, v, 0.0, 500.0),
    );
    let w5500 = (
        ec_wavg(pres, hagl, u, 5500.0, 500.0),
        ec_wavg(pres, hagl, v, 5500.0, 500.0),
    );
    let sh = (w5500.0 - w500.0, w5500.1 - w500.1);
    let m = mag(sh.0, sh.1);
    if m < 1e-12 {
        return mean;
    }
    let dev = 7.5 / m;
    (mean.0 + sh.1 * dev, mean.1 - sh.0 * dev)
}

/// Mean 0-1 km AGL storm-relative wind speed (m/s): ecape-rs `calc_sr_wind`.
fn ec_sr_wind(hgts: &[f64], u: &[f64], v: &[f64], su: f64, sv: f64) -> f64 {
    let z0 = hgts[0];
    let mut sum = 0.0;
    let mut n = 0usize;
    for i in 0..hgts.len() {
        let agl = hgts[i] - z0;
        if (0.0..=1000.0).contains(&agl) {
            sum += mag(u[i] - su, v[i] - sv);
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { sum / n as f64 }
}

/// NCAPE dilution integral, clamped at >= 0: ecape-rs `compute_ncape_reference`.
fn ec_ncape(h: &[f64], p_pa: &[f64], t_k: &[f64], qv: &[f64], lfc_m: f64, el_m: f64) -> f64 {
    ec_ncape_with_saturation(h, p_pa, t_k, qv, lfc_m, el_m, ec_r_sat)
}

fn ec_ncape_with_saturation<F>(
    h: &[f64],
    p_pa: &[f64],
    t_k: &[f64],
    qv: &[f64],
    lfc_m: f64,
    el_m: f64,
    mut saturation_mixing_ratio: F,
) -> f64
where
    F: FnMut(f64, f64) -> f64,
{
    if el_m <= lfc_m {
        return 0.0;
    }

    let nearest = |target: f64| -> usize {
        let mut best = 0usize;
        let mut bd = f64::INFINITY;
        for (i, &z) in h.iter().enumerate() {
            let dd = (z - target).abs();
            if dd < bd {
                bd = dd;
                best = i;
            }
        }
        best
    };
    let ind_lfc = nearest(lfc_m);
    let ind_el = nearest(el_m);
    if ind_el <= ind_lfc + 1 {
        return 0.0;
    }

    // Only the surface-through-EL layer contributes to NCAPE. Upper-
    // stratospheric temperatures can be physically valid while liquid-water
    // saturation is undefined because saturation vapor pressure meets or
    // exceeds total pressure. Do not evaluate those unused levels.
    let n = ind_el + 1;
    if p_pa.len() < n || t_k.len() < n || qv.len() < n {
        return f64::NAN;
    }
    let mse0: Vec<f64> = (0..n).map(|i| ec_mse(h[i], t_k[i], qv[i])).collect();
    let mse0_star: Vec<f64> = (0..n)
        .map(|i| {
            let rsat = saturation_mixing_ratio(t_k[i], p_pa[i]);
            ec_mse(h[i], t_k[i], rsat / (1.0 + rsat))
        })
        .collect();
    let mut mse0bar = vec![0.0; n];
    mse0bar[0] = mse0[0];
    for iz in 1..n {
        let mut sum = 0.0;
        for j in 0..iz {
            sum += (mse0[j] + mse0[j + 1]) * (h[j + 1] - h[j]);
        }
        mse0bar[iz] = 0.5 * sum / (h[iz] - h[0]);
    }
    let int_arg: Vec<f64> = (0..n)
        .map(|i| -(EC_G / (EC_CPD * t_k[i])) * (mse0bar[i] - mse0_star[i]))
        .collect();
    let mut ncape = 0.0;
    for i in ind_lfc..(ind_el - 1) {
        ncape += (0.5 * int_arg[i] + 0.5 * int_arg[i + 1]) * (h[i + 1] - h[i]);
    }
    ncape.max(0.0)
}

#[cfg(test)]
mod moshe_no_effective_layer_tests {
    use super::DerivedParams;
    use crate::{Profile, SoundingData};

    /// A dry high-plateau column (surface 780 hPa, dewpoint depression 25 C at
    /// every level, so every parcel has CAPE 0): no effective inflow layer,
    /// light veering wind, weak ascent.
    fn dry_plateau_column() -> SoundingData {
        let pres: Vec<f64> = (0..=34).map(|i| 780.0 - 20.0 * i as f64).collect();
        let (z0, t0) = (2240.0_f64, 22.0_f64);
        let mut hght = vec![z0];
        let mut tmpc = vec![t0];
        for i in 1..pres.len() {
            let dz = 287.04 * (tmpc[i - 1] + 273.15) / 9.80665 * (pres[i - 1] / pres[i]).ln();
            let z = hght[i - 1] + dz;
            hght.push(z);
            tmpc.push(t0 - 6.5 * (z.min(16_000.0) - z0) / 1000.0);
        }
        let n = pres.len();
        SoundingData {
            dwpc: tmpc.iter().map(|t| t - 25.0).collect(),
            wdir: (0..n).map(|i| 90.0 + 4.0 * i as f64).collect(),
            wspd: (0..n).map(|i| 3.0 + 0.8 * i as f64).collect(),
            omeg: Some((0..n).map(|i| -0.05 * (i % 7) as f64).collect()),
            pres,
            hght,
            tmpc,
            latitude: Some(19.43),
            longitude: Some(-99.13),
            missing: None,
        }
    }

    #[test]
    fn moshe_is_unavailable_without_an_effective_layer() {
        let profile = Profile::new(dry_plateau_column()).unwrap();
        assert!(!profile.ebottom.is_finite() || !profile.etop.is_finite());
        let d = DerivedParams::compute(&profile);
        // Displayed MOSHE: `--`, not the -9999 kt sentinel run through the formula.
        assert!(d.modified_sherbe.is_nan(), "MOSHE {}", d.modified_sherbe);
        // sharpmod's number stays available for parity proofs, and is the junk
        // magnitude the display used to show.
        assert!(
            d.modified_sherbe_sharpmod.is_finite() && d.modified_sherbe_sharpmod.abs() > 50.0,
            "sharpmod MOSHE {}",
            d.modified_sherbe_sharpmod
        );
    }
}

#[cfg(test)]
mod ecape_ncape_tests {
    use super::{ec_ncape, ec_ncape_with_saturation, ec_r_sat};

    const HEIGHT_M: [f64; 5] = [0.0, 1_000.0, 2_000.0, 3_000.0, 49_420.0];
    const PRESSURE_PA: [f64; 5] = [100_000.0, 90_000.0, 80_000.0, 70_000.0, 100.0];
    const TEMPERATURE_K: [f64; 5] = [300.0, 294.0, 288.0, 282.0, 278.35];
    const SPECIFIC_HUMIDITY: [f64; 5] = [0.014, 0.010, 0.007, 0.004, 0.000_001];

    #[test]
    fn ncape_never_evaluates_saturation_above_equilibrium_level() {
        let mut evaluated_pressures = Vec::new();
        let value = ec_ncape_with_saturation(
            &HEIGHT_M,
            &PRESSURE_PA,
            &TEMPERATURE_K,
            &SPECIFIC_HUMIDITY,
            1_000.0,
            3_000.0,
            |temperature, pressure| {
                assert!(
                    pressure > 100.0,
                    "upper-stratospheric saturation must not be evaluated"
                );
                evaluated_pressures.push(pressure);
                ec_r_sat(temperature, pressure)
            },
        );

        assert!(value.is_finite());
        assert_eq!(evaluated_pressures, PRESSURE_PA[..=3]);
    }

    #[test]
    fn upper_stratospheric_level_does_not_change_ncape() {
        let baseline = ec_ncape(
            &HEIGHT_M[..4],
            &PRESSURE_PA[..4],
            &TEMPERATURE_K[..4],
            &SPECIFIC_HUMIDITY[..4],
            1_000.0,
            3_000.0,
        );
        let extended = ec_ncape(
            &HEIGHT_M,
            &PRESSURE_PA,
            &TEMPERATURE_K,
            &SPECIFIC_HUMIDITY,
            1_000.0,
            3_000.0,
        );

        assert!(baseline.is_finite());
        assert_eq!(extended, baseline);
    }
}

/// Entrainment parameter psi (ecape-rs `calc_psi`, sigma = 1.1).
fn ec_psi(el_z: f64) -> f64 {
    let (sigma, alpha, l_mix, pr, ksq) = (1.1_f64, 0.8_f64, 120.0_f64, 1.0 / 3.0, 0.18_f64);
    (ksq * alpha * alpha * std::f64::consts::PI.powi(2) * l_mix)
        / (4.0 * pr * sigma * sigma * el_z.max(1.0))
}

/// Analytic entraining CAPE (ecape-rs `calc_ecape_a`).
fn ec_ecape_a(vsr: f64, psi: f64, ncape: f64, cape: f64) -> f64 {
    let sr2 = (vsr * vsr).max(1e-9);
    let denom = 4.0 * psi / sr2;
    let term_a = sr2 / 2.0;
    let term_b = (-1.0 - psi - (2.0 * psi / sr2) * ncape) / denom;
    let term_c = ((1.0 + psi + (2.0 * psi / sr2) * ncape).powi(2)
        + 8.0 * (psi / sr2) * (cape - psi * ncape))
        .sqrt()
        / denom;
    (term_a + term_b + term_c).max(0.0)
}

/// Analytic MU ECAPE (J/kg). CAPE / LFC / EL come from the sharprs MU parcel
/// (the reference solver re-lifts its own MetPy-parity parcel; the resulting
/// engine difference is ~2.5% for the golden sounding, within tolerance).
fn ecape(inner: &sharprs::Profile, mupcl: &ParcelResult) -> f64 {
    if !qc(mupcl.bplus) {
        return f64::NAN;
    }
    if mupcl.bplus <= 0.0 {
        return 0.0;
    }
    // Levels with every required column valid, mirroring sharpmod's filter.
    let mut hgts = Vec::new();
    let mut pres_hpa = Vec::new();
    let mut t_k = Vec::new();
    let mut qv = Vec::new();
    let mut u_ms = Vec::new();
    let mut v_ms = Vec::new();
    for i in 0..inner.num_levels() {
        let (p, h, t, td, u, v) = (
            inner.pres[i],
            inner.hght[i],
            inner.tmpc[i],
            inner.dwpc[i],
            inner.u[i],
            inner.v[i],
        );
        if qc(p) && qc(h) && qc(t) && qc(td) && qc(u) && qc(v) {
            pres_hpa.push(p);
            hgts.push(h);
            t_k.push(t + 273.15);
            qv.push(ec_spec_hum(p * 100.0, td));
            u_ms.push(u * EC_KTS_TO_MS);
            v_ms.push(v * EC_KTS_TO_MS);
        }
    }
    if hgts.len() < 3 {
        return f64::NAN;
    }
    let hagl: Vec<f64> = hgts.iter().map(|z| z - hgts[0]).collect();
    let (su, sv) = ec_bunkers_rm(&pres_hpa, &hagl, &u_ms, &v_ms);
    let vsr = ec_sr_wind(&hgts, &u_ms, &v_ms, su, sv);
    if !vsr.is_finite() || vsr <= 0.0 {
        return 0.0;
    }
    let lfc_msl = mupcl.lfchght + inner.sfc_height();
    let el_msl = mupcl.elhght + inner.sfc_height();
    if !lfc_msl.is_finite() || !el_msl.is_finite() {
        return f64::NAN;
    }
    let p_pa: Vec<f64> = pres_hpa.iter().map(|p| p * 100.0).collect();
    let ncape = ec_ncape(&hgts, &p_pa, &t_k, &qv, lfc_msl, el_msl);
    let psi = ec_psi(el_msl);
    let value = ec_ecape_a(vsr, psi, ncape, mupcl.bplus);
    if !value.is_finite() {
        return f64::NAN;
    }
    value.clamp(0.0, mupcl.bplus)
}

// ═══════════════════════════════════════════════════════════════════════════
// Inferred temperature advection (port of `params.inferred_temp_adv`)
// ═══════════════════════════════════════════════════════════════════════════

fn inferred_temp_adv(inner: &sharprs::Profile) -> (Vec<f64>, Vec<(f64, f64)>) {
    if inner.u.iter().all(|u| !u.is_finite()) {
        return (Vec::new(), Vec::new());
    }
    let sfc_pres = inner.sfc_pressure();
    // Deepest reported level still at/above 100 hPa (arrays run sfc -> top).
    let stop = match inner
        .pres
        .iter()
        .filter(|p| p.is_finite() && **p >= 100.0)
        .last()
    {
        Some(p) => *p,
        None => return (Vec::new(), Vec::new()),
    };
    if !qc(sfc_pres) || sfc_pres <= 100.0 {
        return (Vec::new(), Vec::new());
    }
    // np.arange(sfc, stop, -100): values strictly above `stop`.
    let mut pressures = Vec::new();
    let mut p = sfc_pres;
    while p > stop {
        pressures.push(p);
        p -= 100.0;
    }
    if pressures.len() < 2 {
        return (Vec::new(), Vec::new());
    }
    let lat = inner.station.latitude;
    let omega = 2.0 * std::f64::consts::PI / 86164.0;
    let f = 2.0 * omega * lat.to_radians().sin();
    let multiplier = (f / 9.81) * (std::f64::consts::PI / 180.0);

    let temps: Vec<f64> = pressures
        .iter()
        .map(|&p| inner.interp_tmpc(p) + 273.15)
        .collect();
    let heights: Vec<f64> = pressures.iter().map(|&p| inner.interp_hght(p)).collect();
    let dirs: Vec<f64> = pressures.iter().map(|&p| inner.interp_vec(p).0).collect();

    let mut temp_adv = Vec::with_capacity(pressures.len() - 1);
    let mut bounds = Vec::with_capacity(pressures.len() - 1);
    for i in 1..pressures.len() {
        let bottom_pres = pressures[i - 1];
        let top_pres = pressures[i];
        bounds.push((bottom_pres, top_pres));
        let avg_temp = (temps[i] + temps[i - 1]) * 2.0;
        let (mean_u, mean_v) = mean_wind(inner, bottom_pres, top_pres, 0.0, 0.0);
        let mean_wspd = kts2ms(mag(mean_u, mean_v));
        let mut top_wdir = dirs[i] + (180.0 - dirs[i - 1]);
        if top_wdir < 0.0 {
            top_wdir += 360.0;
        } else if top_wdir >= 360.0 {
            top_wdir -= 360.0;
        }
        let d_theta = top_wdir - 180.0;
        let t_adv =
            multiplier * mean_wspd.powi(2) * avg_temp * (d_theta / (heights[i] - heights[i - 1]));
        temp_adv.push(t_adv * 3600.0);
    }
    (temp_adv, bounds)
}

// ═══════════════════════════════════════════════════════════════════════════
// Storm slinky trajectory (port of `params.parcelTraj`)
// ═══════════════════════════════════════════════════════════════════════════

fn parcel_traj(
    inner: &sharprs::Profile,
    pcl: &ParcelResult,
    smu: f64,
    smv: f64,
) -> (Vec<(f64, f64)>, f64) {
    if !qc(pcl.bplus) || pcl.bplus < 1e-3 || !qc(smu) || !qc(smv) {
        return (Vec::new(), f64::NAN);
    }
    let mut elhght = pcl.elhght;
    if !qc(elhght) {
        elhght = *inner.hght.last().unwrap_or(&f64::NAN);
    }
    let mut z0 = pcl.lfchght;
    let mut p0 = pcl.lfcpres;
    if !qc(z0) || !qc(p0) || !qc(elhght) {
        return (Vec::new(), f64::NAN);
    }
    // Parcel trace ordered by ascending log10 pressure for interpolation.
    let mut trace: Vec<(f64, f64)> = pcl
        .ptrace
        .iter()
        .zip(pcl.ttrace.iter())
        .filter(|(p, t)| p.is_finite() && t.is_finite())
        .map(|(p, t)| (p.log10(), *t))
        .collect();
    trace.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    if trace.len() < 2 {
        return (Vec::new(), f64::NAN);
    }
    // np.interp-style: clamped at the ends.
    let trace_temp = |logp: f64| -> f64 {
        if logp <= trace[0].0 {
            return trace[0].1;
        }
        if logp >= trace[trace.len() - 1].0 {
            return trace[trace.len() - 1].1;
        }
        for w in trace.windows(2) {
            if logp >= w[0].0 && logp <= w[1].0 {
                let f = (logp - w[0].0) / (w[1].0 - w[0].0);
                return w[0].1 + f * (w[1].1 - w[0].1);
            }
        }
        f64::NAN
    };

    let dt = 25.0;
    let (mut x0, mut y0, mut w0) = (0.0_f64, 0.0_f64, 5.0_f64);
    let mut pos = vec![(0.0, 0.0)];
    let mut z_last = z0;
    let mut steps = 0;
    while z0 < elhght {
        let env_tv = inner.interp_by_pressure(&inner.vtmp, p0) + 273.15;
        let pcl_tv = trace_temp(p0.log10()) + 273.15;
        if !env_tv.is_finite() || !pcl_tv.is_finite() {
            break;
        }
        let accel = 9.8 * (pcl_tv - env_tv) / env_tv;
        let z1 = 0.5 * accel * dt * dt + w0 * dt + z0;
        let w1 = accel * dt + w0;
        let (u, v) = inner.interp_wind(p0);
        let u0 = kts2ms(u - smu);
        let v0 = kts2ms(v - smv);
        let x1 = u0 * dt + x0;
        let y1 = v0 * dt + y0;
        pos.push((x1, y1));
        z_last = z1;
        z0 = z1;
        x0 = x1;
        y0 = y1;
        p0 = inner.pres_at_height(inner.to_msl(z1));
        w0 = w1;
        steps += 1;
        if !p0.is_finite() || steps > 5000 {
            break;
        }
    }
    let r = mag(x0, y0);
    let theta = z_last.atan2(r).to_degrees();
    (pos, theta)
}
