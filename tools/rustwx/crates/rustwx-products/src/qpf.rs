use rustwx_render::{
    Color, DiscreteColorScale, ExtendMode, LegendMode, Rgba, WeatherPalette, palette_scale,
};

/// The precipitation scale, in inches: one listed colour per band.
///
/// THE BREAKAGE THIS PREVENTS (3.0.3 candidate check, 2026-11-15): the scale
/// handed the 0.01-15 in levels the whole precipitation palette, which is
/// laid out linearly in value (100 entries an inch), and drew a linear colour
/// bar with whole-inch ticks.  Every amount under 0.1 in was a pale grey and
/// every band under 1 in sat in the bottom fifteenth of the bar, so the 1 h and 3 h
/// rain of an ordinary forecast could not be read off its map.  The bands
/// keep the palette's families at the same amounts (green to 0.5 in, blue to
/// 1, purple to 2, red to 4, yellow and brown above), with green from the
/// first 0.01 in and no near-white band, and are drawn with
/// [`qpf_legend_mode`]: one colour per band and every band the same length
/// on the bar.
pub(crate) fn qpf_inches_scale() -> DiscreteColorScale {
    DiscreteColorScale {
        levels: qpf_inches_levels(),
        colors: qpf_band_colors(),
        extend: ExtendMode::Max,
        mask_below: Some(0.01),
    }
}

/// How every precipitation map draws [`qpf_inches_scale`]: one listed colour
/// per band, each band the same length on the colour bar with its edges
/// labelled, as WPC and Pivotal draw rain.  Under the default stepped legend
/// the listed colours would be resampled linearly in value, which is the
/// defect [`qpf_inches_scale`] names.
pub(crate) const fn qpf_legend_mode() -> LegendMode {
    LegendMode::Thresholds
}

pub(crate) fn qpf_inches_levels() -> Vec<f64> {
    vec![
        0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20, 1.40, 1.60,
        1.80, 2.00, 2.50, 3.00, 3.50, 4.00, 5.00, 6.00, 8.00, 10.00, 15.00,
    ]
}

/// One colour per band of [`qpf_inches_levels`], family by family.
fn qpf_band_colors() -> Vec<Color> {
    // (anchors, bands): the bands each family covers, in level order.
    const FAMILIES: &[(&[&str], usize)] = &[
        // 0.01 to 0.50 in
        (&["#b8f0c1", "#156471"], 6),
        // 0.50 to 1.00 in
        (&["#164fba", "#a9d3ec"], 5),
        // 1.00 to 2.00 in
        (&["#cfbddd", "#a134b1"], 5),
        // 2.00 to 4.00 in
        (&["#a43c32", "#dd9c98"], 4),
        // 4.00 to 15.00 in
        (&["#f6f0a3", "#7e4b26", "#542f17"], 5),
    ];
    FAMILIES
        .iter()
        .flat_map(|(anchors, bands)| lerp(anchors, *bands))
        .map(Color::from)
        .collect()
}

fn lerp(anchors: &[&str], n: usize) -> Vec<Rgba> {
    let anchors: Vec<Rgba> = anchors.iter().map(|hex| Rgba::from_hex(hex)).collect();
    (0..n)
        .map(|i| {
            let t = if n <= 1 {
                0.0
            } else {
                i as f64 / (n - 1) as f64 * (anchors.len() - 1) as f64
            };
            let lo = (t as usize).min(anchors.len() - 1);
            let hi = (lo + 1).min(anchors.len() - 1);
            let f = t - lo as f64;
            let mix = |a: u8, b: u8| (a as f64 + f * (b as f64 - a as f64)).round() as u8;
            let (a, b) = (anchors[lo], anchors[hi]);
            Rgba::new(mix(a.r, b.r), mix(a.g, b.g), mix(a.b, b.b))
        })
        .collect()
}

pub(crate) fn snowfall_inches_scale() -> DiscreteColorScale {
    palette_scale(WeatherPalette::Precip,
        vec![0.1,0.5,1.,2.,3.,4.,6.,8.,10.,12.,18.,24.,30.,36.,48.,60.],
        ExtendMode::Max, Some(0.1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustwx_render::{ColorScale, ColormapBuildOptions, LegendControls, build_colormap};

    #[test]
    fn qpf_inches_scale_keeps_standard_precip_family_breakpoints() {
        let scale = qpf_inches_scale();
        assert!(scale.levels.contains(&0.1));
        assert!(scale.levels.contains(&0.5));
        assert!(scale.levels.contains(&1.0));
        assert_eq!(scale.levels.last(), Some(&15.0));
        assert_eq!(scale.extend, ExtendMode::Max);
        assert_eq!(scale.mask_below, Some(0.01));
    }

    fn drawn(value: f64) -> Rgba {
        let cmap = build_colormap(
            &ColorScale::Discrete(qpf_inches_scale()),
            ColormapBuildOptions {
                legend: LegendControls { mode: qpf_legend_mode(), ..LegendControls::default() },
                ..ColormapBuildOptions::default()
            },
        );
        cmap.map(value)
    }

    /// HSV saturation and value of a drawn colour.
    fn saturation_value(colour: Rgba) -> (f64, f64) {
        let high = colour.r.max(colour.g).max(colour.b) as f64;
        let low = colour.r.min(colour.g).min(colour.b) as f64;
        (if high == 0.0 { 0.0 } else { (high - low) / high }, high / 255.0)
    }

    // THE BREAKAGE THIS PREVENTS (3.0.3 candidate check, 2026-11-15): light
    // rain drawn in a pale grey that read as no rain.
    #[test]
    fn light_rain_is_drawn_in_colour_not_grey() {
        for value in [0.01, 0.03, 0.07, 0.15, 0.25] {
            let (saturation, _) = saturation_value(drawn(value));
            assert!(saturation > 0.15, "{value} in drew {:?}", drawn(value));
        }
        assert_eq!(drawn(0.005).a, 0, "below 0.01 in stays clear");
    }

    #[test]
    fn every_band_has_its_own_colour_and_none_is_near_white() {
        let scale = qpf_inches_scale();
        assert_eq!(scale.colors.len(), scale.levels.len() - 1);
        let drawn_bands: Vec<Rgba> = scale
            .levels
            .windows(2)
            .map(|band| drawn(0.5 * (band[0] + band[1])))
            .collect();
        for (index, pair) in drawn_bands.windows(2).enumerate() {
            assert_ne!(pair[0], pair[1], "bands {index} and {} share a colour", index + 1);
        }
        for colour in &drawn_bands {
            let (saturation, value) = saturation_value(*colour);
            assert!(!(saturation < 0.12 && value > 0.85), "near-white band {colour:?}");
        }
    }

    #[test]
    fn the_families_sit_at_the_same_amounts() {
        let hue_family = |value: f64| {
            let c = drawn(value);
            (c.r as i32, c.g as i32, c.b as i32)
        };
        let (r, g, b) = hue_family(0.3);
        assert!(g > r && g > b - 40, "0.3 in is green: {:?}", (r, g, b));
        let (r, g, b) = hue_family(0.75);
        assert!(b > r && b > g, "0.75 in is blue: {:?}", (r, g, b));
        let (r, g, b) = hue_family(1.5);
        assert!(r > g && b > g, "1.5 in is purple: {:?}", (r, g, b));
        let (r, g, b) = hue_family(2.25);
        assert!(r > g && r > b, "2.25 in is red: {:?}", (r, g, b));
    }

    #[test]
    fn the_colour_bar_gives_every_band_the_same_length() {
        let cmap = build_colormap(
            &ColorScale::Discrete(qpf_inches_scale()),
            ColormapBuildOptions {
                legend: LegendControls { mode: qpf_legend_mode(), ..LegendControls::default() },
                ..ColormapBuildOptions::default()
            },
        );
        assert!(cmap.equal_bands);
    }
}
