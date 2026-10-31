//! Shared Rust radar model bridge and native volume writers.

use bowecho_simradar::radar_core::{MomentStorage, MomentType, RadarVolume};
use recast_radar_core::model::{
    Field, FieldData, FieldName, FloatCoding, IntCoding, LinearTransform, Location,
    SimulationProvenance, Sweep, SweepMode, Volume,
};
use serde_json::{Value, json};

/// The shared radar model, for main.rs's tests: the model crate's name is
/// written in this file and the other pinned radar-writer files only.
#[cfg(test)]
pub(crate) use recast_radar_core::model as radar_model;

/// Convert simulated polar gates without regridding or changing their values.
pub fn to_fm301(source: &RadarVolume) -> Result<Volume, String> {
    let mut result = Volume::new(&source.site.id, source.volume_time);
    result.attrs.title = Some("WOOF simulated radar volume".into());
    result.attrs.source = Some("WOOF simulated atmosphere".into());
    result.attrs.simulated = true;
    result.attrs.site_name.clone_from(&source.site.name);
    result.location = Location {
        latitude_deg: source.site.latitude_deg.map(f64::from),
        longitude_deg: source.site.longitude_deg.map(f64::from),
        altitude_m: source.site.elevation_m.map(f64::from),
        altitude_agl_m: None,
    };
    result.scan.vcp_pattern = source.vcp.as_ref().map(|vcp| vcp.pattern);
    result.scan.id = result.scan.vcp_pattern.map(i64::from);
    result.scan.name.clone_from(&source.metadata.scan_name);
    result.radar_parameters.beam_width_h_deg = source.metadata.beam_width_h_deg;
    result.radar_parameters.beam_width_v_deg = source.metadata.beam_width_v_deg;
    result.radar_parameters.pulse_width_s = source.metadata.pulse_width_us.map(|v| v * 1e-6);
    result.radar_parameters.prt_s = source.metadata.prt_s;
    result.radar_parameters.unambiguous_range_m =
        source.metadata.unambiguous_range_km.map(|v| v * 1000.0);
    if let Some(mhz) = source.metadata.radar_frequency_mhz {
        result
            .radar_parameters
            .frequency_hz
            .push(f64::from(mhz) * 1e6);
    }
    result.simulation = Some(Box::new(SimulationProvenance {
        forward_operator: source.metadata.forward_operator.clone(),
        forward_operator_config: source.metadata.forward_operator_config.clone(),
        source_model: Some("WOOF".into()),
        microphysics_scheme: source.metadata.microphysics_scheme.clone(),
        scattering_model: source.metadata.scattering_model.clone(),
    }));
    let fraction_s = f64::from(source.volume_time.timestamp_subsec_nanos()) * 1e-9;
    for (number, cut) in source.cuts.iter().enumerate() {
        let mut sweep = Sweep::new(
            number as u32,
            SweepMode::AzimuthSurveillance,
            cut.elevation_deg,
        );
        sweep.elevation_number = cut.elevation_number.map(u16::from);
        for radial in &cut.radials {
            sweep.push_ray(
                fraction_s + f64::from(radial.time_offset_ms) * 0.001,
                radial.azimuth_deg,
                radial.elevation_deg,
            );
        }
        if cut
            .radials
            .iter()
            .any(|ray| ray.nyquist_velocity_mps.is_some())
        {
            sweep.ray_vars.nyquist_velocity_mps = Some(
                cut.radials
                    .iter()
                    .map(|ray| ray.nyquist_velocity_mps.unwrap_or(f32::NAN))
                    .collect(),
            );
        }
        if let Some(instrument) = cut
            .aligned_ray_instrument_metadata()
            .map_err(|e| e.to_string())?
        {
            if instrument.iter().any(|ray| ray.prt_s.is_some()) {
                sweep.ray_vars.prt_s = Some(
                    instrument
                        .iter()
                        .map(|ray| ray.prt_s.unwrap_or(f32::NAN))
                        .collect(),
                );
            }
            if instrument
                .iter()
                .any(|ray| ray.unambiguous_range_km.is_some())
            {
                sweep.ray_vars.unambiguous_range_m = Some(
                    instrument
                        .iter()
                        .map(|ray| {
                            ray.unambiguous_range_km
                                .map(|v| v * 1000.0)
                                .unwrap_or(f32::NAN)
                        })
                        .collect(),
                );
            }
        }
        for (moment, grid) in &cut.moments {
            let name = match moment {
                MomentType::Reflectivity => "DBZH",
                MomentType::Velocity => "VRADH",
                MomentType::SpectrumWidth => "WRADH",
                MomentType::DifferentialReflectivity => "ZDR",
                MomentType::CorrelationCoefficient => "RHOHV",
                MomentType::DifferentialPhase => "PHIDP",
                MomentType::SpecificDifferentialPhase => "KDP",
                MomentType::Unknown(name) => name.as_str(),
            };
            let ngates = u32::try_from(grid.gate_range.gate_count)
                .map_err(|_| "radar field has too many gates")?;
            let mapping = sweep
                .attach_geometry(
                    f64::from(grid.gate_range.first_gate_m),
                    f64::from(grid.gate_range.gate_spacing_m),
                    ngates,
                )
                .map_err(|e| e.to_string())?;
            let mut source_rows = vec![None; cut.radials.len()];
            for (row, &ray) in grid.radial_indices.iter().enumerate() {
                let slot = source_rows
                    .get_mut(ray)
                    .ok_or("radar moment row points outside its sweep")?;
                if slot.replace(row).is_some() {
                    return Err("radar moment repeats a ray row".into());
                }
            }
            let data = match &grid.storage {
                MomentStorage::F32(values) => FieldData::F32 {
                    values: align_rows(values, ngates as usize, &source_rows, f32::NAN)?,
                    coding: FloatCoding::default(),
                },
                MomentStorage::U8(values) => {
                    let mut coding = IntCoding::new(LinearTransform::IcdScaleOffset {
                        scale: grid.scale,
                        offset: grid.offset,
                    });
                    coding.fill_value = grid
                        .nodata
                        .map(|v| u8::try_from(v).map_err(|_| "8-bit fill code does not fit"))
                        .transpose()?;
                    coding.range_folded = grid
                        .range_folded
                        .map(|v| u8::try_from(v).map_err(|_| "8-bit folded code does not fit"))
                        .transpose()?;
                    FieldData::U8 {
                        values: align_rows(
                            values,
                            ngates as usize,
                            &source_rows,
                            coding.fill_value.unwrap_or(0),
                        )?,
                        coding,
                    }
                }
                MomentStorage::U16(values) => {
                    let mut coding = IntCoding::new(LinearTransform::IcdScaleOffset {
                        scale: grid.scale,
                        offset: grid.offset,
                    });
                    coding.fill_value = grid.nodata;
                    coding.range_folded = grid.range_folded;
                    FieldData::U16 {
                        values: align_rows(
                            values,
                            ngates as usize,
                            &source_rows,
                            coding.fill_value.unwrap_or(0),
                        )?,
                        coding,
                    }
                }
            };
            let mut field = Field::new(FieldName::parse(name), mapping, ngates, data);
            field.absent_rows = source_rows
                .iter()
                .enumerate()
                .filter_map(|(index, row)| row.is_none().then_some(index as u32))
                .collect();
            sweep.add_field(field).map_err(|e| e.to_string())?;
        }
        result.sweeps.push(sweep);
    }
    result.seal().map_err(|e| e.to_string())?;
    Ok(result)
}

fn align_rows<T: Copy>(
    values: &[T],
    ngates: usize,
    source_rows: &[Option<usize>],
    fill: T,
) -> Result<Vec<T>, String> {
    let total = source_rows
        .len()
        .checked_mul(ngates)
        .ok_or("radar field dimensions overflow")?;
    let mut result = Vec::with_capacity(total);
    for row in source_rows {
        if let Some(row) = row {
            let start = row
                .checked_mul(ngates)
                .ok_or("radar field offset overflows")?;
            let end = start
                .checked_add(ngates)
                .ok_or("radar field offset overflows")?;
            result.extend_from_slice(
                values
                    .get(start..end)
                    .ok_or("radar field row is truncated")?,
            );
        } else {
            result.resize(result.len() + ngates, fill);
        }
    }
    Ok(result)
}

/// Encode a format and return its exact writer summary for the manifest.
pub fn encode(volume: &Volume, format: &str) -> Result<(Vec<u8>, Value), String> {
    match format {
        "level2" => {
            use recast_radar_io_nexrad::write::{WriteOptions, write_volume_to};
            let mut options = WriteOptions::default();
            options.icao = Some(volume.attrs.instrument_name.clone());
            options.keep_sweep_order = true;
            let mut bytes = Vec::new();
            let summary =
                write_volume_to(volume, &options, &mut bytes).map_err(|e| e.to_string())?;
            let report = json!({
                "site": summary.icao, "sweeps": summary.sweeps, "radials": summary.radials,
                "notes": summary.notes, "skipped_sweeps": summary.skipped_sweeps,
                "skipped_fields": summary.skipped_fields.iter().map(|item| json!({"sweep":item.sweep,"field":item.field.as_str(),"reason":item.reason})).collect::<Vec<_>>(),
                "moments": summary.moments.iter().map(|item| json!({"sweep":item.sweep,"field":item.field.as_str(),"word_size":item.word_size,"scale":item.scale,"offset":item.offset,"max_abs_error":item.max_abs_error,"exact":item.exact})).collect::<Vec<_>>(),
            });
            Ok((bytes, report))
        }
        "cfradial1" => {
            recast_radar_io_cfradial::write::write_cfradial1(volume, &Default::default())
                .map(|b| (b, json!({})))
                .map_err(|e| e.to_string())
        }
        "cfradial2" => {
            recast_radar_io_cfradial::write::write_cfradial2(volume, &Default::default())
                .map(|b| (b, json!({})))
                .map_err(|e| e.to_string())
        }
        "odim" => recast_radar_io_odim::write_odim_h5_volume(volume, &Default::default())
            .map(|b| (b, json!({})))
            .map_err(|e| e.to_string()),
        _ => Err(format!("unsupported radar output format: {format}")),
    }
}
