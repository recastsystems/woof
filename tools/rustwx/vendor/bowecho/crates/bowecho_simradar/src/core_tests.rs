    #[test]
    fn property_build_budget_reserves_every_owned_category_before_raw_read() {
        assert_eq!(
            checked_property_tmatrix_build_remainder(1_000, 100, 200, 300, 50, 300).unwrap(),
            350
        );
        assert!(checked_property_tmatrix_build_remainder(650, 100, 200, 300, 50, 1).is_err());
        assert!(checked_property_tmatrix_build_remainder(651, 100, 200, 300, 50, 2).is_err());
        assert!(
            checked_property_tmatrix_build_remainder(usize::MAX, usize::MAX, 1, 0, 0, 1,).is_err()
        );
    }

    #[test]
    fn property_scene_failures_propagate_while_bulk_failures_remain_skippable() {
        let research = SyntheticRadarConfig {
            polarimetric_kernel: PolarimetricKernel::PropertyTMatrixResearchV1,
            ..SyntheticRadarConfig::default()
        };
        let mut notes = Vec::new();
        assert_eq!(
            record_or_propagate_scene_failure(
                &research,
                &mut notes,
                "property scene failed".to_string(),
            )
            .unwrap_err(),
            "property scene failed"
        );
        assert!(notes.is_empty());

        let bulk = SyntheticRadarConfig::default();
        record_or_propagate_scene_failure(&bulk, &mut notes, "bulk scene failed".to_string())
            .unwrap();
        assert_eq!(notes, vec!["bulk scene failed".to_string()]);
    }

    #[test]
    fn standard_geometry_matches_the_historical_four_thirds_earth_path() {
        let gate_range = GateRange {
            first_gate_m: 0,
            gate_spacing_m: 250,
            gate_count: 10,
        };
        let slant_m = 1_250.0;
        let point = BeamPropagationGeometry::StandardFourThirdsEarth
            .point(0.5, &gate_range, 5, 0.0, slant_m)
            .unwrap();
        assert_eq!(
            point,
            BeamGeometryPoint {
                ground_range_m: beam_ground_range_m(slant_m, 0.5),
                height_above_radar_m: beam_height_above_radar_m(slant_m, 0.5),
            }
        );
    }

    #[test]
    fn wrf_refractivity_selection_fails_closed_without_a_qualified_profile() {
        let fields = uniform_box_fields();
        let config = SyntheticRadarConfig {
            propagation_geometry: PropagationGeometry::WrfRefractivityResearch,
            ..SyntheticRadarConfig::default()
        };
        let error = BeamPropagationGeometry::resolve(&fields, &config, 39.0, -95.0, 10.0)
            .err()
            .unwrap();
        assert!(error.contains("no qualified P/PB/T/QVAPOR profile"));
    }

    fn fingerprint_scene(
        path: &str,
        source_identity: &str,
        time_index: usize,
    ) -> app_ui::wrf_scene_inventory::WrfScene {
        use app_ui::wrf_scene_inventory::{
            WrfDomainId, WrfGridSignature, WrfProducerIdentity, WrfRunDomain, WrfRunId,
            WrfSceneTime, WrfSourceIdentity,
        };
        use chrono::TimeZone;

        app_ui::wrf_scene_inventory::WrfScene {
            path: PathBuf::from(path),
            time_index,
            run_domain: WrfRunDomain {
                run: WrfRunId("2026-07-12_00:00:00".to_owned()),
                domain: WrfDomainId(3),
            },
            grid_signature: WrfGridSignature::from_meters(
                400,
                300,
                Some(50),
                Some(3_000.0),
                Some(3_000.0),
                "lambert",
                0x1234,
            ),
            producer: WrfProducerIdentity::Wrf,
            source_identity: WrfSourceIdentity(source_identity.to_owned()),
            time: WrfSceneTime::InternalTimes {
                valid_time: Utc.with_ymd_and_hms(2026, 7, 12, 1, 0, 0).unwrap(),
                raw: "2026-07-12_01:00:00".to_owned(),
            },
        }
    }

    fn fingerprint_group(
        scene: app_ui::wrf_scene_inventory::WrfScene,
    ) -> app_ui::wrf_scene_inventory::WrfSceneGroup {
        app_ui::wrf_scene_inventory::WrfSceneInventory::from_scenes([scene])
            .groups
            .into_iter()
            .next()
            .unwrap()
    }

    #[test]
    fn temporal_preflight_counts_retained_frames_scratch_and_actual_scene_count() {
        let group = fingerprint_group(fingerprint_scene(
            "C:/private/a/wrfout_d03",
            "sha256:source-a",
            0,
        ));
        let config = SyntheticRadarConfig {
            elevations_deg: vec![0.5],
            azimuth_count: 12,
            gate_spacing_m: 1_000.0,
            max_range_m: 10_000.0,
            ..SyntheticRadarConfig::default()
        };
        let one = temporal_memory_estimate(&group, &config, 1).unwrap();
        let three = temporal_memory_estimate(&group, &config, 3).unwrap();
        assert_eq!(three.output_bytes, one.output_bytes * 3);
        assert!(one.shared_static_bytes > 12 * 10 * 13 * std::mem::size_of::<f32>());
        let single_peak = temporal_required_peak_bytes(one, false).unwrap();
        let two_scene_peak = temporal_required_peak_bytes(one, true).unwrap();
        assert_eq!(two_scene_peak - single_peak, one.scene_bytes().unwrap());
    }

    #[test]
    fn temporal_preflight_counts_nine_compact_polar_bytes_per_cell() {
        let group = fingerprint_group(fingerprint_scene(
            "C:/private/a/wrfout_d03",
            "sha256:source-a",
            0,
        ));
        let scalar_config = SyntheticRadarConfig::default();
        let polar_config = SyntheticRadarConfig {
            dual_pol: true,
            ..scalar_config.clone()
        };
        let scalar = temporal_memory_estimate(&group, &scalar_config, 1).unwrap();
        let polar = temporal_memory_estimate(&group, &polar_config, 1).unwrap();
        assert_eq!(
            polar.compact_bytes_per_scene - scalar.compact_bytes_per_scene,
            polar.cells_per_scene * 9
        );
    }

    #[test]
    fn temporal_budget_error_points_to_the_distinct_build_cap_and_reductions() {
        let error = temporal_build_budget_error(57 * 1024_usize.pow(3), 8 * 1024_usize.pow(3));
        assert!(error.contains("57.00 GiB"));
        assert!(error.contains("8.00 GiB"));
        assert!(error.contains("Temporal build RAM cap"));
        assert!(error.contains("select fewer WRF frames"));
        assert!(error.contains("turn off Ideal/Measured stage diagnostics"));
    }

    #[test]
    fn scene_build_fingerprint_tracks_input_identity_without_private_paths() {
        let config = SyntheticRadarConfig::default();
        let first = fingerprint_group(fingerprint_scene(
            "C:/private/a/wrfout_d03",
            "sha256:source-a",
            0,
        ));
        let moved = fingerprint_group(fingerprint_scene(
            "D:/moved/wrfout_d03",
            "sha256:source-a",
            0,
        ));
        let changed_source = fingerprint_group(fingerprint_scene(
            "C:/private/a/wrfout_d03",
            "sha256:source-b",
            0,
        ));
        let changed_time_index = fingerprint_group(fingerprint_scene(
            "C:/private/a/wrfout_d03",
            "sha256:source-a",
            1,
        ));

        let first_fingerprint = scene_build_fingerprint(&config, &first);
        assert_eq!(
            first_fingerprint,
            scene_build_fingerprint(&config, &moved),
            "absolute paths are deliberately outside refresh/cache identity"
        );
        assert_ne!(
            first_fingerprint,
            scene_build_fingerprint(&config, &changed_source)
        );
        assert_ne!(
            first_fingerprint,
            scene_build_fingerprint(&config, &changed_time_index)
        );
    }

    #[test]
    fn parses_wrf_times_with_colon_or_underscore() {
        let expected = "2026-05-19T00:00:00+00:00";
        assert_eq!(
            parse_wrf_time("2026-05-19_00:00:00").unwrap().to_rfc3339(),
            expected
        );
        assert_eq!(
            parse_wrf_time(" 2026-05-19_00:00:00 ")
                .unwrap()
                .to_rfc3339(),
            expected
        );
        assert!(parse_wrf_time("not-a-time").is_none());
    }

    #[test]
    fn radial_velocity_projection_signs_are_physical() {
        // Beam pointing due east (az=90°) at 0° elevation: a pure eastward wind
        // (u>0, v=0) blows AWAY from the radar → positive Vr; a westward wind →
        // negative. Verifies the (sinAz·cosEl, cosAz·cosEl, sinEl) projection.
        let az_rad: f32 = 90f32.to_radians();
        let (sin_az, cos_az) = (az_rad.sin(), az_rad.cos());
        let (u, v, w) = (12.0f32, 0.0, 0.0);
        let vr = u * sin_az * 1.0 + v * cos_az * 1.0 + w * 0.0;
        assert!(
            (vr - 12.0).abs() < 1e-3,
            "east wind due-east beam Vr = {vr}"
        );

        // Straight-up beam (el=90°) sees only w.
        let el_rad: f32 = 90f32.to_radians();
        let vr_up = 0.0 * 0.0 + 0.0 * 0.0 + 3.5 * el_rad.sin();
        assert!((vr_up - 3.5).abs() < 1e-3, "vertical beam Vr = {vr_up}");
    }

    fn uniform_box_fields() -> WrfRadarFields {
        let nx = 2;
        let ny = 2;
        let nz = 2;
        let cells = nx * ny;
        // Grid centred near (39, -95) with ~0.2° spacing.
        let lat = vec![38.9f32, 38.9, 39.1, 39.1];
        let lon = vec![-95.1f32, -94.9, -95.1, -94.9];
        let height_msl = {
            let mut h = vec![0.0f32; nz * cells];
            for c in 0..cells {
                h[c] = 100.0; // level 0 ~100 m MSL
                h[cells + c] = 8000.0; // level 1 ~8 km MSL
            }
            h
        };
        let dbz = vec![40.0f32; nz * cells];
        let u = vec![10.0f32; nz * cells];
        let v = vec![0.0f32; nz * cells];
        let w = vec![0.0f32; nz * cells];
        let terrain_m = vec![0.0f32; cells];
        let lut = InverseLut::build_with_shape_domain_bounded(&lat, &lon, nx, ny).expect("lut");
        WrfRadarFields {
            nx,
            ny,
            nz,
            lat,
            lon,
            height_msl,
            dbz,
            u,
            v,
            w,
            terrain_m,
            property_scattering: None,
            property_table_identity: None,
            property_table_resident_bytes: 0,
            raw_property_scene: None,
            refractivity_model: None,
            polarimetric: None,
            dual_pol_status: None,
            tke_tenths_m2s2: None,
            ref_source: "test",
            dx_m: None,
            source_model_override: None,
            source_microphysics_override: None,
            source_scattering_override: None,
            source_provenance_fragment: None,
            lut,
        }
    }

    #[test]
    fn temporal_column_sampling_blends_linear_z_and_wind_before_gate_physics() {
        let mut anchor = uniform_box_fields();
        anchor.dbz.fill(0.0);
        anchor.u.fill(10.0);
        let mut neighbor = uniform_box_fields();
        neighbor.dbz.fill(20.0);
        neighbor.u.fill(30.0);

        let midpoint = sample_column_temporal(
            &anchor,
            Some(&neighbor),
            0.5,
            anchor.cells(),
            39.0,
            -95.0,
            1_000.0,
            0.5,
            AtmosphereTimeMode::LinearAdjacent,
            ReflectivitySampling::LinearZ,
            None,
            None,
        )
        .expect("midpoint query")
        .expect("midpoint model column");
        assert!((z_to_dbz(midpoint.z_linear) - 17.032913).abs() < 1.0e-5);
        assert!((midpoint.u - 20.0).abs() < 1.0e-6);

        let exact_anchor = sample_column_temporal(
            &anchor,
            None,
            0.0,
            anchor.cells(),
            39.0,
            -95.0,
            1_000.0,
            0.5,
            AtmosphereTimeMode::LinearAdjacent,
            ReflectivitySampling::LinearZ,
            None,
            None,
        )
        .expect("anchor query")
        .expect("exact anchor column");
        assert_eq!(z_to_dbz(exact_anchor.z_linear), 0.0);
        assert_eq!(exact_anchor.u, 10.0);
    }

    fn attach_uniform_polar(
        fields: &mut WrfRadarFields,
        sample: crate::wrf_radar_physics::IntrinsicPolarSample,
    ) {
        let profile = crate::wrf_radar_physics::detect_scheme(
            Some(10),
            ["QRAIN", "QNRAIN", "QSNOW", "QNSNOW"],
        );
        let mut compact = CompactPolarFields::new(
            fields.dbz.len(),
            profile,
            vec!["QRAIN".to_string(), "QNRAIN".to_string()],
        );
        for index in 0..fields.dbz.len() {
            compact.store(index, sample);
        }
        fields.polarimetric = Some(compact);
    }

    #[test]
    fn compact_polar_fields_preserve_signed_kdp() {
        let profile = crate::wrf_radar_physics::detect_scheme(Some(10), ["QRAIN", "QNRAIN"]);
        let mut compact = CompactPolarFields::new(2, profile, vec!["QRAIN".to_string()]);
        let base = crate::wrf_radar_physics::IntrinsicPolarSample {
            zh: 10.0,
            zv: 10.0,
            covariance_magnitude: 10.0,
            rho_hv: 1.0,
            ..crate::wrf_radar_physics::IntrinsicPolarSample::default()
        };
        compact.store(
            0,
            crate::wrf_radar_physics::IntrinsicPolarSample {
                kdp_deg_km: -12.3,
                ..base
            },
        );
        compact.store(
            1,
            crate::wrf_radar_physics::IntrinsicPolarSample {
                kdp_deg_km: 12.3,
                ..base
            },
        );

        assert_eq!(compact.kdp, [-123, 123]);
        assert!((compact.contribution_at(0, 10.0, 0.5).kdp_deg_km + 12.3).abs() < 1.0e-6);
        assert!((compact.contribution_at(1, 10.0, 0.5).kdp_deg_km - 12.3).abs() < 1.0e-6);
    }

    #[test]
    fn compact_polar_precision_audit_counts_zeroing_and_saturation_without_changing_codes() {
        let profile = crate::wrf_radar_physics::detect_scheme(Some(10), ["QRAIN", "QNRAIN"]);
        let mut compact = CompactPolarFields::new(1, profile, vec!["QRAIN".to_string()]);
        let zh = 100.0;
        let zv = 10.0;
        let covariance = 20.0;
        let phase = 45.0f32.to_radians();
        compact.store(
            0,
            crate::wrf_radar_physics::IntrinsicPolarSample {
                zh,
                zv,
                cov_re: covariance * phase.cos(),
                cov_im: covariance * phase.sin(),
                covariance_magnitude: covariance,
                kdp_deg_km: 0.04,
                ah_db_km: 0.5,
                av_db_km: 0.1,
                fall_speed_mps: 30.0,
                fall_speed_variance_m2s2: 20.0f32.powi(2),
                zdr_db: 10.0,
                rho_hv: 0.8,
            },
        );

        assert_eq!(compact.zdr[0], i8::MAX);
        assert_eq!(compact.covariance_phase[0], i8::MAX);
        assert_eq!(compact.kdp[0], 0);
        assert_eq!(compact.ah[0], u8::MAX);
        assert_eq!(compact.adp[0], i8::MAX);
        assert_eq!(compact.fall_speed[0], u8::MAX);
        assert_eq!(compact.fall_speed_std[0], u8::MAX);
        assert!(compact.precision_audit.total_clamps() >= 6);
        assert!(compact.precision_audit.kdp_deg_km.quantized_to_zero >= 1);
        assert!(compact.precision_audit.zdr_db.max_abs_reconstruction_error > 3.0);
        assert!(
            compact
                .precision_audit
                .ah_db_km
                .max_abs_reconstruction_error
                > 0.2
        );
    }

    #[test]
    fn linear_z_interpolation_preserves_received_power() {
        let mut fields = uniform_box_fields();
        // West half 0 dBZ, east half 60 dBZ at both vertical levels. The box
        // centre therefore has equal contributors at the two powers.
        for level in 0..fields.nz {
            let base = level * fields.cells();
            fields.dbz[base] = 0.0;
            fields.dbz[base + 1] = 60.0;
            fields.dbz[base + 2] = 0.0;
            fields.dbz[base + 3] = 60.0;
        }
        let legacy = sample_column(
            &fields,
            fields.cells(),
            39.0,
            -95.0,
            1_000.0,
            0.5,
            ReflectivitySampling::LegacyDbz,
        )
        .expect("legacy query")
        .expect("legacy sample");
        let linear = sample_column(
            &fields,
            fields.cells(),
            39.0,
            -95.0,
            1_000.0,
            0.5,
            ReflectivitySampling::LinearZ,
        )
        .expect("linear-Z query")
        .expect("linear-Z sample");
        assert!((z_to_dbz(legacy.z_linear) - 30.0).abs() < 0.05);
        assert!((z_to_dbz(linear.z_linear) - 56.9897).abs() < 0.05);
    }

    #[test]
    fn pulse_volume_sample_counts_match_the_execution_quadratures() {
        for mode in [
            BeamIntegration::Center,
            BeamIntegration::Balanced,
            BeamIntegration::Reference,
        ] {
            assert_eq!(
                mode.pulse_volume_sample_count(),
                quadrature_points(mode).len()
            );
        }
    }

    #[test]
    fn raw_tmatrix_gate_chunks_reuse_one_bounded_column_budget() {
        assert_eq!(raw_tmatrix_pipeline_gate_chunk(0), 1);
        assert_eq!(raw_tmatrix_pipeline_gate_chunk(CENTER_QUADRATURE.len()), 64);
        assert_eq!(
            raw_tmatrix_pipeline_gate_chunk(BALANCED_QUADRATURE.len()),
            24
        );
        assert_eq!(
            raw_tmatrix_pipeline_gate_chunk(REFERENCE_QUADRATURE.len()),
            8
        );
        assert_eq!(
            raw_tmatrix_pipeline_gate_chunk(RAW_TMATRIX_PIPELINE_TARGET_COLUMNS),
            1
        );
        assert_eq!(
            raw_tmatrix_pipeline_gate_chunk(RAW_TMATRIX_PIPELINE_TARGET_COLUMNS + 1),
            1
        );
        for point_count in [1, 9, 27, RAW_TMATRIX_PIPELINE_TARGET_COLUMNS] {
            let gates = raw_tmatrix_pipeline_gate_chunk(point_count);
            assert!((1..=RAW_TMATRIX_PIPELINE_MAX_GATE_CHUNK).contains(&gates));
            assert!(gates.saturating_mul(point_count) <= RAW_TMATRIX_PIPELINE_TARGET_COLUMNS);
        }
    }

    #[test]
    fn raw_tmatrix_batches_identify_exact_center_geometry() {
        for (mode, expected_index) in [
            (BeamIntegration::Center, 0),
            (BeamIntegration::Balanced, 0),
            (BeamIntegration::Reference, 13),
        ] {
            let config = SyntheticRadarConfig {
                beam_integration: mode,
                ..SyntheticRadarConfig::default()
            };
            let points = resolved_beam_sample_points(&config, None, 250.0);
            assert_eq!(exact_center_point_index(&points), Some(expected_index));
        }

        let geometry = BeamGeometryPoint {
            ground_range_m: 1_234.5,
            height_above_radar_m: 67.25,
        };
        let batch = RawGateColumnBatch {
            first_gate: 7,
            gate_count: 1,
            points: vec![ResolvedBeamSamplePoint {
                az_sigma: 0.0,
                el_sigma: 0.0,
                range_offset_m: 0.0,
                weight: 1.0,
            }],
            center_point_index: Some(0),
            locations: vec![BeamColumnPoint {
                azimuth_deg: 90.0,
                elevation_deg: 0.5,
                latitude: 39.0,
                longitude: -95.0,
                z_msl: 500.0,
                geometry,
            }],
            samples: vec![None],
        };
        assert_eq!(batch.center_geometry(7).unwrap(), Some(geometry));
    }

    #[test]
    fn quadrature_tiers_leave_a_uniform_scene_invariant() {
        let fields = uniform_box_fields();
        let mut config = SyntheticRadarConfig {
            ref_gate_texture: false,
            vel_gate_texture: false,
            spectrum_width: true,
            spectrum_width_floor_mps: 0.0,
            ..SyntheticRadarConfig::default()
        };
        let mut samples = Vec::new();
        for tier in [
            BeamIntegration::Center,
            BeamIntegration::Balanced,
            BeamIntegration::Reference,
        ] {
            config.beam_integration = tier;
            samples.push(
                sample_gate(
                    &fields,
                    None,
                    0.0,
                    fields.cells(),
                    39.0,
                    -95.0,
                    200.0,
                    90.0,
                    0.5,
                    2_000.0,
                    8,
                    250.0,
                    &config,
                    None,
                )
                .expect("sample uniform gate")
                .expect("uniform gate"),
            );
        }
        for sample in &samples {
            assert!((z_to_dbz(sample.z_linear) - 40.0).abs() < 0.02);
            assert!((sample.velocity_mps - 10.0).abs() < 0.02);
        }
        assert!(samples[0].spectrum_width_mps < 1.0e-4);
        assert!(samples[1].spectrum_width_mps >= 0.0);
        assert!(samples[2].spectrum_width_mps >= 0.0);
    }

    #[test]
    fn gate_quality_reports_full_and_missing_model_support() {
        let fields = uniform_box_fields();
        let config = SyntheticRadarConfig {
            beam_integration: BeamIntegration::Balanced,
            ref_gate_texture: false,
            ..SyntheticRadarConfig::default()
        };
        let full = sample_gate_with_quality(
            &fields,
            None,
            0.0,
            fields.cells(),
            39.0,
            -95.0,
            200.0,
            90.0,
            0.5,
            1_000.0,
            4,
            250.0,
            &config,
            None,
        )
        .unwrap();
        assert_eq!(full.quality.model_coverage_fraction, 1.0);
        assert_eq!(full.quality.terrain_unblocked_fraction, 1.0);
        assert_eq!(full.quality.meteorological_signal_fraction, 1.0);
        assert!(full.physical.is_some());

        let missing = sample_gate_with_quality(
            &fields,
            None,
            0.0,
            fields.cells(),
            39.0,
            -95.0,
            200.0,
            90.0,
            0.5,
            2_000_000.0,
            8_000,
            250.0,
            &config,
            None,
        )
        .unwrap();
        assert_eq!(missing.quality, GateQualityFractions::default());
        assert!(missing.physical.is_none());
    }

    #[test]
    fn synthetic_volume_emits_compact_quality_grids_and_coverage_mask_is_configurable() {
        let fields = uniform_box_fields();
        let base = SyntheticRadarConfig {
            site_lat_deg: Some(39.0),
            site_lon_deg: Some(-95.0),
            antenna_msl_m: Some(200.0),
            elevations_deg: vec![0.5],
            azimuth_count: 36,
            gate_spacing_m: 500.0,
            max_range_m: 20_000.0,
            beam_integration: BeamIntegration::Balanced,
            ref_floor_dbz: -20.0,
            ref_gate_texture: false,
            emit_quality_fields: true,
            minimum_model_coverage_fraction: 0.0,
            ..SyntheticRadarConfig::default()
        };
        let time = DateTime::<Utc>::from_timestamp(1_700_000_000, 0).unwrap();
        let permissive = build_synthetic_volume(&fields, time, &base);
        for quality in QualityMoment::ALL {
            let grid = &permissive.cuts[0].moments[&quality.moment_type()];
            assert!(matches!(&grid.storage, MomentStorage::U8(_)));
            assert_eq!(grid.scale, 255.0);
        }

        let strict = build_synthetic_volume(
            &fields,
            time,
            &SyntheticRadarConfig {
                minimum_model_coverage_fraction: 1.0,
                ..base
            },
        );
        let finite_count = |volume: &RadarVolume| {
            let MomentStorage::F32(values) =
                &volume.cuts[0].moments[&MomentType::Reflectivity].storage
            else {
                panic!("synthetic reflectivity is f32");
            };
            values.iter().filter(|value| value.is_finite()).count()
        };
        assert!(
            finite_count(&strict) < finite_count(&permissive),
            "a strict full-support threshold must mask partially covered edge gates"
        );
        assert_eq!(
            strict.cuts[0].moments[&QualityMoment::ModelCoverage.moment_type()].radial_count(),
            strict.cuts[0].radials.len()
        );
    }

    #[test]
    fn ray_plan_resolves_sampling_time_and_status_before_rendering() {
        let rays = plan_synthetic_rays(1, 2, 4, ScanTiming::TimedVolume, 10.0, 40_000);
        assert_eq!(
            rays.iter()
                .map(|ray| (ray.azimuth_deg, ray.time_offset_ms))
                .collect::<Vec<_>>(),
            vec![
                (0.0, 40_000),
                (90.0, 49_000),
                (180.0, 58_000),
                (270.0, 67_000),
            ]
        );
        assert_eq!(
            rays.first().map(|ray| ray.radial_status),
            Some(radar_core::RadialStatus::StartElevation)
        );
        assert_eq!(
            rays.last().map(|ray| ray.radial_status),
            Some(radar_core::RadialStatus::EndVolume)
        );

        let frozen = plan_synthetic_rays(0, 1, 3, ScanTiming::InstantaneousTruth, 18.0, 99);
        assert!(frozen.iter().all(|ray| ray.time_offset_ms == 0));
    }

    #[test]
    fn timed_volume_stamps_monotonic_nonzero_ray_times() {
        let fields = uniform_box_fields();
        let config = SyntheticRadarConfig {
            site_lat_deg: Some(39.0),
            site_lon_deg: Some(-95.0),
            antenna_msl_m: Some(200.0),
            elevations_deg: vec![0.5, 1.5],
            azimuth_count: 12,
            gate_spacing_m: 500.0,
            max_range_m: 5_000.0,
            scan_timing: ScanTiming::TimedVolume,
            rotation_rate_deg_s: 12.0,
            transition_delay_s: 2.0,
            ref_gate_texture: false,
            ..SyntheticRadarConfig::default()
        };
        let volume = build_synthetic_volume(
            &fields,
            DateTime::<Utc>::from_timestamp(1_700_000_000, 0).unwrap(),
            &config,
        );
        let offsets: Vec<i32> = volume
            .cuts
            .iter()
            .flat_map(|cut| cut.radials.iter().map(|radial| radial.time_offset_ms))
            .collect();
        assert_eq!(offsets[0], 0);
        assert!(offsets.windows(2).all(|pair| pair[1] > pair[0]));
        assert!(offsets.last().copied().unwrap_or(0) > 50_000);
        assert_eq!(config.planned_scan_duration_ms(), 59_500);
        assert_eq!(
            config.planned_scan_duration_ms(),
            i64::from(*offsets.last().expect("last timed ray")),
            "temporal planner duration must be the exact latest sampled ray"
        );
    }

    #[test]
    fn cut_completion_callback_exposes_first_tilt_before_volume_finishes() {
        let fields = uniform_box_fields();
        let config = SyntheticRadarConfig {
            site_lat_deg: Some(39.0),
            site_lon_deg: Some(-95.0),
            antenna_msl_m: Some(200.0),
            elevations_deg: vec![0.5, 1.5],
            azimuth_count: 4,
            gate_spacing_m: 1_000.0,
            max_range_m: 2_000.0,
            ref_gate_texture: false,
            ..SyntheticRadarConfig::default()
        };
        let updates = std::cell::RefCell::new(Vec::new());
        let volume = build_synthetic_volume_reporting_inner(
            &fields,
            DateTime::<Utc>::from_timestamp(1_700_000_000, 0).unwrap(),
            &config,
            &|_| {},
            None,
            Some(&|partial, completed, total| {
                updates.borrow_mut().push((
                    completed,
                    total,
                    partial.cuts.len(),
                    partial.metadata.decoded_radial_count,
                ));
            }),
            None,
        )
        .unwrap();

        assert_eq!(updates.into_inner(), vec![(1, 2, 1, 4), (2, 2, 2, 8)]);
        assert_eq!(volume.cuts.len(), 2);
        assert_eq!(volume.metadata.decoded_radial_count, 8);
    }

    #[test]
    fn dual_pol_path_emits_propagation_and_attenuation_moments() {
        let mut fields = uniform_box_fields();
        let zh = dbz_to_z(40.0);
        let zv = zh / 10.0f32.powf(0.1);
        let covariance = 0.97 * (zh * zv).sqrt();
        attach_uniform_polar(
            &mut fields,
            crate::wrf_radar_physics::IntrinsicPolarSample {
                zh,
                zv,
                cov_re: covariance,
                cov_im: 0.0,
                covariance_magnitude: covariance,
                kdp_deg_km: 1.0,
                ah_db_km: 0.01,
                av_db_km: 0.008,
                fall_speed_mps: 5.0,
                fall_speed_variance_m2s2: 1.0,
                zdr_db: 1.0,
                rho_hv: 0.97,
            },
        );
        let config = SyntheticRadarConfig {
            site_lat_deg: Some(39.0),
            site_lon_deg: Some(-95.0),
            antenna_msl_m: Some(200.0),
            elevations_deg: vec![0.5],
            azimuth_count: 4,
            gate_spacing_m: 1_000.0,
            max_range_m: 5_000.0,
            ref_floor_dbz: -20.0,
            ref_gate_texture: false,
            dual_pol: true,
            propagation: true,
            system_phidp_deg: 7.0,
            ..SyntheticRadarConfig::default()
        };
        let volume = build_synthetic_volume(
            &fields,
            DateTime::<Utc>::from_timestamp(1_700_000_000, 0).unwrap(),
            &config,
        );
        let cut = &volume.cuts[0];
        for moment in [
            MomentType::DifferentialReflectivity,
            MomentType::CorrelationCoefficient,
            MomentType::DifferentialPhase,
            MomentType::SpecificDifferentialPhase,
            MomentType::Unknown("AH".to_string()),
            MomentType::Unknown("PIA".to_string()),
            MomentType::Unknown("REFC".to_string()),
            MomentType::Unknown("ADP".to_string()),
            MomentType::Unknown("PIDA".to_string()),
            MomentType::Unknown("ZDRC".to_string()),
        ] {
            assert!(cut.moments.contains_key(&moment), "missing {moment}");
        }
        let phi = cut.moments[&MomentType::DifferentialPhase]
            .scaled_value(0, 2)
            .expect("PhiDP gate");
        let pia = cut.moments[&MomentType::Unknown("PIA".to_string())]
            .scaled_value(0, 2)
            .expect("PIA gate");
        let refc = cut.moments[&MomentType::Unknown("REFC".to_string())]
            .scaled_value(0, 2)
            .expect("REFC gate");
        let observed = cut.moments[&MomentType::Reflectivity]
            .scaled_value(0, 2)
            .expect("REF gate");
        assert!((phi - 11.0).abs() < 0.05, "PhiDP={phi}");
        assert!((pia - 0.04).abs() < 0.005, "PIA={pia}");
        assert!((refc - observed - pia).abs() < 0.005);
        let rho = cut.moments[&MomentType::CorrelationCoefficient]
            .scaled_value(0, 2)
            .expect("rho gate");
        assert!((0.0..=1.0).contains(&rho));
    }

    #[test]
    fn band_path_retains_strong_specific_and_differential_attenuation() {
        let mut fields=uniform_box_fields();
        let zh=dbz_to_z(40.);
        let zv=zh/10f32.powf(0.1);
        attach_uniform_polar(&mut fields,crate::wrf_radar_physics::IntrinsicPolarSample {
            zh,zv,cov_re:(zh*zv).sqrt(),rho_hv:1.,zdr_db:1.,..Default::default()
        });
        let moment=crate::wrf_radar_physics::BulkContribution {
            zh,zv,cov_re:(zh*zv).sqrt(),kdp_deg_km:1.,ah_db_km:3.,av_db_km:2.,..Default::default()
        };
        fields.polarimetric.as_mut().unwrap().band_moments=Some(vec![[moment;5];fields.dbz.len()]);
        let config=SyntheticRadarConfig {
            site_lat_deg:Some(39.),site_lon_deg:Some(-95.),antenna_msl_m:Some(200.),
            elevations_deg:vec![0.5],azimuth_count:4,gate_spacing_m:1000.,max_range_m:5000.,
            ref_floor_dbz:-30.,ref_gate_texture:false,dual_pol:true,propagation:true,
            ..Default::default()
        };
        let volume=try_build_synthetic_volume(&fields,DateTime::<Utc>::from_timestamp(1_700_000_000,0).unwrap(),&config).unwrap();
        let cut=&volume.cuts[0];
        let value=|moment|cut.moments[&moment].scaled_value(0,2).unwrap();
        assert!((value(MomentType::Unknown("PIA".into()))-12.).abs()<0.01);
        assert!((value(MomentType::Unknown("PIDA".into()))-4.).abs()<0.01);
        assert!((value(MomentType::Reflectivity)-28.).abs()<0.01);
        assert!((value(MomentType::DifferentialReflectivity)+3.).abs()<0.01);
        assert!((value(MomentType::DifferentialPhase)-4.).abs()<0.01);
    }

    #[test]
    fn terminal_fall_speed_projects_toward_an_upward_beam() {
        let mut fields = uniform_box_fields();
        let zh = dbz_to_z(40.0);
        attach_uniform_polar(
            &mut fields,
            crate::wrf_radar_physics::IntrinsicPolarSample {
                zh,
                zv: zh,
                cov_re: zh,
                cov_im: 0.0,
                covariance_magnitude: zh,
                kdp_deg_km: 0.0,
                ah_db_km: 0.0,
                av_db_km: 0.0,
                fall_speed_mps: 5.0,
                fall_speed_variance_m2s2: 0.0,
                zdr_db: 0.0,
                rho_hv: 1.0,
            },
        );
        let mut config = SyntheticRadarConfig {
            ref_gate_texture: false,
            ..SyntheticRadarConfig::default()
        };
        let air = sample_gate(
            &fields,
            None,
            0.0,
            fields.cells(),
            39.0,
            -95.0,
            200.0,
            90.0,
            30.0,
            2_000.0,
            8,
            250.0,
            &config,
            None,
        )
        .expect("sample air-motion gate")
        .expect("air-motion gate");
        config.terminal_fall_speed = true;
        let scatterer = sample_gate(
            &fields,
            None,
            0.0,
            fields.cells(),
            39.0,
            -95.0,
            200.0,
            90.0,
            30.0,
            2_000.0,
            8,
            250.0,
            &config,
            None,
        )
        .expect("sample scatterer-motion gate")
        .expect("scatterer-motion gate");
        assert!((scatterer.velocity_mps - (air.velocity_mps - 2.5)).abs() < 0.08);
    }

    #[test]
    fn terrain_horizon_blocks_downstream_low_tilt() {
        let mut fields = uniform_box_fields();
        fields.terrain_m.fill(1_500.0);
        let horizon = TerrainHorizon::build(&fields, 39.0, -95.0, 200.0, 36, 20, 0, 500.0);
        let config = SyntheticRadarConfig {
            terrain_blockage: true,
            ref_gate_texture: false,
            ..SyntheticRadarConfig::default()
        };
        let blocked = sample_gate(
            &fields,
            None,
            0.0,
            fields.cells(),
            39.0,
            -95.0,
            200.0,
            90.0,
            0.5,
            2_000.0,
            4,
            500.0,
            &config,
            Some(&horizon),
        )
        .expect("sample blocked gate");
        assert!(
            blocked.is_none(),
            "a 1.5-km ridge must block the 0.5-degree beam"
        );
        assert!(
            sample_gate(
                &fields,
                None,
                0.0,
                fields.cells(),
                39.0,
                -95.0,
                200.0,
                90.0,
                0.5,
                2_000.0,
                4,
                500.0,
                &config,
                None,
            )
            .expect("sample visible gate")
            .is_some(),
            "same model gate is visible without blockage"
        );
    }

    #[test]
    fn synthetic_box_model_samples_ref_and_velocity() {
        let fields = uniform_box_fields();

        let config = SyntheticRadarConfig {
            site_lat_deg: Some(39.0),
            site_lon_deg: Some(-95.0),
            antenna_msl_m: Some(200.0),
            elevations_deg: vec![0.5],
            azimuth_count: 360,
            gate_spacing_m: 250.0,
            max_range_m: 10_000.0,
            // Smooth field: this test asserts exact sampled values, so both
            // textures are off (the default enables reflectivity texture).
            ref_gate_texture: false,
            vel_gate_texture: false,
            ..SyntheticRadarConfig::default()
        };
        let time = DateTime::<Utc>::from_timestamp(1_700_000_000, 0).unwrap();
        let volume = build_synthetic_volume(&fields, time, &config);

        assert_eq!(volume.cuts.len(), 1);
        let cut = &volume.cuts[0];
        let ref_grid = &cut.moments[&MomentType::Reflectivity];
        let vel_grid = &cut.moments[&MomentType::Velocity];

        // Interior gates read the uniform 40 dBZ.
        let mut finite_ref = 0;
        if let MomentStorage::F32(values) = &ref_grid.storage {
            for value in values {
                if value.is_finite() {
                    finite_ref += 1;
                    assert!((value - 40.0).abs() < 0.5, "ref {value}");
                }
            }
        }
        assert!(
            finite_ref > 100,
            "expected many finite REF gates, got {finite_ref}"
        );

        // Radial nearest az=90° (due east): near-ground gates blow away from
        // the radar at ~+10 m/s.
        let east_radial = (90 * 360 / 360) as usize; // az index for 90°
        let vel = vel_grid
            .scaled_value(east_radial, 4)
            .expect("east radial near gate");
        assert!((vel - 10.0).abs() < 1.5, "due-east Vr = {vel}");

        // West radial (az=270°) is the mirror image: toward the radar.
        let west_vel = vel_grid.scaled_value(270, 4).expect("west radial");
        assert!((west_vel + 10.0).abs() < 1.5, "due-west Vr = {west_vel}");
    }

    fn moment_bits(volume: &RadarVolume) -> Vec<u32> {
        let mut bits = Vec::new();
        for cut in &volume.cuts {
            for moment in [MomentType::Reflectivity, MomentType::Velocity] {
                let MomentStorage::F32(values) = &cut.moments[&moment].storage else {
                    panic!("synthetic moments must be F32");
                };
                bits.extend(values.iter().map(|value| value.to_bits()));
            }
        }
        bits
    }

    fn box_model_config() -> SyntheticRadarConfig {
        SyntheticRadarConfig {
            site_lat_deg: Some(39.0),
            site_lon_deg: Some(-95.0),
            antenna_msl_m: Some(200.0),
            elevations_deg: vec![0.5, 3.1],
            azimuth_count: 360,
            gate_spacing_m: 250.0,
            max_range_m: 10_000.0,
            ref_gate_texture: false,
            vel_gate_texture: false,
            ..SyntheticRadarConfig::default()
        }
    }

    #[test]
    fn reflectivity_operator_selects_the_expected_source() {
        use ReflectivityOperator::{ClassicStoelinga, ModelNative};

        // Model native: prefers REFL_10CM when present, falls back to CALCDBZ.
        assert_eq!(planned_ref_source(ModelNative, true), REFL_10CM_SOURCE);
        assert_eq!(planned_ref_source(ModelNative, false), CALCDBZ_SOURCE);

        // Classic Stoelinga forces CALCDBZ EVEN when REFL_10CM is present, and
        // stamps a distinct label so a run documents the deliberate choice
        // rather than a fallback.
        assert_eq!(planned_ref_source(ClassicStoelinga, true), STOELINGA_SOURCE);
        assert_eq!(
            planned_ref_source(ClassicStoelinga, false),
            STOELINGA_SOURCE
        );

        // Only the model-native operator ever reads REFL_10CM.
        assert!(ModelNative.prefers_refl_10cm());
        assert!(!ClassicStoelinga.prefers_refl_10cm());

        // Default operator is model native (the historical behavior).
        assert_eq!(ReflectivityOperator::default(), ModelNative);
        assert_eq!(
            SyntheticRadarConfig::default().reflectivity_operator,
            ModelNative
        );

        // The three labels are distinct so the import note is unambiguous.
        assert_ne!(REFL_10CM_SOURCE, CALCDBZ_SOURCE);
        assert_ne!(CALCDBZ_SOURCE, STOELINGA_SOURCE);
    }

    #[test]
    fn low_tilt_option_prepends_the_community_lowest_tilt() {
        let standard = elevation_ladder(false);
        assert_eq!(
            standard, DEFAULT_ELEVATIONS_DEG,
            "off = the classic ladder, unchanged"
        );

        let with_low = elevation_ladder(true);
        assert_eq!(
            with_low.len(),
            DEFAULT_ELEVATIONS_DEG.len() + 1,
            "one extra tilt"
        );
        assert_eq!(with_low[0], LOW_TILT_DEG, "0.1° comes first");
        assert!(
            with_low[0] < DEFAULT_ELEVATIONS_DEG[0],
            "the extra tilt is below the standard lowest tilt"
        );
        assert_eq!(
            &with_low[1..],
            DEFAULT_ELEVATIONS_DEG,
            "the standard ladder follows unchanged"
        );
    }

    #[test]
    fn physical_scan_plan_keeps_every_build_24_source_row() {
        for strategy in SyntheticScanStrategy::BUILD_24 {
            let config = SyntheticRadarConfig {
                scan_strategy: strategy,
                ..SyntheticRadarConfig::default()
            };
            let definition = strategy.definition().unwrap();
            let legs = config.physical_scan_legs();
            assert_eq!(legs.len(), definition.rows.len(), "{strategy:?}");
            for (index, (leg, row)) in legs.iter().zip(definition.rows).enumerate() {
                assert_eq!(leg.source_row_index, Some(index));
                assert_eq!(leg.source_row, Some(row));
                assert_eq!(leg.elevation_deg, f64::from(row.elevation_deg));
                assert_eq!(
                    leg.azimuth_rate_deg_per_second,
                    row.azimuth_rate_deg_per_second
                );
                assert_eq!(leg.source_period_seconds, row.source_period_seconds);
                assert_eq!(leg.transition_after_seconds, 0.0);
                assert_eq!(leg.moments, row.moments);
                assert_eq!(leg.waveform, Some(row.waveform));
            }
        }
    }

    #[test]
    fn physical_scan_plan_preserves_vcp_112_duplicate_split_and_mpda_cuts() {
        let config = SyntheticRadarConfig {
            scan_strategy: SyntheticScanStrategy::Build24Vcp112,
            ..SyntheticRadarConfig::default()
        };
        let legs = config.physical_scan_legs();
        assert_eq!(legs.len(), 20);
        assert_eq!(
            legs.iter()
                .take(3)
                .map(|leg| leg.elevation_deg)
                .collect::<Vec<_>>(),
            vec![0.5, 0.5, 0.5]
        );
        assert_eq!(legs[0].waveform, Some(Waveform::Sz2ContiguousSurveillance));
        assert_eq!(legs[1].waveform, Some(Waveform::Sz2ContiguousDoppler));
        assert_eq!(legs[2].waveform, Some(Waveform::Sz2ContiguousDoppler));
        assert_eq!(legs[0].moments, MomentCoverage::SURVEILLANCE);
        assert_eq!(legs[1].moments, MomentCoverage::DOPPLER);
        assert_eq!(legs[2].moments, MomentCoverage::DOPPLER);
    }

    #[test]
    fn custom_scan_plan_remains_the_legacy_all_moment_ladder() {
        let config = SyntheticRadarConfig::default();
        assert_eq!(config.scan_strategy, SyntheticScanStrategy::CustomLegacy);
        let legs = config.physical_scan_legs();
        assert_eq!(legs.len(), DEFAULT_ELEVATIONS_DEG.len());
        assert_eq!(
            legs.iter().map(|leg| leg.elevation_deg).collect::<Vec<_>>(),
            DEFAULT_ELEVATIONS_DEG
        );
        assert!(legs.iter().all(|leg| {
            leg.moments == MomentCoverage::ALL && leg.waveform.is_none() && leg.source_row.is_none()
        }));
    }

    #[test]
    fn named_vcp_identity_and_versioned_rows_move_the_fingerprint() {
        let custom = SyntheticRadarConfig::default();
        let vcp12 = SyntheticRadarConfig {
            scan_strategy: SyntheticScanStrategy::Build24Vcp12,
            // VCP 12 has the same unique elevation ladder as the legacy
            // default; identity/physical rows must still distinguish it.
            elevations_deg: DEFAULT_ELEVATIONS_DEG.to_vec(),
            ..custom.clone()
        };
        let vcp212 = SyntheticRadarConfig {
            scan_strategy: SyntheticScanStrategy::Build24Vcp212,
            elevations_deg: DEFAULT_ELEVATIONS_DEG.to_vec(),
            ..custom.clone()
        };
        assert_ne!(custom.data_fingerprint(), vcp12.data_fingerprint());
        assert_ne!(vcp12.data_fingerprint(), vcp212.data_fingerprint());
        assert_eq!(vcp12.data_fingerprint(), vcp12.clone().data_fingerprint());
    }

    #[test]
    fn property_tmatrix_static_contract_is_exact_and_fail_closed() {
        assert!(
            SyntheticRadarConfig::default()
                .validate_science_contract()
                .is_ok()
        );

        let supported = SyntheticRadarConfig {
            dual_pol: true,
            polarimetric_kernel: PolarimetricKernel::PropertyTMatrixResearchV1,
            radar_frequency_mhz: PROPERTY_TMATRIX_RESEARCH_FREQUENCY_MHZ,
            reflectivity_sampling: ReflectivitySampling::LinearZ,
            beam_integration: BeamIntegration::Balanced,
            elevations_deg: elevation_ladder(true),
            ..SyntheticRadarConfig::default()
        };
        let supported_result = supported.validate_science_contract();
        assert!(
            supported_result.is_ok(),
            "the optional 0.1-degree cut and default one-degree beam fit the declared view axis: {supported_result:?}"
        );
        let raw_state = SyntheticRadarConfig {
            atmosphere_time_mode: AtmosphereTimeMode::RawStateLinear,
            scan_timing: ScanTiming::TimedVolume,
            ..supported.clone()
        };
        assert!(raw_state.validate_science_contract().is_ok());
        let hybrid = SyntheticRadarConfig {
            polarimetric_kernel: PolarimetricKernel::PropertyTMatrixHybridV1,
            atmosphere_time_mode: AtmosphereTimeMode::FrozenAtVolumeStart,
            ..supported.clone()
        };
        assert!(hybrid.validate_science_contract().is_ok());
        assert_eq!(
            hybrid.polarimetric_kernel.scattering_policy(),
            app_ui::wrf_tmatrix_scene::WrfTMatrixScatteringPolicy::HybridBulkRayleighV1
        );
        let raw_hybrid = SyntheticRadarConfig {
            atmosphere_time_mode: AtmosphereTimeMode::RawStateLinear,
            scan_timing: ScanTiming::TimedVolume,
            ..hybrid
        };
        assert!(
            raw_hybrid
                .validate_science_contract()
                .unwrap_err()
                .contains("RawStateLinear is available only with experimental Full")
        );
        let hybrid_json = serde_json::to_string(&PolarimetricKernel::PropertyTMatrixHybridV1)
            .expect("serialize Hybrid kernel");
        assert_eq!(hybrid_json, "\"property_t_matrix_hybrid_v1\"");
        assert_eq!(
            serde_json::from_str::<PolarimetricKernel>(&hybrid_json)
                .expect("deserialize Hybrid kernel"),
            PolarimetricKernel::PropertyTMatrixHybridV1
        );
        let raw_with_bulk = SyntheticRadarConfig {
            atmosphere_time_mode: AtmosphereTimeMode::RawStateLinear,
            scan_timing: ScanTiming::TimedVolume,
            ..SyntheticRadarConfig::default()
        };
        assert!(
            raw_with_bulk
                .validate_science_contract()
                .unwrap_err()
                .contains("only with the P3/ISHMAEL property T-matrix")
        );

        let wrong_frequency = SyntheticRadarConfig {
            radar_frequency_mhz: 2_900,
            ..supported.clone()
        };
        assert!(
            wrong_frequency
                .validate_science_contract()
                .unwrap_err()
                .contains("one exact supported choice")
        );
        let legacy_c_band = SyntheticRadarConfig {
            radar_frequency_mhz: 5_600,
            ..supported.clone()
        };
        assert!(
            legacy_c_band
                .validate_science_contract()
                .unwrap_err()
                .contains("BowEcho S research data pack exists only")
        );
        let raw_external = SyntheticRadarConfig {
            atmosphere_time_mode: AtmosphereTimeMode::RawStateLinear,
            scan_timing: ScanTiming::TimedVolume,
            property_tmatrix_table_source:
                app_ui::wrf_tmatrix_assets::PropertyTMatrixTableSourceKind::ExternalValidatedPack,
            ..supported.clone()
        };
        assert!(
            raw_external
                .validate_science_contract()
                .unwrap_err()
                .contains("RawStateLinear currently requires the BowEcho S research pack")
        );
        let raw_frozen_only = SyntheticRadarConfig {
            atmosphere_time_mode: AtmosphereTimeMode::RawStateLinear,
            scan_timing: ScanTiming::TimedVolume,
            property_tmatrix_rain_sensitivity: PropertyTMatrixRainSensitivity::FrozenOnly,
            ..supported.clone()
        };
        assert!(
            raw_frozen_only
                .validate_science_contract()
                .unwrap_err()
                .contains("Full property rain/melting")
        );
        let legacy_log_sampling = SyntheticRadarConfig {
            reflectivity_sampling: ReflectivitySampling::LegacyDbz,
            ..supported.clone()
        };
        assert!(
            legacy_log_sampling
                .validate_science_contract()
                .unwrap_err()
                .contains("linear-Z")
        );
        let no_dual_pol = SyntheticRadarConfig {
            dual_pol: false,
            ..supported.clone()
        };
        assert!(
            no_dual_pol
                .validate_science_contract()
                .unwrap_err()
                .contains("requires dual polarization")
        );
        let too_wide_for_low_cut = SyntheticRadarConfig {
            beam_width_deg: 1.5,
            ..supported
        };
        assert!(
            too_wide_for_low_cut
                .validate_science_contract()
                .unwrap_err()
                .contains("outside the exact")
        );
    }

    #[test]
    fn data_fingerprint_changes_with_every_data_field() {
        use ReflectivityOperator::ClassicStoelinga;

        let base = SyntheticRadarConfig {
            site_id: "WRF".to_string(),
            site_name: Some("Simulated WRF radar".to_string()),
            site_lat_deg: Some(35.0),
            site_lon_deg: Some(-97.0),
            antenna_msl_m: Some(400.0),
            elevations_deg: vec![0.5, 1.5, 2.4],
            azimuth_count: 720,
            gate_spacing_m: 250.0,
            max_range_m: 230_000.0,
            match_gate_to_grid: false,
            ref_floor_dbz: 0.0,
            nyquist_mps: 25.0,
            fold_velocity: false,
            reflectivity_operator: ReflectivityOperator::ModelNative,
            ref_gate_texture: true,
            vel_gate_texture: false,
            clutter_intensity: 0.0,
            ..SyntheticRadarConfig::default()
        };
        let fingerprint = base.data_fingerprint();

        // Stable across a clone (deterministic, no per-run seed).
        assert_eq!(fingerprint, base.clone().data_fingerprint());

        // Presentation-only fields must NOT change the fingerprint.
        let renamed = SyntheticRadarConfig {
            site_name: Some("A different label".to_string()),
            ..base.clone()
        };
        assert_eq!(
            renamed.data_fingerprint(),
            fingerprint,
            "site_name is a label — it must not move the data fingerprint"
        );

        let larger_memory_budget = SyntheticRadarConfig {
            temporal_memory_budget_mib: base.temporal_memory_budget_mib * 2,
            ..base.clone()
        };
        assert_eq!(
            larger_memory_budget.data_fingerprint(),
            fingerprint,
            "memory budget gates execution but does not alter a successful sample"
        );
        let forced_cpu = SyntheticRadarConfig {
            compute_preference: SyntheticRadarComputePreference::Cpu,
            ..base.clone()
        };
        assert_eq!(
            forced_cpu.data_fingerprint(),
            fingerprint,
            "execution preference does not alter the deterministic scientific product"
        );

        // Every data-affecting field must move the fingerprint. `differs`
        // clones the base, applies one edit, and reports whether it changed.
        let differs = |mutate: &dyn Fn(&mut SyntheticRadarConfig)| {
            let mut config = base.clone();
            mutate(&mut config);
            config.data_fingerprint() != fingerprint
        };
        assert!(differs(&|c| c.site_id = "KTLX".to_string()), "site_id");
        assert!(differs(&|c| c.site_lat_deg = Some(36.0)), "site_lat_deg");
        assert!(differs(&|c| c.site_lat_deg = None), "site_lat None");
        assert!(differs(&|c| c.site_lon_deg = Some(-96.0)), "site_lon_deg");
        assert!(differs(&|c| c.antenna_msl_m = Some(401.0)), "antenna_msl");
        assert!(differs(&|c| c.antenna_msl_m = None), "antenna_msl None");
        assert!(
            differs(&|c| c.elevations_deg = vec![0.1, 0.5]),
            "elevations"
        );
        assert!(differs(&|c| c.azimuth_count = 360), "azimuth_count");
        assert!(differs(&|c| c.gate_spacing_m = 500.0), "gate_spacing_m");
        assert!(differs(&|c| c.max_range_m = 460_000.0), "max_range_m");
        assert!(
            differs(&|c| c.match_gate_to_grid = true),
            "match_gate_to_grid — toggling grid-matching resizes gates and must rebuild"
        );
        assert!(differs(&|c| c.ref_floor_dbz = 5.0), "ref_floor_dbz");
        assert!(differs(&|c| c.nyquist_mps = 64.0), "nyquist_mps");
        assert!(
            differs(&|c| c.fold_velocity = true),
            "fold_velocity — toggling folding re-aliases VEL and re-stamps Nyquist"
        );
        assert!(differs(&|c| c.ref_gate_texture = false), "ref_gate_texture");
        assert!(differs(&|c| c.vel_gate_texture = true), "vel_gate_texture");
        assert!(
            differs(&|c| c.clutter_intensity = 0.5),
            "clutter_intensity — the slider must rebuild the volume"
        );
        assert!(
            differs(&|c| c.reflectivity_operator = ClassicStoelinga),
            "operator"
        );
        assert!(
            differs(&|c| c.simulation_mode = SimulationMode::Truth),
            "simulation_mode"
        );
        assert!(
            differs(&|c| c.reflectivity_sampling = ReflectivitySampling::LegacyDbz),
            "reflectivity_sampling"
        );
        assert!(
            differs(&|c| c.beam_integration = BeamIntegration::Balanced),
            "beam_integration"
        );
        assert!(differs(&|c| c.beam_width_deg = 1.2), "beam_width_deg");
        assert!(differs(&|c| c.pulse_width_us = 2.0), "pulse_width_us");
        assert!(
            differs(&|c| c.radar_frequency_mhz = 5_600),
            "radar_frequency_mhz"
        );
        assert!(
            differs(&|c| c.terminal_fall_speed = true),
            "terminal_fall_speed"
        );
        assert!(differs(&|c| c.terrain_blockage = true), "terrain_blockage");
        assert!(differs(&|c| c.spectrum_width = true), "spectrum_width");
        assert!(
            differs(&|c| c.spectrum_width_floor_mps = 0.8),
            "spectrum_width_floor_mps"
        );
        assert!(differs(&|c| c.dual_pol = true), "dual_pol");
        assert!(
            differs(&|c| c.polarimetric_kernel = PolarimetricKernel::PropertyTMatrixResearchV1),
            "polarimetric_kernel"
        );
        assert!(
            differs(&|c| {
                c.property_tmatrix_table_source = app_ui::wrf_tmatrix_assets::PropertyTMatrixTableSourceKind::ExternalValidatedPack
            }),
            "property_tmatrix_table_source"
        );
        assert!(
            differs(&|c| {
                c.property_tmatrix_rain_sensitivity = PropertyTMatrixRainSensitivity::FrozenOnly
            }),
            "property_tmatrix_rain_sensitivity"
        );
        assert!(differs(&|c| c.propagation = true), "propagation");
        assert!(
            differs(&|c| { c.propagation_geometry = PropagationGeometry::WrfRefractivityResearch }),
            "propagation geometry"
        );
        assert!(differs(&|c| c.system_phidp_deg = 11.0), "system_phidp_deg");
        assert!(differs(&|c| c.zdr_bias_db = 0.3), "zdr_bias_db");
        assert!(
            differs(&|c| c.scan_timing = ScanTiming::TimedVolume),
            "scan_timing"
        );
        assert!(
            differs(&|c| c.atmosphere_time_mode = AtmosphereTimeMode::LinearAdjacent),
            "atmosphere_time_mode"
        );
        assert!(
            differs(&|c| c.missing_neighbor_policy = MissingNeighborPolicy::DropFrame),
            "missing_neighbor_policy"
        );
        assert!(
            differs(&|c| c.rotation_rate_deg_s = 12.0),
            "rotation_rate_deg_s"
        );
        assert!(
            differs(&|c| c.transition_delay_s = 5.0),
            "transition_delay_s"
        );
        assert!(differs(&|c| c.prf_hz = 1_200.0), "prf_hz");
        assert!(
            differs(&|c| c.coupled_single_prf_estimator = true),
            "coupled_single_prf_estimator"
        );
        assert!(
            differs(&|c| c.estimator_dwell_ms = 75.0),
            "estimator_dwell_ms"
        );
        assert!(
            differs(&|c| c.estimator_pulse_count = Some(64)),
            "estimator_pulse_count"
        );
        assert!(
            differs(&|c| c.estimator_independent_sample_fraction = 0.75),
            "estimator_independent_sample_fraction"
        );
        assert!(
            differs(&|c| c.estimator_minimum_snr_db = 3.0),
            "estimator_minimum_snr_db"
        );
        assert!(
            differs(&|c| c.emit_stage_diagnostics = true),
            "emit_stage_diagnostics"
        );
        assert!(differs(&|c| c.instrument_noise = true), "instrument_noise");
        assert!(
            differs(&|c| c.sensitivity_dbz_at_1km = -38.0),
            "sensitivity_dbz_at_1km"
        );
        assert!(
            differs(&|c| c.emit_quality_fields = false),
            "emit_quality_fields"
        );
        assert!(
            differs(&|c| c.minimum_model_coverage_fraction = 0.75),
            "minimum_model_coverage_fraction"
        );
    }

    #[test]
    fn coupled_custom_instrument_resolves_one_physical_timing_contract() {
        let config = SyntheticRadarConfig {
            coupled_single_prf_estimator: true,
            radar_frequency_mhz: 2_800,
            pulse_width_us: 2.0,
            prf_hz: 1_200.0,
            estimator_dwell_ms: 40.0,
            estimator_pulse_count: None,
            estimator_independent_sample_fraction: 0.5,
            ..SyntheticRadarConfig::default()
        };
        config.validate_science_contract().unwrap();
        let coupled = resolve_coupled_instrument(&config).unwrap().unwrap();
        assert!((coupled.timing.prf_hz - 1_200.0).abs() < f64::EPSILON);
        assert!((coupled.timing.prt_s - 1.0 / 1_200.0).abs() < 1.0e-15);
        assert_eq!(coupled.sampling.transmitted_pulses, 48);
        assert!((coupled.sampling.independent_samples - 24.0).abs() < f64::EPSILON);
        assert_eq!(coupled.balanced_quadrature.len(), 9);
        assert_eq!(coupled.reference_quadrature.len(), 27);
        for points in [&coupled.balanced_quadrature, &coupled.reference_quadrature] {
            let weight_sum = points.iter().map(|point| point.weight).sum::<f64>();
            assert!((weight_sum - 1.0).abs() < 1.0e-12);
            assert!(points.iter().any(|point| point.range_offset_m < 0.0));
            assert!(points.iter().any(|point| point.range_offset_m > 0.0));
            assert!(
                points
                    .iter()
                    .all(|point| point.range_offset_m.abs() <= coupled.range_resolution_m)
            );
        }

        let different_gate_spacing = SyntheticRadarConfig {
            gate_spacing_m: 4_000.0,
            ..config
        };
        let other = resolve_coupled_instrument(&different_gate_spacing)
            .unwrap()
            .unwrap();
        assert_eq!(
            coupled
                .reference_quadrature
                .iter()
                .map(|point| point.range_offset_m.to_bits())
                .collect::<Vec<_>>(),
            other
                .reference_quadrature
                .iter()
                .map(|point| point.range_offset_m.to_bits())
                .collect::<Vec<_>>(),
            "matched-filter range offsets depend on pulse width, not gate spacing"
        );
    }

    #[test]
    fn coupled_estimator_rejects_named_vcp_prf_codes() {
        let config = SyntheticRadarConfig {
            coupled_single_prf_estimator: true,
            scan_strategy: SyntheticScanStrategy::Build24Vcp12,
            ..SyntheticRadarConfig::default()
        };
        let error = config.validate_science_contract().unwrap_err();
        assert!(error.contains("PRF codes are identifiers, not frequencies"));
        assert!(resolve_coupled_instrument(&config).is_err());

        let named_without_physical_prf = SyntheticRadarConfig {
            coupled_single_prf_estimator: false,
            scan_strategy: SyntheticScanStrategy::Build24Vcp12,
            azimuth_count: 4,
            gate_spacing_m: 500.0,
            max_range_m: 500.0,
            ref_gate_texture: false,
            ..box_model_config()
        };
        let volume = build_synthetic_volume(
            &uniform_box_fields(),
            DateTime::<Utc>::from_timestamp(1_700_000_000, 0).unwrap(),
            &named_without_physical_prf,
        );
        assert!(
            volume
                .cuts
                .iter()
                .all(|cut| cut.ray_instrument_metadata.is_empty()),
            "named VCP PRF codes never become physical per-ray timing"
        );
    }

    #[test]
    fn coupled_builder_stamps_timing_and_opt_in_stage_grids() {
        let fields = uniform_box_fields();
        let config = SyntheticRadarConfig {
            coupled_single_prf_estimator: true,
            emit_stage_diagnostics: true,
            elevations_deg: vec![0.5],
            azimuth_count: 4,
            gate_spacing_m: 500.0,
            max_range_m: 2_000.0,
            beam_integration: BeamIntegration::Balanced,
            spectrum_width: true,
            ref_gate_texture: false,
            vel_gate_texture: false,
            prf_hz: 1_000.0,
            estimator_dwell_ms: 50.0,
            estimator_pulse_count: Some(50),
            ..box_model_config()
        };
        let coupled = resolve_coupled_instrument(&config).unwrap().unwrap();
        let time = DateTime::<Utc>::from_timestamp(1_700_000_000, 0).unwrap();
        let volume = build_synthetic_volume(&fields, time, &config);
        assert!(
            stamped_nyquists(&volume)
                .iter()
                .all(|nyquist| { (*nyquist - coupled.stamped_nyquist_mps()).abs() < f32::EPSILON })
        );
        assert_eq!(
            volume.metadata.prt_s.unwrap().to_bits(),
            (coupled.timing.prt_s as f32).to_bits()
        );
        assert_eq!(
            volume.metadata.unambiguous_range_km.unwrap().to_bits(),
            ((coupled.timing.unambiguous_range_m / 1_000.0) as f32).to_bits()
        );
        let ray_metadata = volume.cuts[0]
            .aligned_ray_instrument_metadata()
            .unwrap()
            .expect("coupled synthetic rays carry physical metadata");
        assert_eq!(ray_metadata.len(), volume.cuts[0].radials.len());
        for metadata in ray_metadata {
            assert_eq!(
                metadata.prt_s.unwrap().to_bits(),
                (coupled.timing.prt_s as f32).to_bits()
            );
            assert_eq!(
                metadata.unambiguous_range_km.unwrap().to_bits(),
                ((coupled.timing.unambiguous_range_m / 1_000.0) as f32).to_bits()
            );
            assert_eq!(metadata.pulse_count, Some(50));
            assert_eq!(metadata.independent_samples, Some(25.0));
        }
        let provenance = volume.metadata.forward_operator_config.as_deref().unwrap();
        assert!(provenance.contains("estimator=CustomSinglePrfV1"));
        assert!(provenance.contains("transmitted_pulses=50"));
        assert!(provenance.contains(IDEAL_STAGE_DEFINITION));
        assert!(provenance.contains(MEASURED_STAGE_DEFINITION));
        assert!(provenance.contains(PRESENTED_STAGE_DEFINITION));
        for name in [
            "IREF", "IVEL", "ISW", "IZDR", "IRHO", "IKDP", "MREF", "MVEL", "MSW", "MZDR", "MRHO",
            "MKDP",
        ] {
            let moment = MomentType::Unknown(name.to_string());
            let grid = volume.cuts[0]
                .moments
                .get(&moment)
                .unwrap_or_else(|| panic!("missing {name} stage diagnostic"));
            assert!(matches!(&grid.storage, MomentStorage::F32(_)));
        }
    }

    fn irregular_observed_replay_volume() -> RadarVolume {
        let mut site = RadarSite::new("KOBS");
        site.name = Some("Observed replay fixture".to_string());
        site.latitude_deg = Some(39.0);
        site.longitude_deg = Some(-95.0);
        site.elevation_m = Some(200.0);
        let time = DateTime::<Utc>::from_timestamp(1_700_000_000, 0).unwrap();
        let mut volume = RadarVolume::new(site, time);
        volume.vcp = Some(VcpInfo { pattern: 212 });
        volume.metadata.archive_version = Some("AR2V0006".to_string());
        volume.metadata.prt_s = Some(0.001_25);
        volume.metadata.unambiguous_range_km = Some(119.9);
        let midnight_ms = ((time.timestamp() % 86_400) * 1_000) as i32;
        for split in 0..2usize {
            let mut cut = ElevationCut::new(0.5, Some((split + 1) as u8));
            for (ray_index, azimuth_deg) in [12.25, 97.75, 281.5].into_iter().enumerate() {
                cut.radials.push(Radial {
                    azimuth_deg,
                    elevation_deg: 0.48 + split as f32 * 0.04 + ray_index as f32 * 0.01,
                    time_offset_ms: midnight_ms + (split * 2_000 + ray_index * 175) as i32,
                    gate_range: GateRange {
                        first_gate_m: 125 + ray_index as i32 * 25,
                        gate_spacing_m: 250,
                        gate_count: 5 - usize::from(ray_index == 2),
                    },
                    nyquist_velocity_mps: Some(18.0 + ray_index as f32),
                    radial_status: Some(if ray_index == 0 {
                        radar_core::RadialStatus::StartElevation
                    } else if ray_index == 2 {
                        radar_core::RadialStatus::EndElevation
                    } else {
                        radar_core::RadialStatus::Intermediate
                    }),
                });
            }
            let ref_range = GateRange {
                first_gate_m: 125,
                gate_spacing_m: 250,
                gate_count: 4,
            };
            cut.moments.insert(
                MomentType::Reflectivity,
                f32_grid(
                    MomentType::Reflectivity,
                    ref_range,
                    if split == 0 {
                        vec![0, 2]
                    } else {
                        vec![0, 1, 2]
                    },
                    if split == 0 {
                        vec![20.0; 8]
                    } else {
                        vec![25.0; 12]
                    },
                ),
            );
            if split == 0 {
                cut.moments.insert(
                    MomentType::Velocity,
                    f32_grid(
                        MomentType::Velocity,
                        GateRange {
                            first_gate_m: 500,
                            gate_spacing_m: 500,
                            gate_count: 2,
                        },
                        vec![1, 2],
                        vec![3.0, 4.0, -3.0, -4.0],
                    ),
                );
            }
            volume.cuts.push(cut);
        }
        volume.metadata.decoded_radial_count = 6;
        volume
    }

    #[test]
    fn exact_observed_replay_preserves_irregular_geometry_and_builds_difference() {
        let fields = uniform_box_fields();
        let mut observed_volume = irregular_observed_replay_volume();
        for (cut_index, cut) in observed_volume.cuts.iter_mut().enumerate() {
            cut.ray_instrument_metadata = (0..cut.radials.len())
                .map(|ray_index| RayInstrumentMetadata {
                    prt_s: Some(0.000_8 + cut_index as f32 * 0.000_1),
                    unambiguous_range_km: Some(100.0 + ray_index as f32),
                    pulse_count: Some(40 + ray_index as u32),
                    independent_samples: Some(12.0 + ray_index as f32),
                })
                .collect();
        }
        let observed = Arc::new(observed_volume);
        let observed_handle = Arc::clone(&observed);
        let config = SyntheticRadarConfig {
            ref_gate_texture: false,
            vel_gate_texture: false,
            emit_quality_fields: false,
            ..box_model_config()
        };
        let products = build_exact_replay_products(&fields, observed, &config).unwrap();
        assert!(Arc::ptr_eq(&products.observed, &observed_handle));
        assert!(products.unavailable_observed_moments.is_empty());
        let simulated = &products.simulated;
        assert_eq!(simulated.site, observed_handle.site);
        assert_eq!(simulated.volume_time, observed_handle.volume_time);
        assert_eq!(simulated.vcp, observed_handle.vcp);
        assert_eq!(simulated.metadata.prt_s, observed_handle.metadata.prt_s);
        assert_eq!(
            simulated.metadata.unambiguous_range_km,
            observed_handle.metadata.unambiguous_range_km
        );
        assert_eq!(simulated.cuts.len(), 2);
        for (observed_cut, simulated_cut) in observed_handle.cuts.iter().zip(&simulated.cuts) {
            assert_eq!(
                simulated_cut.ray_instrument_metadata,
                observed_cut.ray_instrument_metadata
            );
            assert_eq!(simulated_cut.elevation_deg, observed_cut.elevation_deg);
            assert_eq!(
                simulated_cut.elevation_number,
                observed_cut.elevation_number
            );
            assert_eq!(simulated_cut.radials.len(), observed_cut.radials.len());
            for (observed_ray, simulated_ray) in
                observed_cut.radials.iter().zip(&simulated_cut.radials)
            {
                assert_eq!(simulated_ray.azimuth_deg, observed_ray.azimuth_deg);
                assert_eq!(simulated_ray.elevation_deg, observed_ray.elevation_deg);
                assert_eq!(simulated_ray.gate_range, observed_ray.gate_range);
                assert_eq!(
                    simulated_ray.nyquist_velocity_mps,
                    observed_ray.nyquist_velocity_mps
                );
                assert_eq!(simulated_ray.radial_status, observed_ray.radial_status);
                assert_eq!(
                    app_ui::wrf_radar_validation::radial_acquisition_time_utc(
                        &observed_handle,
                        observed_ray
                    ),
                    app_ui::wrf_radar_validation::radial_acquisition_time_utc(
                        simulated,
                        simulated_ray
                    )
                );
            }
            for (moment, observed_grid) in &observed_cut.moments {
                let simulated_grid = &simulated_cut.moments[moment];
                assert_eq!(simulated_grid.gate_range, observed_grid.gate_range);
                assert_eq!(simulated_grid.radial_indices, observed_grid.radial_indices);
            }
            for quality in QualityMoment::ALL {
                assert!(simulated_cut.moments.contains_key(&quality.moment_type()));
            }
        }
        assert!(
            simulated
                .metadata
                .forward_operator_config
                .as_deref()
                .unwrap()
                .contains("vcp_reconstruction=false")
        );
        assert_eq!(products.difference.cuts.len(), observed_handle.cuts.len());
        assert!(
            products.difference.cuts[0]
                .moments
                .contains_key(&MomentType::Unknown("DIF_REF".to_string()))
        );
        assert!(
            products.difference.cuts[0]
                .moments
                .contains_key(&MomentType::Unknown("DIF_VEL".to_string()))
        );
    }

    #[test]
    fn exact_replay_geometry_moves_config_fingerprint() {
        let observed = irregular_observed_replay_volume();
        let first = Arc::new(ExactScanTemplate::from_volume(&observed).unwrap());
        let mut changed = observed.clone();
        changed.cuts[0].radials[1].azimuth_deg += 0.25;
        let second = Arc::new(ExactScanTemplate::from_volume(&changed).unwrap());
        let first_config = SyntheticRadarConfig {
            exact_replay_template: Some(first),
            ..SyntheticRadarConfig::default()
        };
        let second_config = SyntheticRadarConfig {
            exact_replay_template: Some(second),
            ..SyntheticRadarConfig::default()
        };
        assert_ne!(
            first_config.data_fingerprint(),
            second_config.data_fingerprint()
        );
    }

    #[test]
    fn exact_replay_reports_unavailable_observed_polar_moment() {
        let mut observed = irregular_observed_replay_volume();
        observed.cuts[0].moments.insert(
            MomentType::DifferentialReflectivity,
            f32_grid(
                MomentType::DifferentialReflectivity,
                GateRange {
                    first_gate_m: 125,
                    gate_spacing_m: 250,
                    gate_count: 4,
                },
                vec![0, 2],
                vec![1.0; 8],
            ),
        );
        let products = build_exact_replay_products(
            &uniform_box_fields(),
            Arc::new(observed),
            &SyntheticRadarConfig {
                ref_gate_texture: false,
                ..box_model_config()
            },
        )
        .unwrap();
        assert_eq!(products.unavailable_observed_moments.len(), 1);
        assert_eq!(products.unavailable_observed_moments[0].moment, "ZDR");
        assert!(
            !products.simulated.cuts[0]
                .moments
                .contains_key(&MomentType::DifferentialReflectivity)
        );
        assert!(
            !products.difference.cuts[0]
                .moments
                .contains_key(&MomentType::Unknown("DIF_ZDR".to_string()))
        );
    }

    #[test]
    fn effective_gate_spacing_resolves_match_and_fallback() {
        let base = SyntheticRadarConfig {
            gate_spacing_m: 250.0,
            ..SyntheticRadarConfig::default()
        };

        // Matching OFF: the configured spacing, regardless of any DX.
        assert_eq!(effective_gate_spacing(&base, None), 250.0);
        assert_eq!(effective_gate_spacing(&base, Some(3000.0)), 250.0);

        let matched = SyntheticRadarConfig {
            match_gate_to_grid: true,
            ..base.clone()
        };

        // Matching ON with a valid DX: the grid resolution (a 3 km grid → 3 km
        // gates, ~77 gates over 230 km instead of 920).
        assert_eq!(effective_gate_spacing(&matched, Some(3000.0)), 3000.0);
        assert_eq!(effective_gate_spacing(&matched, Some(250.0)), 250.0);

        // Clamped both ways so a garbage-but-positive DX can't blow up / collapse
        // the gate count.
        assert_eq!(
            effective_gate_spacing(&matched, Some(5.0)),
            GRID_GATE_MIN_M,
            "a sub-100 m DX clamps up to the floor"
        );
        assert_eq!(
            effective_gate_spacing(&matched, Some(50_000.0)),
            GRID_GATE_MAX_M,
            "a huge DX clamps down to the ceiling"
        );

        // Matching ON but DX missing / invalid: fall back to the configured value.
        assert_eq!(
            effective_gate_spacing(&matched, None),
            250.0,
            "no DX attribute → configured spacing"
        );
        assert_eq!(
            effective_gate_spacing(&matched, Some(0.0)),
            250.0,
            "DX == 0 is invalid → configured spacing"
        );
        assert_eq!(
            effective_gate_spacing(&matched, Some(-3000.0)),
            250.0,
            "negative DX is invalid → configured spacing"
        );
        assert_eq!(
            effective_gate_spacing(&matched, Some(f64::NAN)),
            250.0,
            "NaN DX is invalid → configured spacing"
        );

        // `matched_grid_dx` (drives the import note) agrees with the resolver.
        assert!(matched_grid_dx(&matched, Some(3000.0)));
        assert!(!matched_grid_dx(&matched, None));
        assert!(!matched_grid_dx(&matched, Some(0.0)));
        assert!(
            !matched_grid_dx(&base, Some(3000.0)),
            "off → never grid-matched"
        );
    }

    #[test]
    fn build_honours_match_gate_to_grid_dx() {
        let config = SyntheticRadarConfig {
            site_lat_deg: Some(39.0),
            site_lon_deg: Some(-95.0),
            antenna_msl_m: Some(200.0),
            elevations_deg: vec![0.5],
            azimuth_count: 90,
            gate_spacing_m: 250.0,
            max_range_m: 10_000.0,
            match_gate_to_grid: true,
            ref_gate_texture: false,
            vel_gate_texture: false,
            ..SyntheticRadarConfig::default()
        };
        let time = DateTime::<Utc>::from_timestamp(1_700_000_000, 0).unwrap();

        // DX = 1 km on the fields → 1 km gates (10 over 10 km), overriding the
        // configured 250 m.
        let mut fields = uniform_box_fields();
        fields.dx_m = Some(1000.0);
        let volume = build_synthetic_volume(&fields, time, &config);
        let grid = &volume.cuts[0].moments[&MomentType::Reflectivity];
        assert_eq!(grid.gate_range.gate_spacing_m, 1000);
        assert_eq!(grid.gate_range.gate_count, 10);

        // No DX on the fields → the configured 250 m (40 over 10 km).
        let mut no_dx = uniform_box_fields();
        no_dx.dx_m = None;
        let fallback = build_synthetic_volume(&no_dx, time, &config);
        let fb_grid = &fallback.cuts[0].moments[&MomentType::Reflectivity];
        assert_eq!(fb_grid.gate_range.gate_spacing_m, 250);
        assert_eq!(fb_grid.gate_range.gate_count, 40);
    }

    fn stamped_nyquists(volume: &RadarVolume) -> Vec<f32> {
        volume
            .cuts
            .iter()
            .flat_map(|cut| cut.radials.iter())
            .filter_map(|radial| radial.nyquist_velocity_mps)
            .collect()
    }

    fn finite_velocities(volume: &RadarVolume) -> Vec<f32> {
        let mut out = Vec::new();
        for cut in &volume.cuts {
            let MomentStorage::F32(vels) = &cut.moments[&MomentType::Velocity].storage else {
                panic!("synthetic velocity must be F32");
            };
            out.extend(vels.iter().copied().filter(|v| v.is_finite()));
        }
        out
    }

    #[test]
    fn fold_velocity_math_matches_level_ii_convention() {
        let vn = 25.0f32;

        // Inside the co-interval: identity.
        assert_eq!(fold_velocity_mps(0.0, vn), 0.0);
        assert_eq!(fold_velocity_mps(5.0, vn), 5.0);
        assert_eq!(fold_velocity_mps(-7.0, vn), -7.0);
        assert_eq!(fold_velocity_mps(24.0, vn), 24.0);

        // One Nyquist past the top wraps to just inside the bottom, and mirror.
        assert_eq!(fold_velocity_mps(vn + 5.0, vn), -vn + 5.0); // 30 -> -20
        assert_eq!(fold_velocity_mps(-vn - 5.0, vn), vn - 5.0); // -30 -> 20

        // Whole co-intervals fold cleanly: 2·Vn -> 0, 3·Vn -> -Vn.
        assert_eq!(fold_velocity_mps(2.0 * vn, vn), 0.0);
        assert_eq!(fold_velocity_mps(3.0 * vn, vn), -vn);

        // Boundary convention: half-open [-Vn, +Vn) — BOTH +Vn and -Vn map to
        // -Vn (+Vn aliases in, -Vn is already the representable end).
        assert_eq!(fold_velocity_mps(vn, vn), -vn);
        assert_eq!(fold_velocity_mps(-vn, vn), -vn);

        // Sweep: every fold stays in the co-interval and differs from the true
        // value only by a whole multiple of 2·Vn (pure aliasing, no distortion).
        let mut v = -103.0f32;
        while v <= 103.0 {
            let r = fold_velocity_mps(v, vn);
            assert!(
                (-vn - 1e-4..vn + 1e-4).contains(&r),
                "fold({v}) = {r} escaped [-{vn}, {vn})"
            );
            let k = (v - r) / (2.0 * vn);
            assert!(
                (k - k.round()).abs() < 1e-3,
                "fold({v}) = {r} is not a whole-Nyquist alias of the truth (k = {k})"
            );
            // Already-folded values are fixed points (idempotent).
            assert!((fold_velocity_mps(r, vn) - r).abs() < 1e-4);
            v += 0.37;
        }

        // Missing/degenerate inputs pass through untouched.
        assert!(fold_velocity_mps(f32::NAN, vn).is_nan());
        assert_eq!(fold_velocity_mps(12.0, 0.0), 12.0, "Vn=0 is a no-op");
        assert_eq!(fold_velocity_mps(12.0, -5.0), 12.0, "Vn<0 is a no-op");
    }

    #[test]
    fn stamped_nyquist_reports_320_off_and_the_fold_nyquist_on() {
        let off = SyntheticRadarConfig::default();
        assert!(!off.fold_velocity);
        assert_eq!(off.stamped_nyquist_mps(), UNFOLDED_NYQUIST_MPS);
        assert_eq!(UNFOLDED_NYQUIST_MPS, 320.0);

        let on = SyntheticRadarConfig {
            fold_velocity: true,
            nyquist_mps: 18.0,
            ..SyntheticRadarConfig::default()
        };
        assert_eq!(on.stamped_nyquist_mps(), 18.0);

        // Folding off: nyquist_mps is inert (still stamps the historical 320).
        let off_custom = SyntheticRadarConfig {
            fold_velocity: false,
            nyquist_mps: 18.0,
            ..SyntheticRadarConfig::default()
        };
        assert_eq!(off_custom.stamped_nyquist_mps(), UNFOLDED_NYQUIST_MPS);

        // The library default folding Nyquist is the documented 25 m/s.
        assert_eq!(off.nyquist_mps, DEFAULT_FOLD_NYQUIST_MPS);
        assert_eq!(DEFAULT_FOLD_NYQUIST_MPS, 25.0);
    }

    #[test]
    fn folding_off_stamps_320_and_leaves_velocity_unfolded() {
        let fields = uniform_box_fields();
        let time = DateTime::<Utc>::from_timestamp(1_700_000_000, 0).unwrap();

        let default_off = box_model_config(); // fold_velocity false, nyquist 25
        let other_nyquist = SyntheticRadarConfig {
            nyquist_mps: 999.0,
            ..box_model_config()
        };
        assert!(!default_off.fold_velocity && !other_nyquist.fold_velocity);

        let a = build_synthetic_volume(&fields, time, &default_off);
        let b = build_synthetic_volume(&fields, time, &other_nyquist);

        // Every radial stamped with the historical unfolded Nyquist.
        for nyq in stamped_nyquists(&a) {
            assert_eq!(
                nyq, UNFOLDED_NYQUIST_MPS,
                "off must stamp the historical 320"
            );
        }
        // No gate folded: the 10 m/s box wind stays inside a wide margin.
        for v in finite_velocities(&a) {
            assert!(v.abs() <= 12.0, "unfolded Vr {v} must not be aliased");
        }
        // `nyquist_mps` is inert with folding off: identical gates AND stamps.
        assert!(
            moment_bits(&a) == moment_bits(&b),
            "nyquist_mps must not touch the data when folding is off"
        );
        assert_eq!(stamped_nyquists(&a), stamped_nyquists(&b));
    }

    #[test]
    fn folding_on_aliases_velocity_and_stamps_the_nyquist() {
        let fields = uniform_box_fields();
        let time = DateTime::<Utc>::from_timestamp(1_700_000_000, 0).unwrap();
        let vn = 8.0f32;

        let off = box_model_config();
        let on = SyntheticRadarConfig {
            fold_velocity: true,
            nyquist_mps: vn,
            ..box_model_config()
        };
        let off_vol = build_synthetic_volume(&fields, time, &off);
        let on_vol = build_synthetic_volume(&fields, time, &on);

        // Every radial now carries the folding Nyquist as the truth.
        for nyq in stamped_nyquists(&on_vol) {
            assert_eq!(nyq, vn, "folding on must stamp the folding Nyquist");
        }

        // Gate-by-gate: the folded field is exactly the fold of the true field,
        // and at least some gates genuinely wrapped (peak wind > Vn).
        let mut wrapped = 0usize;
        for (cut_off, cut_on) in off_vol.cuts.iter().zip(&on_vol.cuts) {
            let (MomentStorage::F32(a), MomentStorage::F32(b)) = (
                &cut_off.moments[&MomentType::Velocity].storage,
                &cut_on.moments[&MomentType::Velocity].storage,
            ) else {
                panic!("F32");
            };
            for (va, vb) in a.iter().zip(b) {
                assert_eq!(va.is_finite(), vb.is_finite());
                if !va.is_finite() {
                    continue;
                }
                let expect = fold_velocity_mps(*va, vn);
                assert!(
                    (vb - expect).abs() < 1e-4,
                    "gate {va} -> {vb}, expected {expect}"
                );
                assert!(
                    (-vn - 1e-4..vn + 1e-4).contains(vb),
                    "folded {vb} left co-interval"
                );
                if (vb - va).abs() > 1e-3 {
                    wrapped += 1;
                }
            }
        }
        assert!(
            wrapped > 0,
            "the ~10 m/s box wind must fold past an 8 m/s Nyquist"
        );
    }

    #[test]
    fn folding_leaves_near_zero_clutter_gates_untouched() {
        let fields = uniform_box_fields();
        let time = DateTime::<Utc>::from_timestamp(1_700_000_000, 0).unwrap();
        let vn = 8.0f32;

        // Full ground clutter, folding off vs on at the same 8 m/s Nyquist.
        let off = clutter_box_config(1.0);
        let on = SyntheticRadarConfig {
            fold_velocity: true,
            nyquist_mps: vn,
            ..clutter_box_config(1.0)
        };
        let off_vol = build_synthetic_volume(&fields, time, &off);
        let on_vol = build_synthetic_volume(&fields, time, &on);

        let mut near_zero = 0usize;
        for (cut_off, cut_on) in off_vol.cuts.iter().zip(&on_vol.cuts) {
            let (MomentStorage::F32(a), MomentStorage::F32(b)) = (
                &cut_off.moments[&MomentType::Velocity].storage,
                &cut_on.moments[&MomentType::Velocity].storage,
            ) else {
                panic!("F32");
            };
            for (va, vb) in a.iter().zip(b) {
                if va.is_finite() && va.abs() <= 0.5 {
                    // The clutter ±0.5 m/s band: folding must be the identity here.
                    assert_eq!(
                        vb.to_bits(),
                        va.to_bits(),
                        "near-zero gate {va} must survive folding untouched"
                    );
                    near_zero += 1;
                }
            }
        }
        assert!(
            near_zero > 0,
            "expected near-zero (clutter-dominated) gates in the clutter box"
        );
    }

    #[test]
    fn synthetic_frame_path_keys_on_fingerprint_and_frame() {
        let fp = 0x1234_5678_9abc_def0u64;
        let a = synthetic_frame_path("WRF", fp, 0, "20250621_013000");
        // Deterministic: same inputs → identical key (so an unchanged re-import
        // hits the reuse path).
        assert_eq!(a, synthetic_frame_path("WRF", fp, 0, "20250621_013000"));
        // A different fingerprint → different key (the config-change trigger).
        assert_ne!(a, synthetic_frame_path("WRF", 1, 0, "20250621_013000"));
        // Per-frame uniqueness: index, stamp, and site all discriminate.
        assert_ne!(a, synthetic_frame_path("WRF", fp, 1, "20250621_013000"));
        assert_ne!(a, synthetic_frame_path("WRF", fp, 0, "20250621_014500"));
        assert_ne!(a, synthetic_frame_path("KTLX", fp, 0, "20250621_013000"));
        // Fixed-width hex keeps the key stable/greppable.
        assert!(a.to_string_lossy().contains("123456789abcdef0"));
    }

    #[test]
    fn texture_defaults_ref_on_velocity_off_and_off_builds_are_bit_identical() {
        let defaults = SyntheticRadarConfig::default();
        assert!(
            defaults.ref_gate_texture,
            "reflectivity texture ships ON — the smooth field looks garbage without it"
        );
        assert!(
            !defaults.vel_gate_texture,
            "velocity texture stays opt-in — the clean Vr feeds dealias/GBVTD"
        );

        let fields = uniform_box_fields();
        // box_model_config() is the both-off baseline; an explicit both-off
        // build must be bit-identical to it.
        let config = box_model_config();
        assert!(!config.ref_gate_texture && !config.vel_gate_texture);
        let off = SyntheticRadarConfig {
            ref_gate_texture: false,
            vel_gate_texture: false,
            ..config.clone()
        };
        let time = DateTime::<Utc>::from_timestamp(1_700_000_000, 0).unwrap();
        let a = build_synthetic_volume(&fields, time, &config);
        let b = build_synthetic_volume(&fields, time, &off);
        assert!(
            moment_bits(&a) == moment_bits(&b),
            "both textures off must be bit-identical to a textureless build"
        );
    }

    #[test]
    fn gate_texture_is_deterministic_bounded_and_gentle_on_velocity() {
        let fields = uniform_box_fields();
        let time = DateTime::<Utc>::from_timestamp(1_700_000_000, 0).unwrap();
        let off_config = box_model_config();
        let on_config = SyntheticRadarConfig {
            ref_gate_texture: true,
            vel_gate_texture: true,
            ..off_config.clone()
        };
        let off = build_synthetic_volume(&fields, time, &off_config);
        let on = build_synthetic_volume(&fields, time, &on_config);
        let on_again = build_synthetic_volume(&fields, time, &on_config);
        assert!(
            moment_bits(&on) == moment_bits(&on_again),
            "texture must be deterministic frame to frame"
        );

        let mut compared = 0usize;
        let mut ref_changed = 0usize;
        let mut max_ref_diff = 0.0f32;
        for (cut_off, cut_on) in off.cuts.iter().zip(&on.cuts) {
            for (moment, bound) in [
                (MomentType::Reflectivity, 2.51f32),
                (MomentType::Velocity, 0.51f32),
            ] {
                let MomentStorage::F32(a) = &cut_off.moments[&moment].storage else {
                    panic!("F32");
                };
                let MomentStorage::F32(b) = &cut_on.moments[&moment].storage else {
                    panic!("F32");
                };
                for (va, vb) in a.iter().zip(b) {
                    // Uniform 40 dBZ against a 0 dBZ floor: texture can
                    // never flip a gate across the floor here, so finite
                    // patterns must match exactly.
                    assert_eq!(va.is_finite(), vb.is_finite());
                    if !va.is_finite() {
                        continue;
                    }
                    let diff = (vb - va).abs();
                    assert!(diff <= bound, "{moment:?} perturbed by {diff}");
                    if moment == MomentType::Reflectivity {
                        compared += 1;
                        max_ref_diff = max_ref_diff.max(diff);
                        if diff > 0.05 {
                            ref_changed += 1;
                        }
                    }
                }
            }
        }
        assert!(compared > 10_000, "too few echo gates: {compared}");
        assert!(
            ref_changed * 2 > compared,
            "texture must actually move most REF gates ({ref_changed}/{compared})"
        );
        assert!(
            max_ref_diff > 1.5,
            "texture peak {max_ref_diff} dB is too tame to read as speckle"
        );
    }

    #[test]
    fn gate_texture_pokes_ragged_gates_through_the_ref_floor() {
        let fields = uniform_box_fields();
        let time = DateTime::<Utc>::from_timestamp(1_700_000_000, 0).unwrap();
        let off_config = SyntheticRadarConfig {
            ref_floor_dbz: 40.5,
            ..box_model_config()
        };
        let on_config = SyntheticRadarConfig {
            ref_gate_texture: true,
            ..off_config.clone()
        };
        let count_finite = |volume: &RadarVolume| {
            moment_bits(volume)
                .iter()
                .filter(|bits| f32::from_bits(**bits).is_finite())
                .count()
        };
        let off = build_synthetic_volume(&fields, time, &off_config);
        assert_eq!(
            count_finite(&off),
            0,
            "smooth 40 dBZ under a 40.5 floor must stay empty"
        );
        let on = build_synthetic_volume(&fields, time, &on_config);
        assert!(
            count_finite(&on) > 0,
            "texture must poke some gates through the floor"
        );
        for cut in &on.cuts {
            let MomentStorage::F32(values) = &cut.moments[&MomentType::Reflectivity].storage else {
                panic!("F32");
            };
            for value in values.iter().filter(|value| value.is_finite()) {
                assert!(
                    (40.5..=42.6).contains(value),
                    "ragged-edge gate {value} outside the floor..floor+peak band"
                );
            }
        }
    }

    fn clutter_box_config(intensity: f32) -> SyntheticRadarConfig {
        SyntheticRadarConfig {
            clutter_intensity: intensity,
            ..box_model_config()
        }
    }

    fn weak_echo_box_fields() -> WrfRadarFields {
        let mut fields = uniform_box_fields();
        fields.dbz = vec![8.0f32; fields.dbz.len()];
        fields
    }

    fn ref_values(volume: &RadarVolume) -> Vec<f32> {
        let mut out = Vec::new();
        for cut in &volume.cuts {
            let MomentStorage::F32(values) = &cut.moments[&MomentType::Reflectivity].storage else {
                panic!("F32");
            };
            out.extend(values.iter().copied());
        }
        out
    }

    #[test]
    fn clutter_is_deterministic_and_zero_amount_injects_nothing() {
        let fields = weak_echo_box_fields();
        let time = DateTime::<Utc>::from_timestamp(1_700_000_000, 0).unwrap();

        // Amount 0: the clutter path is skipped, so every finite REF is the pure
        // 8 dBZ echo — nothing is injected (bit-identical to the featureless path).
        let clean = build_synthetic_volume(&fields, time, &clutter_box_config(0.0));
        for value in ref_values(&clean).iter().filter(|value| value.is_finite()) {
            assert!(
                (value - 8.0).abs() < 1e-6,
                "amount-0 REF {value} must be the pristine model echo — no clutter injected"
            );
        }

        // Amount 1: two rebuilds of the SAME frame are bit-identical (no
        // shimmer), and clutter actually changed the field.
        let full = build_synthetic_volume(&fields, time, &clutter_box_config(1.0));
        let full_again = build_synthetic_volume(&fields, time, &clutter_box_config(1.0));
        assert_eq!(
            moment_bits(&full),
            moment_bits(&full_again),
            "same frame + config must rebuild bit-identically"
        );
        assert_ne!(
            moment_bits(&clean),
            moment_bits(&full),
            "clutter at full amount must change the field"
        );

        // A DIFFERENT forecast frame (different valid time) gets a DIFFERENT
        // clutter pattern — the seed folds in the frame time.
        let other_time = DateTime::<Utc>::from_timestamp(1_700_003_600, 0).unwrap();
        let other = build_synthetic_volume(&fields, other_time, &clutter_box_config(1.0));
        assert_ne!(
            moment_bits(&full),
            moment_bits(&other),
            "distinct forecast frames must get distinct clutter"
        );
    }

    #[test]
    fn clutter_amount_increases_the_number_of_cluttered_gates() {
        let fields = weak_echo_box_fields();
        let time = DateTime::<Utc>::from_timestamp(1_700_000_000, 0).unwrap();
        let count = |intensity: f32| -> usize {
            let volume = build_synthetic_volume(&fields, time, &clutter_box_config(intensity));
            ref_values(&volume)
                .iter()
                .filter(|value| value.is_finite() && **value > 8.0 + 1e-3)
                .count()
        };
        let zero = count(0.0);
        let low = count(0.5);
        let mid = count(0.75);
        let high = count(1.0);
        assert_eq!(zero, 0, "amount 0 produces no clutter");
        assert!(low > 0, "amount 0.5 produces some clutter ({low})");
        assert!(
            mid >= low,
            "clutter is monotonic: 0.75 ≥ 0.5 ({mid} ≥ {low})"
        );
        assert!(
            high >= mid,
            "clutter is monotonic: 1.0 ≥ 0.75 ({high} ≥ {mid})"
        );
        assert!(
            high > low,
            "full amount must clutter strictly more gates than a lower amount ({high} > {low})"
        );
    }

    #[test]
    fn clutter_never_overwrites_stronger_echo() {
        let fields = uniform_box_fields(); // 40 dBZ everywhere
        let time = DateTime::<Utc>::from_timestamp(1_700_000_000, 0).unwrap();
        let clean = build_synthetic_volume(&fields, time, &clutter_box_config(0.0));
        let cluttered = build_synthetic_volume(&fields, time, &clutter_box_config(1.0));
        assert_eq!(
            moment_bits(&clean),
            moment_bits(&cluttered),
            "40 dBZ echo out-values any clutter — the field must be untouched"
        );
    }

    #[test]
    fn clutter_velocity_is_near_zero_where_wind_existed_and_blank_in_clear_air() {
        let fields = weak_echo_box_fields(); // 8 dBZ, 10 m/s east wind
        let time = DateTime::<Utc>::from_timestamp(1_700_000_000, 0).unwrap();

        // Echo present (floor 0): cluttered gates keep a (now near-zero)
        // velocity; the surviving 8 dBZ gates keep the wind projection.
        let volume = build_synthetic_volume(&fields, time, &clutter_box_config(1.0));
        let mut clutter_vel_gates = 0usize;
        let mut fast_wind_gates = 0usize;
        for cut in &volume.cuts {
            let MomentStorage::F32(refs) = &cut.moments[&MomentType::Reflectivity].storage else {
                panic!("F32");
            };
            let MomentStorage::F32(vels) = &cut.moments[&MomentType::Velocity].storage else {
                panic!("F32");
            };
            for (r, v) in refs.iter().zip(vels) {
                if !r.is_finite() {
                    continue;
                }
                if *r > 8.0 + 1e-3 {
                    assert!(v.is_finite(), "clutter gate over echo must keep a velocity");
                    assert!(
                        v.abs() <= 0.51,
                        "clutter-dominated Vr {v} must be ~0 (stationary ground)"
                    );
                    clutter_vel_gates += 1;
                } else if v.is_finite() && v.abs() > 1.0 {
                    fast_wind_gates += 1;
                }
            }
        }
        assert!(clutter_vel_gates > 0, "expected clutter-dominated gates");
        assert!(
            fast_wind_gates > 0,
            "pure-echo gates must keep the wind projection"
        );

        // Clear air (floor 30, so the 8 dBZ echo never passes): any finite REF
        // is clutter, and its velocity must be BLANK (no wind projection existed
        // to replace).
        let clear_cfg = SyntheticRadarConfig {
            ref_floor_dbz: 30.0,
            ..clutter_box_config(1.0)
        };
        let clear = build_synthetic_volume(&fields, time, &clear_cfg);
        let mut clear_clutter = 0usize;
        for cut in &clear.cuts {
            let MomentStorage::F32(refs) = &cut.moments[&MomentType::Reflectivity].storage else {
                panic!("F32");
            };
            let MomentStorage::F32(vels) = &cut.moments[&MomentType::Velocity].storage else {
                panic!("F32");
            };
            for (r, v) in refs.iter().zip(vels) {
                if r.is_finite() {
                    assert!(
                        !v.is_finite(),
                        "clear-air clutter gate must leave velocity blank (gated on vel existing)"
                    );
                    clear_clutter += 1;
                }
            }
        }
        assert!(clear_clutter > 0, "clutter must fill some clear-air gates");
    }

    fn rotated_domain_fields(
        n: usize,
        spacing: f32,
        theta_deg: f32,
        bounded: bool,
    ) -> WrfRadarFields {
        let c = (n as f32 - 1.0) / 2.0;
        let (sin_t, cos_t) = theta_deg.to_radians().sin_cos();
        let cells = n * n;
        let nz = 2;
        let mut lat = Vec::with_capacity(cells);
        let mut lon = Vec::with_capacity(cells);
        for j in 0..n {
            for i in 0..n {
                let x = (i as f32 - c) * spacing;
                let y = (j as f32 - c) * spacing;
                lon.push(-95.0 + x * cos_t - y * sin_t);
                lat.push(39.0 + x * sin_t + y * cos_t);
            }
        }
        let mut height_msl = vec![100.0f32; nz * cells];
        height_msl[cells..].fill(8000.0);
        let lut = if bounded {
            InverseLut::build_with_shape_domain_bounded(&lat, &lon, n, n).expect("bounded lut")
        } else {
            InverseLut::build_with_shape(&lat, &lon, n, n).expect("plain lut")
        };
        WrfRadarFields {
            nx: n,
            ny: n,
            nz,
            lat,
            lon,
            height_msl,
            dbz: vec![40.0f32; nz * cells],
            u: vec![10.0f32; nz * cells],
            v: vec![0.0f32; nz * cells],
            w: vec![0.0f32; nz * cells],
            terrain_m: vec![0.0f32; cells],
            property_scattering: None,
            property_table_identity: None,
            property_table_resident_bytes: 0,
            raw_property_scene: None,
            refractivity_model: None,
            polarimetric: None,
            dual_pol_status: None,
            tke_tenths_m2s2: None,
            ref_source: "test",
            dx_m: None,
            source_model_override: None,
            source_microphysics_override: None,
            source_scattering_override: None,
            source_provenance_fragment: None,
            lut,
        }
    }

    #[test]
    fn synthetic_gates_stay_nan_outside_the_true_domain_edge() {
        let n = 41usize;
        let spacing = 0.02f32; // deg → half-width 0.4° (~44 km)
        let theta = 30.0f32;
        let config = SyntheticRadarConfig {
            site_lat_deg: Some(39.0),
            site_lon_deg: Some(-95.0),
            antenna_msl_m: Some(300.0),
            elevations_deg: vec![0.5],
            azimuth_count: 360,
            gate_spacing_m: 250.0,
            max_range_m: 120_000.0,
            ..SyntheticRadarConfig::default()
        };
        let time = DateTime::<Utc>::from_timestamp(1_700_000_000, 0).unwrap();

        let half = (n as f32 - 1.0) / 2.0 * spacing;
        // The cell solver admits a two-percent edge tolerance; allow
        // another half percent for f32 geolocation rounding.
        let tol = spacing * 0.025;
        let (sin_t, cos_t) = theta.to_radians().sin_cos();

        // Worst finite-gate excursion in the domain's own (rotated) frame.
        let worst_extent = |volume: &RadarVolume| -> (f32, usize) {
            let mut worst = 0.0f32;
            let mut finite = 0usize;
            for cut in &volume.cuts {
                let grid = &cut.moments[&MomentType::Reflectivity];
                let MomentStorage::F32(values) = &grid.storage else {
                    panic!("F32");
                };
                let gate_count = grid.gate_range.gate_count;
                let spacing_m = f64::from(grid.gate_range.gate_spacing_m);
                for (row, radial) in cut.radials.iter().enumerate() {
                    let az_rad = f64::from(radial.azimuth_deg).to_radians();
                    for gate in 0..gate_count {
                        if !values[row * gate_count + gate].is_finite() {
                            continue;
                        }
                        finite += 1;
                        let ground = beam_ground_range_m(
                            gate as f64 * spacing_m,
                            f64::from(radial.elevation_deg),
                        );
                        let (glat, glon) = aeqd_inverse_km(
                            39.0,
                            -95.0,
                            ground * az_rad.sin() / 1000.0,
                            ground * az_rad.cos() / 1000.0,
                        );
                        let (dlat, dlon) = ((glat - 39.0) as f32, (glon + 95.0) as f32);
                        let x = dlon * cos_t + dlat * sin_t;
                        let y = -dlon * sin_t + dlat * cos_t;
                        worst = worst.max(x.abs().max(y.abs()));
                    }
                }
            }
            (worst, finite)
        };

        let fixed = build_synthetic_volume(
            &rotated_domain_fields(n, spacing, theta, true),
            time,
            &config,
        );
        let (worst, finite) = worst_extent(&fixed);
        assert!(
            finite > 10_000,
            "beam must cover the domain: {finite} gates"
        );
        assert!(
            worst <= half + tol,
            "finite gate {worst}° from centre exceeds the true edge ({})",
            half + tol
        );

        let unbounded = build_synthetic_volume(
            &rotated_domain_fields(n, spacing, theta, false),
            time,
            &config,
        );
        let (worst_unbounded, finite_unbounded) = worst_extent(&unbounded);
        assert!(finite_unbounded > 10_000);
        assert!(
            worst_unbounded <= half + tol,
            "unbounded seed lookup must still locate a containing model cell: {worst_unbounded}"
        );
    }

    fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
        let r = 6371.0;
        let (p1, p2) = (lat1.to_radians(), lat2.to_radians());
        let dphi = (lat2 - lat1).to_radians();
        let dlam = (lon2 - lon1).to_radians();
        let a = (dphi / 2.0).sin().powi(2) + p1.cos() * p2.cos() * (dlam / 2.0).sin().powi(2);
        2.0 * r * a.sqrt().asin()
    }

    #[test]
    fn profile_real_wrfout() {
        use std::time::Instant;
        let Some(path) = std::env::var_os("BOWECHO_WRF_RADAR_FIXTURE") else {
            return;
        };
        let path = PathBuf::from(path);
        let t0 = Instant::now();
        let file = WrfFile::open(&path).expect("open real wrfout");
        eprintln!(
            "[prof] open {:.2}s  dims {}x{}x{} nt={}",
            t0.elapsed().as_secs_f64(),
            file.nx,
            file.ny,
            file.nz,
            file.nt
        );
        let config = SyntheticRadarConfig::default();

        let tr = Instant::now();
        let fields =
            read_wrf_radar_fields(&file, 0, config.reflectivity_operator).expect("read fields");
        eprintln!(
            "[prof] read_wrf_radar_fields {:.2}s  refl_source={}",
            tr.elapsed().as_secs_f64(),
            fields.ref_source
        );

        let time = DateTime::<Utc>::from_timestamp(0, 0).unwrap();
        let tb = Instant::now();
        let volume = build_synthetic_volume(&fields, time, &config);
        eprintln!(
            "[prof] build_synthetic_volume {:.2}s  cuts={} radials={}",
            tb.elapsed().as_secs_f64(),
            volume.cuts.len(),
            volume.metadata.decoded_radial_count
        );
        eprintln!("[prof] TOTAL {:.2}s", t0.elapsed().as_secs_f64());
    }

    #[test]
    fn parallel_read_matches_sequential_fields() {
        let Some(path) = std::env::var_os("BOWECHO_WRF_RADAR_FIXTURE") else {
            return;
        };
        let path = PathBuf::from(path);
        let file = WrfFile::open(&path).expect("open real wrfout");
        let nz = file.nz;
        let cells = file.nx * file.ny;

        // Original serial read logic (verbatim from before the parallelization).
        let seq = {
            let height = read_3d(&file, "height", 0, nz * cells).unwrap();
            let (dbz, _src) =
                read_reflectivity(&file, 0, nz * cells, ReflectivityOperator::ModelNative).unwrap();
            let (u, v) = match getvar(&file, "uvmet", Some(0), &ComputeOpts::default()) {
                Ok(uvmet) if uvmet.data.len() == 2 * nz * cells => {
                    let (ue, ve) = uvmet.data.split_at(nz * cells);
                    (to_f32(ue), to_f32(ve))
                }
                _ => {
                    let ua = read_3d(&file, "ua", 0, nz * cells).unwrap();
                    let va = read_3d(&file, "va", 0, nz * cells).unwrap();
                    (ua, va)
                }
            };
            let w = read_3d(&file, "wa", 0, nz * cells).unwrap();
            (height, dbz, u, v, w)
        };

        let par = read_wrf_radar_fields(&file, 0, ReflectivityOperator::ModelNative).unwrap();

        // Bit-identical comparison (compare raw bits so NaNs must match too).
        let same = |a: &[f32], b: &[f32]| -> bool {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
        };
        assert!(same(&seq.0, &par.height_msl), "height differs");
        assert!(same(&seq.1, &par.dbz), "dbz differs");
        assert!(same(&seq.2, &par.u), "u differs");
        assert!(same(&seq.3, &par.v), "v differs");
        assert!(same(&seq.4, &par.w), "w differs");
        eprintln!(
            "[equiv] parallel read == serial read: {} elems x 5 fields bit-identical",
            par.dbz.len()
        );
    }

    #[test]
    fn selected_gate_witness_rejects_stale_geometry_config_and_operator() {
        let mut volume = irregular_observed_replay_volume();
        volume.metadata.forward_operator = Some("BowEcho WRF fixture".to_owned());
        volume.metadata.forward_operator_config = Some("config=v1".to_owned());
        let witness = SyntheticFrameWitness::from_volume(3, 0x1234, &volume);
        assert!(witness.matches(0x1234, &volume));
        assert!(!witness.matches(0x1235, &volume));

        let mut stale_geometry = volume.clone();
        stale_geometry.cuts[0].radials[0].azimuth_deg += 0.25;
        assert!(!witness.matches(0x1234, &stale_geometry));

        let mut stale_operator = volume.clone();
        stale_operator.metadata.forward_operator_config = Some("config=v2".to_owned());
        assert!(!witness.matches(0x1234, &stale_operator));

        // Source paths are deliberately outside the export-safe witness. The
        // retained descriptor owns them privately for this session.
        let mut moved_source = volume;
        moved_source.metadata.source_path = Some("private/moved/wrfout".to_owned());
        assert!(witness.matches(0x1234, &moved_source));
    }

    #[test]
    fn structured_grid_query_does_not_snap_to_a_distant_lut_seed() {
        let (nx, ny, nz) = (17usize, 17usize, 2usize);
        let cells = nx * ny;
        let lat: Vec<_> = (0..ny).flat_map(|j| (0..nx).map(move |_| j as f32 * 0.03)).collect();
        let lon: Vec<_> = (0..ny).flat_map(|_| (0..nx).map(move |i| i as f32 * 0.04)).collect();
        let u: Vec<_> = (0..nz).flat_map(|_| (0..ny).flat_map(|j| (0..nx).map(move |i| 10.0 + i as f32 + 2.0*j as f32))).collect();
        let fields = WrfRadarFields::from_model_fields(ModelRadarFields {
            nx, ny, nz, latitude_deg:lat, longitude_deg:lon,
            height_msl_m:[vec![0.0;cells],vec![10_000.0;cells]].concat(),
            reflectivity_dbz:vec![40.0;cells*nz],eastward_wind_mps:u,
            northward_wind_mps:vec![0.0;cells*nz],upward_wind_mps:vec![0.0;cells*nz],
            terrain_msl_m:vec![0.0;cells],grid_spacing_m:Some(3000.0),model_label:"analytic".to_owned(),
        }).unwrap();
        let mut max_error = 0.0f32;
        for j in 1..ny-2 {
            for i in 1..nx-2 {
                for offset in [0.1f32, 0.4, 0.7] {
                    let x = i as f32 + offset;
                    let y = j as f32 + offset;
                    let value = sample_column(&fields,cells,y*0.03,x*0.04,500.0,0.5,ReflectivitySampling::LinearZ)
                        .unwrap().expect("interior structured-grid query");
                    max_error=max_error.max((value.u-(10.0+x+2.0*y)).abs());
                }
            }
        }
        assert!(max_error<0.001,"linear field error from inverse seed: {max_error}");
    }
