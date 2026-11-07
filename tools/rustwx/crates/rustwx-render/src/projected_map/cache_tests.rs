use super::*;

#[test]
fn repeated_product_mesh_builds_geometry_once_without_changing_it() {
    let lat = [10.0, 10.0, 11.0, 11.0];
    let lon = [-2.0, -1.0, -2.0, -1.0];
    let options = ProjectedMapBuildOptions::full_domain(1.4)
        .with_projection(ProjectionSpec::Geographic)
        .without_basemap();
    let expected = build_projected_map_uncached(&lat, &lon, &options).unwrap();
    let mut cache = ProjectedMapCache {
        entries: Vec::new(),
        bytes: 0,
    };
    let first = cache
        .get_or_build(&lat, &lon, &options, 1_000_000, || {
            build_projected_map_uncached(&lat, &lon, &options)
        })
        .unwrap();
    assert_eq!(*first, expected);
    for _ in 0..64 {
        let reused = cache
            .get_or_build(&lat, &lon, &options, 1_000_000, || {
                panic!("one geometry build per product would stall the live series")
            })
            .unwrap();
        assert_eq!(*reused, expected);
    }
}

#[test]
fn changed_mesh_and_map_options_invalidate_cached_geometry() {
    let lat = [10.0, 10.0, 11.0, 11.0];
    let lon = [-2.0, -1.0, -2.0, -1.0];
    let options = ProjectedMapBuildOptions::full_domain(1.4)
        .with_projection(ProjectionSpec::Geographic)
        .without_basemap();
    let mut cache = ProjectedMapCache {
        entries: Vec::new(),
        bytes: 0,
    };
    cache
        .get_or_build(&lat, &lon, &options, 1_000_000, || {
            build_projected_map_uncached(&lat, &lon, &options)
        })
        .unwrap();
    let mut variants = Vec::new();
    let mut changed = options.clone();
    changed.domain.target_aspect_ratio = 1.8;
    variants.push(changed);
    let mut changed = options.clone();
    changed.domain.pad_fraction = 0.3;
    variants.push(changed);
    let mut changed = options.clone();
    changed.domain.reference_latitude_deg = Some(35.0);
    variants.push(changed);
    variants.push(options.clone().with_natural_frame_aspect());
    variants.push(options.clone().with_projection(ProjectionSpec::Mercator {
        latitude_of_true_scale_deg: 0.0,
        central_meridian_deg: 0.0,
    }));
    variants.push(
        options
            .clone()
            .with_geographic_grid_intersection_frame((-2.0, -1.0, 10.0, 11.0)),
    );
    for changed in variants {
        let calls = std::cell::Cell::new(0);
        let actual = cache
            .get_or_build(&lat, &lon, &changed, 1_000_000, || {
                calls.set(calls.get() + 1);
                build_projected_map_uncached(&lat, &lon, &changed)
            })
            .unwrap();
        assert_eq!(calls.get(), 1);
        assert_eq!(
            *actual,
            build_projected_map_uncached(&lat, &lon, &changed).unwrap()
        );
    }
    let moved = [10.1, 10.0, 11.0, 11.0];
    let calls = std::cell::Cell::new(0);
    let actual = cache
        .get_or_build(&moved, &lon, &options, 1_000_000, || {
            calls.set(calls.get() + 1);
            build_projected_map_uncached(&moved, &lon, &options)
        })
        .unwrap();
    assert_eq!(calls.get(), 1);
    assert_eq!(
        *actual,
        build_projected_map_uncached(&moved, &lon, &options).unwrap()
    );
    assert!(cache.entries.len() <= 4);
}

#[test]
fn geometry_cache_does_not_retain_oversized_grids() {
    let lat = [10.0, 10.0, 11.0, 11.0];
    let lon = [-2.0, -1.0, -2.0, -1.0];
    let options = ProjectedMapBuildOptions::full_domain(1.4).without_basemap();
    let mut cache = ProjectedMapCache {
        entries: Vec::new(),
        bytes: 0,
    };
    for _ in 0..2 {
        cache
            .get_or_build(&lat, &lon, &options, 1, || {
                build_projected_map_uncached(&lat, &lon, &options)
            })
            .unwrap();
        assert_eq!(cache.bytes, 0);
        assert!(cache.entries.is_empty());
    }
}
