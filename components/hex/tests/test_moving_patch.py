import numpy as np
import pytest
from types import SimpleNamespace
from woof.hex.moving_remap import SphericalCells, ConservativeMap, overlap_polygon, spherical_area, sphere_xyz
from woof.hex.moving_track import external_position, StormTracker


def test_tracer_remap_refuses_nonconservative_transfer_before_state_install(monkeypatch):
    # Exercise the retained unbalanced path; balanced products have their own
    # FP64 and representability gates before installation.
    monkeypatch.setenv('WOOF_HEX_MOVING_BALANCE', '0')
    from woof.hex.moving_state import remap_atmosphere

    class BrokenTracerMap:
        redistributes_vertical=True

        def conserved(self,values,density=None):
            # Density and theta transfer are valid; only tracer transfer is
            # broken. Rejection must occur before momentum or state install.
            return values.copy(),{'accounting':'global column integral'}

        def mass_intensive(self,values,density=None):
            return np.asarray(values,dtype=np.float64)*1.25

        intensive=mass_intensive

        def totals(self,values,*,source):
            return np.atleast_1d(np.sum(values,dtype=np.float64))

    vertical=SimpleNamespace(dzw=np.ones(1))
    payload=dict(vertical=vertical,scalar_names=['qv'],state=dict(
        rho=np.ones((1,2),np.float32),rho_theta=np.full((1,2),300.,np.float32),
        scalars=np.full((1,1,2),.01,np.float32)))
    with pytest.raises(ValueError,match='qv FP64 integral'):
        remap_atmosphere(payload,BrokenTracerMap(),source_mesh=None,
                        target_prepared=SimpleNamespace(vertical=vertical),config=None)


def test_moving_constructor_carries_original_pressure_and_reseals_before_backend():
    from woof.hex.moving_checkpoint import carry_physics_top_pressure
    old = float(np.float32(1139.2586669921875))
    new = float(np.nextafter(np.float32(old), np.float32(0)))
    host = dict(constructor_values={'p_top_pa': new, 'dt': 18.75},
                constructor_receipt={'p_top_pa': new, 'sealed_host_contract_audit': {'identity': 'old'}})
    payload = {'backends': [{'seam': {'identity': {'p_top_pa': old}}} for _ in range(4)]}
    sealed = []
    def seal(values):
        sealed.append(dict(values))
        return SimpleNamespace(receipt=lambda: {'identity': 'resealed'})
    selected = carry_physics_top_pressure(host, payload, sealer=seal)
    assert selected['constructor_values']['p_top_pa'] == old
    assert sealed == [selected['constructor_values']]
    assert selected['constructor_receipt']['sealed_host_contract_audit']['identity'] == 'resealed'
    assert selected['constructor_receipt']['moving_top_pressure']['target_init_derived_p_top_pa'] == new
    assert host['constructor_values']['p_top_pa'] == new
    assert host['constructor_receipt']['sealed_host_contract_audit']['identity'] == 'old'


@pytest.mark.parametrize('values', [[], [1., 2.], [0.], [-1.], [float('nan')], [float('inf')]])
def test_moving_constructor_refuses_missing_conflicting_or_invalid_boundary(values):
    from woof.hex.moving_checkpoint import carry_physics_top_pressure
    payload = {'backends': [{'seam': {'identity': {'p_top_pa': value}}} for value in values]}
    with pytest.raises(ValueError):
        carry_physics_top_pressure({}, payload)


def test_mass_flux_divergence_closes_and_respects_edge_orientation():
    from woof.hex.moving_state import mass_flux_divergence
    arrays=dict(edgesOnCell=np.array([[0],[0]]),nEdgesOnCell=np.array([1,1]),
                cellsOnEdge=np.array([[0,1]]),dvEdge=np.array([2.]),areaCell=np.array([2.,3.]))
    value=mass_flux_divergence(np.array([[3.],[5.]],dtype=np.float32),arrays)
    np.testing.assert_allclose(value,[[3.,-2.],[5.,-10./3.]])
    np.testing.assert_allclose((value*arrays['areaCell']).sum(axis=1),0.,atol=2e-15)
    arrays['cellsOnEdge']=np.array([[1,0]])
    np.testing.assert_array_equal(mass_flux_divergence(np.array([[3.],[5.]],dtype=np.float32),arrays),-value)


def test_remapped_feedback_preserves_values_with_contiguous_local_columns():
    from woof.hex.moving_physics import remap_step_carriers
    cells=octants();weights=ConservativeMap.build(cells,cells)
    source=SimpleNamespace(cell_l2g=np.arange(8),n_owned_cells=8)
    targets=[SimpleNamespace(cell_l2g=np.array([6,1,4,7]),n_owned_cells=2),
             SimpleNamespace(cell_l2g=np.array([0,2,3,5,4,7]),n_owned_cells=6)]
    values=np.arange(24,dtype=np.float32).reshape(3,8)
    spectral = dict(time_seconds=1800., draws=7, coefficients=np.arange(6.))
    payload=dict(layouts=[source],previous=[{'surface':values}],
                 stochastic=spectral,
                 forcing=[dict(rthdynten=values,rqvdynten=values+2,time_seconds=1800.)])
    result=remap_step_carriers(payload,weights,targets)
    # Spectral coefficients are mesh-independent; keep the original carrier for
    # the driver's restore instead of interpolating them onto target columns.
    assert payload['stochastic'] is spectral
    assert result['stochastic'] is spectral
    np.testing.assert_array_equal(spectral['coefficients'], np.arange(6.))
    assert spectral['time_seconds'] == 1800. and spectral['draws'] == 7
    for layout,previous,forcing in zip(targets,result['previous'],result['forcing']):
        for name,expected in [('rthdynten',values),('rqvdynten',values+2)]:
            assert forcing[name].flags.c_contiguous
            np.testing.assert_array_equal(forcing[name],expected[...,layout.cell_l2g])
        assert previous['surface'].flags.c_contiguous
        np.testing.assert_array_equal(previous['surface'],values[...,layout.cell_l2g])
        assert forcing['time_seconds']==1800.


def test_noahmp_stores_use_one_geography_compatible_column_and_keep_new_static():
    from copy import deepcopy
    from woof.hex.moving_physics import restore_remapped_backend
    cells=octants();weights=ConservativeMap.build(cells,cells)
    mesh=SimpleNamespace(attrs={'sphere_radius':6371229.},
                         arrays={key:cells.centers[:,i]*6371229. for i,key in enumerate(('xCell','yCell','zCell'))})
    arrays={'fields/xland':np.ones((1,8),np.float32),'fields/xice':np.zeros((1,8),np.float32),
            'fields/isltyp':np.arange(1,9,dtype=np.int32)[None,:],
            'fields/ivgtyp':np.ones((1,8),np.int32),
            'fields/snow':np.arange(8,dtype=np.float32)[None,:]*3,
            'fields/isnowxy':-np.arange(8,dtype=np.int32)[None,:],
            'fields/snliqxy':np.arange(24,dtype=np.float32).reshape(3,1,8)}
    source=dict(identity={},seam=dict(identity={'xice_threshold':.02},arrays=arrays,
                scalars=dict(elapsed_seconds=1800.,carriers={},kf_history_time=1800.)),adapter={})
    fresh=deepcopy(source);fresh['seam']['arrays']['fields/isltyp'][0,0]=5
    layout=SimpleNamespace(cell_l2g=np.arange(8),n_owned_cells=8)
    restored=[]
    backend=SimpleNamespace(restart_state=lambda:deepcopy(fresh),restore_restart_state=restored.append)
    receipt=restore_remapped_backend(backend,dict(backends=[source],layouts=[layout],mesh=mesh,time_seconds=1800.),
                                    weights,layout,base_dt=18.75,rate=8,target_mesh=mesh)
    # The local source column wins over a far exact soil-type match.
    ids=np.arange(8)
    for key in ('fields/snow','fields/isnowxy','fields/snliqxy'):
        np.testing.assert_array_equal(restored[0]['seam']['arrays'][key],arrays[key][...,ids])
    np.testing.assert_array_equal(restored[0]['seam']['arrays']['fields/isltyp'],fresh['seam']['arrays']['fields/isltyp'])
    assert receipt['land_state']['nearest_compatible_fallback_columns']==0
    assert receipt['step_index']==12
    fresh['seam']['arrays']['fields/ivgtyp'][0,0]=2
    receipt=restore_remapped_backend(backend,dict(backends=[source],layouts=[layout],mesh=mesh,time_seconds=1800.),
                                    weights,layout,base_dt=18.75,rate=8,target_mesh=mesh)
    assert receipt['land_state']['relaxed_category_donor_columns']=={
        'fields/xland,fields/xice':1}
    np.testing.assert_array_equal(restored[-1]['seam']['arrays']['fields/snow'],arrays['fields/snow'][...,ids])
    fresh['seam']['arrays']['fields/xland'][0,0]=2
    with pytest.raises(ValueError,match='no carried land column'):
        restore_remapped_backend(backend,dict(backends=[source],layouts=[layout],mesh=mesh,time_seconds=1800.),
                                 weights,layout,base_dt=18.75,rate=8,target_mesh=mesh)


def octants(rotation=0.):
    # Closed spherical octahedron, rotated to change every donor polygon.
    v = np.vstack((np.eye(3), -np.eye(3)))
    r = np.array([[np.cos(rotation),-np.sin(rotation),0], [np.sin(rotation),np.cos(rotation),0],[0,0,1]])
    polygons = []
    for i in (0,3):
        for j in (1,4):
            for k in (2,5):
                p = v[[i,j,k]] @ r.T
                if np.dot(np.cross(p[0],p[1]), p.sum(axis=0)) < 0:
                    p = p[::-1]
                polygons.append(p)
    polygons = np.array(polygons)
    centers = polygons.sum(axis=1); centers /= np.linalg.norm(centers,axis=1,keepdims=True)
    return SphericalCells(centers, polygons, np.full(8,3), np.full(8,np.pi/2), 1.)


def test_exact_closed_sphere_overlap_and_no_creation():
    a,b = octants(),octants(.13)
    mapping = ConservativeMap.build(a,b)
    assert mapping.receipt['raw_coverage_relative_max'] < 2e-15
    np.testing.assert_allclose(mapping.intensive(np.ones(8)), 1., atol=2e-15)
    q=np.array([0.,0.,0.,0.,1.,2.,3.,4.],dtype=np.float32)
    result, receipt=mapping.conserved(q)
    assert result.min()>=0 and result.max()<=q.max()
    assert abs(receipt['absolute_error']) <= receipt['rounding_envelope']+1e-13
    # This independently checks a signed field and a sparse nonconstant tracer.
    for field in (np.arange(8)-3.,q):
        assert abs(np.dot(mapping.intensive(field),b.area)-np.dot(field,a.area)) < 2e-14


def test_identity_and_saved_operator(tmp_path):
    a=octants(); m=ConservativeMap.build(a,a)
    x=np.arange(24,dtype=np.float32).reshape(3,8)
    np.testing.assert_array_equal(m.conserved(x)[0],x)
    path=tmp_path/'weights.npz';m.save(path)
    np.testing.assert_array_equal(ConservativeMap.load(path).conserved(x)[0],x)


def test_metric_rounding_reconciliation_retains_donors_and_refuses_insufficient_budget():
    source, target = octants(), octants(.13)
    # Equal total area, with opposite metric rounding inside a connected
    # hemisphere. One scaling pass cannot close both marginals.
    target.area[0] += 5e-10
    target.area[2] -= 5e-10
    with pytest.raises(ValueError, match='failed to converge'):
        ConservativeMap.build(source, target, reconciliation_limit=1)
    mapping = ConservativeMap.build(source, target)
    assert mapping.receipt['reconciliation_iterations'] > 1
    assert mapping.receipt['donor_area_relative_max'] <= 2e-14
    assert mapping.receipt['reconciled_relative_max'] <= 1e-10
    assert mapping.receipt['recipient_area_relative_tolerance'] <= np.finfo(np.float32).eps / 1024
    assert mapping.receipt['fp32_constant_reproduction_abs_max'] == 0.
    np.testing.assert_array_equal(mapping.intensive(np.ones(8)).astype(np.float32), np.ones(8, dtype=np.float32))
    fields = np.array([[-3., 2., 0., 9., -4., 5., 2., -1.],
                       [0., 0., 0., 0., 1., 2., 3., 4.]])
    mapped = mapping.intensive(fields)
    np.testing.assert_allclose(mapped @ target.area, fields @ source.area, rtol=0., atol=3e-14)


def test_declared_metric_rounding_keeps_fp32_constants_and_fp64_donor_integrals():
    source, target = octants(), octants()
    target.area[0] *= 1 + 5e-11
    target.area[1] -= target.area[0] - source.area[0]
    mapping = ConservativeMap.build(source, target)
    assert mapping.receipt['reconciled_relative_max'] > 1e-11
    assert mapping.receipt['donor_area_relative_max'] <= 2e-14
    np.testing.assert_array_equal(mapping.intensive(np.ones(8)).astype(np.float32), np.ones(8, dtype=np.float32))
    field = np.arange(8, dtype=np.float64) - 3.
    assert abs(mapping.intensive(field) @ target.area - field @ source.area) < 3e-14
    # A mismatch above the precision budget still refuses, even when the
    # global area agrees and every donor could individually be conserved.
    target.area[0] = source.area[0]*(1 + 5e-10)
    target.area[1] = source.area[1] - (target.area[0] - source.area[0])
    with pytest.raises(ValueError, match='failed to converge'):
        ConservativeMap.build(source, target, reconciliation_limit=32)


def test_global_fine_metric_rounding_checks_actual_polygon_coverage():
    source, target = octants(), octants(.13)
    target.area[0] += 4e-8
    target.area[2] -= 4e-8
    mapping = ConservativeMap.build(source, target)
    assert mapping.receipt['raw_coverage_relative_max'] > 1e-8
    assert mapping.receipt['geometric_coverage_relative_max'] < 1e-13
    assert mapping.receipt['donor_area_relative_max'] <= 2e-14
    np.testing.assert_array_equal(mapping.intensive(np.ones(8)).astype(np.float32),
                                  np.ones(8, dtype=np.float32))
    target.area[0] += 1e-5
    target.area[2] -= 1e-5
    with pytest.raises(ValueError, match='beyond FP32 metric rounding'):
        ConservativeMap.build(source, target)


def test_geometry_validation_batches_keep_padding_and_first_bad_cell():
    from dataclasses import replace
    cells = octants()
    count = 8200  # crosses the native validation block boundary
    ids = np.arange(count) % len(cells.area)
    vertices = np.pad(cells.vertices[ids], ((0, 0), (0, 1), (0, 0)))
    many = replace(cells, centers=cells.centers[ids], vertices=vertices,
                   counts=cells.counts[ids], area=cells.area[ids])
    assert many.validate() is many
    many.vertices[8193, 1] = many.vertices[8193, 0]
    with pytest.raises(ValueError, match='cell 8193 is not consistently'):
        many.validate()
    many.counts[0] = many.vertices.shape[1] + 1
    with pytest.raises(ValueError, match='invalid cell area or vertex count'):
        many.validate()


def test_refuse_incomplete_mesh_and_wrong_metrics():
    a,b=octants(),octants(.3)
    b.area[0]*=2
    with pytest.raises(ValueError,match='total declared areas'):
        ConservativeMap.build(a,b)
    b=octants();b.vertices[0]=b.vertices[1]
    with pytest.raises(ValueError,match='convex|coverage'):
        ConservativeMap.build(a,b)


def test_external_track_dateline_and_no_extrapolation():
    lat,lon=external_position([[0,0,179],[3600,0,-179]],1800)
    assert abs(lat)<1e-12 and abs(abs(lon)-180)<1e-12
    with pytest.raises(ValueError,match='cover'):
        external_position([[0,0,179],[3600,0,-179]],7200)


def test_tracker_selects_connected_cluster_and_area_votes():
    mesh=SimpleNamespace(arrays=dict(latCell=np.zeros(4),lonCell=np.deg2rad([0,1,2,3]),
        areaCell=np.array([1.,1.,10.,1.]),cellsOnCell=np.array([[1],[0],[3],[2]]),nEdgesOnCell=np.ones(4,dtype=int)))
    result=StormTracker(deadband_km=0).diagnose(mesh,[40,40,40,40],(0,0),interval_seconds=3600)
    assert result['cluster_cells']==2
    assert abs(result['center'][1]-(2+1/11))<1e-4


def test_accelerator_matches_independent_spherical_clipper(monkeypatch):
    from woof.hex.moving_remap import _accelerator
    native=_accelerator()
    if native is None:
        pytest.skip('CPU accelerator library not supplied')
    a,b=octants(),octants(.13)
    ids=np.arange(8,dtype=np.int64);out=np.empty(8)
    for j in range(8):
        assert native(np.ascontiguousarray(a.vertices),3,a.counts,ids,8,np.ascontiguousarray(b.vertices[j]),3,
                      np.ascontiguousarray(b.centers[j]),1.,out)==0
        expected=[spherical_area(overlap_polygon(a.vertices[i],b.vertices[j],b.centers[j])) for i in range(8)]
        np.testing.assert_allclose(out,expected,rtol=1e-14,atol=1e-15)


def test_different_cell_counts_refinement_preserves_each_level_and_species():
    source=octants()
    polygons=[]
    for a,b,c in source.vertices:
        ab,bc,ca=a+b,b+c,c+a
        ab/=np.linalg.norm(ab);bc/=np.linalg.norm(bc);ca/=np.linalg.norm(ca)
        polygons.extend(([a,ab,ca],[ab,b,bc],[ca,bc,c],[ab,bc,ca]))
    polygons=np.asarray(polygons)
    centers=polygons.sum(axis=1);centers/=np.linalg.norm(centers,axis=1,keepdims=True)
    target=SphericalCells(centers,polygons,np.full(32,3),
                          np.array([spherical_area(p) for p in polygons]),1.)
    mapping=ConservativeMap.build(source,target)
    values=np.arange(6*3*8,dtype=np.float32).reshape(6,3,8)/7
    mapped,receipt=mapping.conserved(values)
    assert mapped.shape==(6,3,32)
    np.testing.assert_allclose((mapped.astype(float)*target.area).sum(axis=-1),
                              (values.astype(float)*source.area).sum(axis=-1),rtol=1e-7)
    np.testing.assert_allclose(mapping.intensive(np.ones(8)),1.,atol=2e-15)
    source_area=source.area.astype(np.float32).astype(float)
    target_area=target.area.astype(np.float32).astype(float)
    assert source_area.sum()!=target_area.sum()
    prepared=mapping.with_physical_areas(source_area,target_area)
    mapped,receipt=prepared.conserved(values)
    error=abs((mapped.astype(float)*target_area).sum(axis=-1)-(values.astype(float)*source_area).sum(axis=-1))
    assert np.all(error<=np.asarray(receipt['rounding_envelope'])+1e-11)
    assert prepared.receipt['prepared_donor_relative_error']<2e-14
    # A non-canonical extra perturbation cannot hide behind the cast envelope.
    target_area[0]+=1e-9
    with pytest.raises(ValueError,match='global areas'):
        mapping.with_physical_areas(source_area,target_area)


def test_owner_physics_ignores_scratch_and_carries_padding_explicitly():
    from woof.hex.moving_physics import gather_owner_columns,carry_field
    layouts=[SimpleNamespace(cell_l2g=np.array([0,2,1]),n_owned_cells=2),
             SimpleNamespace(cell_l2g=np.array([1,3,0]),n_owned_cells=2)]
    values,padding=gather_owner_columns([np.array([[10.,30.,-999.,17.]]),
                                        np.array([[20.,40.,-999.,18.]])],layouts,4)
    np.testing.assert_array_equal(values,[[10,20,30,40]])
    assert padding==[17.,18.]
    weights=ConservativeMap.build(octants(),octants(.13))
    categories=np.arange(8,dtype=np.int32)
    result=carry_field(categories,weights)
    assert set(result).issubset(set(categories))
    with pytest.raises(ValueError,match='overlap'):
        gather_owner_columns([np.ones((1,3))]*2,[layouts[0],layouts[0]],4)


def test_physics_owner_cache_matches_independent_map_and_refuses_other_boundary():
    from woof.hex.moving_physics import mapped_owner_field,carry_field
    weights=ConservativeMap.build(octants(),octants(.13))
    layouts=[SimpleNamespace(cell_l2g=np.array([0,1,2,3,4]),n_owned_cells=4),
             SimpleNamespace(cell_l2g=np.array([4,5,6,7,0]),n_owned_cells=4)]
    values=np.arange(24,dtype=np.float32).reshape(3,8)/7
    categories=np.arange(8,dtype=np.int32)
    sources=[dict(seam=dict(arrays={'fields/q':values[...,layout.cell_l2g],
                                   'fields/category':categories[layout.cell_l2g]})) for layout in layouts]
    payload=dict(layouts=layouts,backends=sources);cache={}
    mapped,padding,category=mapped_owner_field(payload,weights,'fields/q',cache)
    np.testing.assert_array_equal(mapped,carry_field(values,weights))
    assert mapped_owner_field(payload,weights,'fields/q',cache)[0] is mapped
    assert not category
    mapped,_,category=mapped_owner_field(payload,weights,'fields/category',cache)
    np.testing.assert_array_equal(mapped,carry_field(categories,weights))
    assert category
    with pytest.raises(ValueError,match='another remap'):
        mapped_owner_field(dict(payload),weights,'fields/q',cache)


def test_physics_horizontal_vector_uses_earth_basis_across_dateline():
    from woof.hex.moving_physics import carry_horizontal_vector,local_vector_bases
    source,target=octants(),octants(.13)
    weights=ConservativeMap.build(source,target)
    def mesh(cells):
        return SimpleNamespace(arrays=dict(latCell=np.arcsin(cells.centers[:,2]),
            lonCell=np.arctan2(cells.centers[:,1],cells.centers[:,0])))
    old,new=mesh(source),mesh(target)
    u=np.arange(8,dtype=np.float32)-3;v=np.arange(8,dtype=np.float32)/3
    east,north=local_vector_bases(old);te,tn=local_vector_bases(new)
    cartesian=u[:,None]*east+v[:,None]*north
    expected=np.stack([weights.intensive(cartesian[:,axis]) for axis in range(3)],axis=-1)
    mapped_u,mapped_v=carry_horizontal_vector(u,v,weights,old,new)
    np.testing.assert_array_equal(mapped_u,np.sum(expected*te,axis=-1).astype(np.float32))
    np.testing.assert_array_equal(mapped_v,np.sum(expected*tn,axis=-1).astype(np.float32))
    identity=ConservativeMap.build(source,source)
    iu,iv=carry_horizontal_vector(u,v,identity,old,old)
    np.testing.assert_array_equal(iu,u);np.testing.assert_allclose(iv,v,atol=1e-15)


def test_physics_vector_preserves_verified_pseudo_y_edge_duplicates():
    from woof.hex.moving_physics import carry_horizontal_vector
    cells=octants();weights=ConservativeMap.build(cells,cells)
    mesh=SimpleNamespace(arrays=dict(latCell=np.arcsin(cells.centers[:,2]),
        lonCell=np.arctan2(cells.centers[:,1],cells.centers[:,0])))
    u=np.arange(16,dtype=np.float32).reshape(2,1,8)
    v=np.repeat((u+1),2,axis=1)
    mu,mv=carry_horizontal_vector(u,v,weights,mesh,mesh)
    np.testing.assert_array_equal(mu,u);np.testing.assert_array_equal(mv,v)
    v[0,1,0]+=1
    with pytest.raises(ValueError,match='y-edge copies differ'):
        carry_horizontal_vector(u,v,weights,mesh,mesh)


def test_rebuilt_column_pbl_faces_follow_new_local_order_and_close_periodically():
    from woof.hex.moving_physics import coupled_column_faces
    du=np.array([[[1.,3.,8.]]],dtype=np.float32);dv=du+2
    ru,rv=coupled_column_faces(du,dv)
    np.testing.assert_array_equal(ru,[[[4.5,2.,5.5,4.5]]])
    np.testing.assert_array_equal(rv,np.repeat(dv,2,axis=1))
    np.testing.assert_array_equal(ru[...,-1],ru[...,0])


def test_patch_rotation_maps_center_and_preserves_sphere():
    from woof.hex.moving_mesh import rotation_between
    start,end=(43.,179.),(41.,-178.)
    rotation=rotation_between(start,end)
    np.testing.assert_allclose(rotation@rotation.T,np.eye(3),atol=4e-16)
    np.testing.assert_allclose(rotation@sphere_xyz(*np.deg2rad(start)),
                              sphere_xyz(*np.deg2rad(end)),atol=3e-16)
    with pytest.raises(ValueError,match='antipodal'):
        rotation_between((0,0),(0,180))


def test_generator_steradians_identify_prepared_physical_fp32_metrics():
    from woof.hex.moving_checkpoint import identify_source_geometry
    source=octants();radius=6371229.
    xyz=source.centers*radius
    mesh=SimpleNamespace(attrs={'sphere_radius':radius},arrays=dict(
        areaCell=(source.area*radius**2).astype(np.float32),
        xCell=xyz[:,0],yCell=xyz[:,1],zCell=xyz[:,2]))
    assert identify_source_geometry(mesh,source,radius)==(radius,radius**2)
    with pytest.raises(ValueError,match='earth radii'):
        identify_source_geometry(mesh,source,6371220.)
    mesh.arrays['areaCell'][0]=np.nextafter(mesh.arrays['areaCell'][0],np.float32(np.inf))
    with pytest.raises(ValueError,match='identify'):
        identify_source_geometry(mesh,source,radius)


def test_conservation_uses_active_fp32_geometry_not_only_generator_areas():
    a,b=octants(),octants(.13)
    mapping=ConservativeMap.build(a,b)
    # Deliberately unequal per-cell cast perturbations, with equal global area.
    rounded=a.area.astype(np.float32)
    rounded[::2]=np.nextafter(rounded[::2],np.float32(np.inf))
    rounded[1::2]=np.nextafter(rounded[1::2],np.float32(-np.inf))
    rounded=rounded.astype(float)
    mapping=mapping.with_physical_areas(rounded,rounded)
    values=np.arange(8,dtype=np.float32)/3
    mapped,receipt=mapping.conserved(values)
    assert abs(np.dot(values.astype(float),rounded)-np.dot(mapped.astype(float),rounded))<=receipt['rounding_envelope']+1e-12
    assert mapping.receipt['prepared_donor_relative_error']<2e-14
    with pytest.raises(ValueError,match='global areas'):
        mapping.with_physical_areas(rounded,rounded+1e-8)


def test_process_overlap_matches_serial_words_and_cleans_shared_geometry(monkeypatch, tmp_path):
    import tempfile
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    monkeypatch.setenv('WOOF_HEX_OVERLAP_THREADS', '2')
    monkeypatch.delenv('WOOF_HEX_SPHERICAL_OVERLAP_LIB', raising=False)
    source, target = octants(), octants(.13)
    monkeypatch.setenv('WOOF_HEX_OVERLAP_EXECUTOR', 'serial')
    reference = ConservativeMap.build(source, target)
    monkeypatch.setenv('WOOF_HEX_OVERLAP_EXECUTOR', 'process')
    candidate = ConservativeMap.build(source, target)
    for name in ('data', 'indices', 'indptr'):
        assert getattr(candidate.overlap, name).tobytes() == getattr(reference.overlap, name).tobytes()
    fields = np.arange(24, dtype=np.float32).reshape(3, 8)
    assert candidate.conserved(fields)[0].tobytes() == reference.conserved(fields)[0].tobytes()
    assert candidate.receipt['candidate_pairs'] == reference.receipt['candidate_pairs']
    receipt = candidate.receipt['overlap_execution']
    assert receipt['executor'] == 'process' and receipt['workers'] == 2
    assert receipt['worker_cpu_seconds'] > 0
    assert not list(tmp_path.glob('hex-overlap-*'))


def test_process_overlap_refusal_cleans_workers_and_shared_geometry(monkeypatch, tmp_path):
    import tempfile
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    monkeypatch.setenv('WOOF_HEX_OVERLAP_THREADS', '2')
    monkeypatch.setenv('WOOF_HEX_OVERLAP_EXECUTOR', 'process')
    monkeypatch.setenv('WOOF_HEX_SPHERICAL_OVERLAP_LIB', str(tmp_path / 'missing.so'))
    from concurrent.futures.process import BrokenProcessPool
    with pytest.raises(BrokenProcessPool):
        ConservativeMap.build(octants(), octants(.13))
    assert not list(tmp_path.glob('hex-overlap-*'))


def test_gil_enabled_operator_automatically_uses_processes(monkeypatch):
    import sys
    monkeypatch.setattr(sys, '_is_gil_enabled', lambda: True, raising=False)
    monkeypatch.delenv('WOOF_HEX_OVERLAP_EXECUTOR', raising=False)
    monkeypatch.delenv('WOOF_HEX_SPHERICAL_OVERLAP_LIB', raising=False)
    monkeypatch.setenv('WOOF_HEX_OVERLAP_THREADS', '2')
    cells = octants()
    polygons = cells.vertices
    for _ in range(2):
        split = []
        for a, b, c in polygons:
            ab = (a+b)/np.linalg.norm(a+b)
            bc = (b+c)/np.linalg.norm(b+c)
            ca = (c+a)/np.linalg.norm(c+a)
            split.extend([[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]])
        polygons = np.asarray(split)
    centers = polygons.sum(axis=1)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    cells = SphericalCells(centers, polygons, np.full(len(centers), 3),
                           np.array([spherical_area(p) for p in polygons]), 1.)
    mapping = ConservativeMap.build(cells, cells)
    assert mapping.receipt['overlap_execution']['executor'] == 'process'
    field = np.arange(len(centers), dtype=np.float32)
    assert mapping.conserved(field)[0].tobytes() == field.tobytes()


def _write_operator_cli_grid(tmp_path):
    from netCDF4 import Dataset
    cells = octants()
    grid = tmp_path / 'grid.nc'
    with Dataset(grid, 'w') as ds:
        ds.createDimension('nCells', 8)
        ds.createDimension('nVertices', 24)
        ds.createDimension('maxEdges', 3)
        ds.sphere_radius = 1.
        for k, axis in enumerate('xyz'):
            ds.createVariable(axis+'Cell', 'f8', ('nCells',))[:] = cells.centers[:, k]
            ds.createVariable(axis+'Vertex', 'f8', ('nVertices',))[:] = cells.vertices[:, :, k].ravel()
        ds.createVariable('verticesOnCell', 'i8', ('nCells', 'maxEdges'))[:] = np.arange(24).reshape(8, 3)+1
        ds.createVariable('nEdgesOnCell', 'i8', ('nCells',))[:] = cells.counts
        ds.createVariable('areaCell', 'f8', ('nCells',))[:] = cells.area
    return grid


@pytest.mark.parametrize("inherited_native_threads", ["1", "8"])
def test_operator_cli_keeps_worker_timing_when_archive_is_retired(tmp_path, inherited_native_threads):
    import json
    import os
    import subprocess
    import sys
    grid = _write_operator_cli_grid(tmp_path)
    out = tmp_path / 'operator.npz'
    env = dict(os.environ, WOOF_HEX_OVERLAP_THREADS='2', WOOF_HEX_OVERLAP_EXECUTOR='process')
    env.pop('WOOF_HEX_SPHERICAL_OVERLAP_LIB', None)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[name] = inherited_native_threads
    # Observe the spawned workers after the real initializer imports and opens
    # its geometry. Coordinator thread counts include its inherited BLAS pool
    # and transient executor teardown threads; neither measures worker pools.
    (tmp_path / 'operator_worker_probe.py').write_text('''
import json, os
from pathlib import Path
from woof.hex.moving_remap import _init_overlap_worker

def initialize(directory, radius):
    _init_overlap_worker(directory, radius)
    report = {
        "native_limits": [os.environ[k] for k in
                          ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")],
        "threads": len(os.listdir("/proc/self/task")) if os.path.isdir("/proc/self/task") else None,
    }
    (Path(os.environ["HEX_TEST_WORKER_REPORT_DIR"]) / f"worker-{os.getpid()}.json").write_text(json.dumps(report))
''')
    env['HEX_TEST_WORKER_REPORT_DIR'] = str(tmp_path)
    env['PYTHONPATH'] = str(tmp_path) + os.pathsep + env.get('PYTHONPATH', '')
    probe = ('import sys, os, json; from woof.hex.cli import main; '
             'from woof.hex import moving_remap; '
             'from operator_worker_probe import initialize; '
             'moving_remap._init_overlap_worker = initialize; '
             'rc = main(sys.argv[1:]); '
             'print(json.dumps({"native_limits": [os.environ[k] for k in '
             '("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")]}), file=sys.stderr); '
             'sys.exit(rc)')
    run = subprocess.run([sys.executable, '-c', probe, 'moving', 'operator',
                          str(grid), str(grid), str(out)], env=env, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    native = json.loads(run.stderr.splitlines()[-1])
    assert native['native_limits'] == ['1', '1', '1']
    worker_reports = [json.loads(p.read_text()) for p in tmp_path.glob('worker-*.json')]
    assert len(worker_reports) == 2
    for worker in worker_reports:
        assert worker['native_limits'] == ['1', '1', '1']
        assert worker['threads'] in (None, 1)
    receipt = json.loads(run.stdout)
    operator = ConservativeMap.load(out)
    assert receipt['overlap_execution'] == operator.receipt['overlap_execution']
    assert receipt['overlap_execution']['executor'] == 'process'
    assert receipt['overlap_execution']['workers'] == 2
    assert receipt['overlap_execution']['worker_cpu_seconds'] > 0

    assert [r['stage'] for r in receipt['stages']] == [
        'read-operator-geometry', 'build-conservative-operator', 'save-conservative-operator']
    assert all(r['status'] == 'done' and r['started_utc'] and r['ended_utc']
               and r['wall_seconds'] >= 0 and r['parallelism']['device'] == 'CPU'
               for r in receipt['stages'])
    field = np.arange(8, dtype=np.float32)
    assert operator.conserved(field)[0].tobytes() == field.tobytes()
    out.unlink()
    assert receipt['overlap_execution']['worker_cpu_seconds'] > 0


@pytest.mark.skipif(__import__('os').name != 'posix', reason='controller uses POSIX process groups')
@pytest.mark.parametrize('cancel_stage', ['overlap', 'save'])
def test_cancelled_operator_cli_retires_shared_geometry(tmp_path, cancel_stage):
    import os
    import select
    import subprocess
    import sys
    from woof.hex.moving_run import start_child, stop_process_tree

    grid = _write_operator_cli_grid(tmp_path)
    # Pause at a real worker result or after saving the unpublished archive.
    # Both stages must retire scratch on the controller's cancellation.
    probe = '''
import sys, time, os
from contextlib import contextmanager
from woof.hex import moving_remap
from woof.hex.cli import main
original = moving_remap._overlap_executor
@contextmanager
def paused(*args):
    with original(*args) as execute:
        def results(bounds):
            for index, result in enumerate(execute(bounds)):
                if index == 0:
                    print('OVERLAP_READY', flush=True)
                    time.sleep(60)
                yield result
        yield results
if os.environ['HEX_TEST_CANCEL_STAGE'] == 'overlap':
    moving_remap._overlap_executor = paused
else:
    original_save = moving_remap.ConservativeMap.save
    def paused_save(self, path):
        original_save(self, path)
        print('OVERLAP_READY', flush=True)
        time.sleep(60)
    moving_remap.ConservativeMap.save = paused_save
sys.exit(main(sys.argv[1:]))
'''
    env = dict(os.environ, TMPDIR=str(tmp_path), HEX_TEST_CANCEL_STAGE=cancel_stage,
               WOOF_HEX_OVERLAP_THREADS='2',
               WOOF_HEX_OVERLAP_EXECUTOR='process', OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env.pop('WOOF_HEX_SPHERICAL_OVERLAP_LIB', None)
    proc = start_child([sys.executable, '-c', probe, 'moving', 'operator',
                       str(grid), str(grid), str(tmp_path / 'operator.npz')],
                      stdout=subprocess.PIPE, env=env)
    try:
        assert select.select([proc.stdout], [], [], 15)[0], 'operator did not start'
        # start_child merges progress stderr into this lifecycle log.
        # The public CLI stdout remains one JSON document (checked above).
        for _ in range(32):
            line = proc.stdout.readline().strip()
            if line == b'OVERLAP_READY':
                break
            assert line.startswith(b'{'), line
        else:
            pytest.fail('operator did not reach the cancellation point')
        scratch_pattern = 'hex-overlap-*' if cancel_stage == 'overlap' else '*.partial.npz'
        assert list(tmp_path.glob(scratch_pattern))
        stop_process_tree(proc, grace_seconds=5)
        assert not list(tmp_path.glob('hex-overlap-*'))
        assert not list(tmp_path.glob('*.partial.npz'))
        assert proc.returncode == 143
        assert not (tmp_path / 'operator.npz').exists()
    finally:
        if proc.poll() is None:
            stop_process_tree(proc, grace_seconds=1)
        proc.stdout.close()


def test_a_class_without_ocean_columns_carries_no_sea_ice_and_still_moves():
    """Owner run run_1227117932ed36db: 4 LTS classes, fields/ice_* only in the 2 with water cells; the move stopped."""
    from copy import deepcopy
    from woof.hex.moving_physics import restore_remapped_backend
    from woof.hex.moving_seaice import SEAICE_FIELDS
    one=lambda v: np.full((1,8),v,np.float32)
    base={'fields/xland':one(1),'fields/xice':one(0),'fields/lakemask':one(0),'fields/tsk':one(280)}
    wet=dict(identity={},seam=dict(identity={'n_columns':8},arrays=dict(base,**{k:one(1) for k in SEAICE_FIELDS}),
             scalars={}),adapter={})
    dry=deepcopy(wet)
    for key in SEAICE_FIELDS:
        del dry['seam']['arrays'][key]
    backend=SimpleNamespace(restart_state=lambda:deepcopy(wet),restore_restart_state=lambda s:None)
    payload=dict(backends=[wet,dry],layouts=[None,None],mesh=None,time_seconds=1800.)
    cache={}
    try:
        restore_remapped_backend(backend,payload,None,SimpleNamespace(cell_l2g=np.arange(8)),base_dt=18.75,rate=1,
                                 mapped_cache=cache)
    except ValueError as error:
        assert 'different persistent fields' not in str(error) and 'sea-ice' not in str(error)
    except Exception:
        pass   # the stand-in payload stops later; the persistent-field check is what this test holds
    completed=cache['ice_complete_payload']['backends'][1]['seam']['arrays']
    np.testing.assert_array_equal(completed['fields/ice_thickness'],np.zeros((1,8),np.float32))
    other=deepcopy(wet);del other['seam']['arrays']['fields/xland']
    with pytest.raises(ValueError,match='different persistent fields: fields/xland'):
        restore_remapped_backend(backend,dict(payload,backends=[wet,other]),None,SimpleNamespace(cell_l2g=np.arange(8)),
                                 base_dt=18.75,rate=1)


def test_a_column_seam_with_an_empty_threshold_takes_the_fractional_rule():
    """run_f3e336fa8500473d: the RUC column seam seals xice_threshold=None; the first move died in float(None)."""
    from woof.hex.moving_physics import sea_ice_threshold
    assert sea_ice_threshold({'xice_threshold': None}) == pytest.approx(0.02)
    assert sea_ice_threshold({'xice_threshold': None, 'fractional_seaice': 0}) == 0.5
    assert sea_ice_threshold({'xice_threshold': 0.3}) == pytest.approx(0.3)


def test_a_column_seam_surface_or_soil_field_maps_across_cells_only():
    """run_2bfb0e1e82195f53: a seam/ field with 9 (soil) or 1 (surface) leading levels hit the height map and stopped."""
    from woof.hex.moving_physics import field_intensive
    class Weights:
        source_dzw = np.ones(55)
        source_area = np.ones(4)
        def intensive(self, values):
            return np.asarray(values, dtype=np.float64) * 2
        def layer_intensive(self, values):
            raise ValueError('height map layer field has invalid source dimensions')
    soil = np.ones((9, 4))
    np.testing.assert_array_equal(field_intensive(soil, Weights(), 'seam/smois'), soil * 2)
    with pytest.raises(ValueError, match='invalid source dimensions'):
        field_intensive(np.ones((55, 4)), Weights(), 'seam/qke')


def test_blended_surface_optics_come_with_the_new_members_ice():
    """A check of run_2bfb0e1e82195f53: carried albedo against the new member's xice gave RUC an ice albedo above 1."""
    from woof.hex.moving_physics import STATIC
    assert {'fields/xice', 'fields/albedo', 'fields/emiss', 'fields/xicem'} <= STATIC



def test_rkind_prepared_areas_are_admitted_through_the_generator_geometry():
    # G03 (2026-11-02): a storm-following region move after an RRFS-style init was refused with
    # "prepared physical areas differ beyond FP32 metric rounding" (metric error 1.32e-7). The
    # prepared areaCell is the RKIND product fp32(unit area) * fp32(R^2) (rw_mpas_init_v831), which
    # can sit more than one FP32 epsilon from the float64 generator area times R^2.
    from scipy.sparse import diags
    rng = np.random.default_rng(20261102)
    unit = rng.uniform(1.5e-7, 3.5e-7, 4000)          # 3 km cells on the unit sphere
    scale = 6371229.0 ** 2                             # MPAS sphere_radius squared
    prepared_area = (unit.astype(np.float32) * np.float32(scale)).astype(float)
    assert float(np.max(abs(prepared_area / (unit * scale) - 1))) > np.finfo(np.float32).eps
    scaled = ConservativeMap(diags(unit * scale, format="csr"), unit * scale, unit * scale, {})
    with pytest.raises(ValueError, match="FP32 metric rounding"):
        scaled.with_physical_areas(prepared_area, prepared_area)
    prepared = scaled.with_physical_areas(prepared_area, prepared_area, generator_areas=(unit, unit),
                                          generator_area_scale=scale)
    values = (np.arange(len(unit)) % 97).astype(np.float32) / 3
    mapped, receipt = prepared.conserved(values)
    np.testing.assert_array_equal(mapped, values)
    # Only the exact RKIND product is admitted: two ULPs more on the worst donor is refused again.
    worst = int(np.argmax(abs(prepared_area / (unit * scale) - 1)))
    bent = prepared_area.copy()
    away = np.float32(np.inf) if bent[worst] > unit[worst] * scale else np.float32(-np.inf)
    bent[worst] = float(np.nextafter(np.nextafter(np.float32(bent[worst]), away), away))
    with pytest.raises(ValueError):
        scaled.with_physical_areas(bent, prepared_area, generator_areas=(unit, unit), generator_area_scale=scale)
    # A generator that is not the admitted geometry is refused by name.
    with pytest.raises(ValueError, match="generator differs"):
        scaled.with_physical_areas(prepared_area, prepared_area, generator_areas=(unit * 1.001, unit),
                                   generator_area_scale=scale)
