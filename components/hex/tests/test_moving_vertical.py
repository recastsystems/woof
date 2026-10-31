import numpy as np
import pytest
from scipy.sparse import csr_matrix
from woof.hex.moving_remap import ConservativeMap
from woof.hex.moving_vertical import HeightConservativeMap


def test_positive_column_roundoff_keeps_endpoints_and_rejects_unbounded_changes():
    from woof.hex.moving_vertical import positive_cumulative_roundoff
    values=np.array([[0.,0.],[1.,0.],[1.+2*np.finfo(float).eps,0.],[1.,0.]])
    fixed,receipt=positive_cumulative_roundoff(values)
    assert np.all(np.diff(fixed,axis=0)>=0)
    np.testing.assert_array_equal(fixed[[0,-1]],values[[0,-1]])
    assert receipt['changed_interfaces']==1
    values[2,0]=1.00001
    with pytest.raises(ValueError,match='FP64 arithmetic allowance'):
        positive_cumulative_roundoff(values)


@pytest.mark.parametrize('absolute_height', [False, True])
def test_sparse_water_remap_is_nonnegative_and_conserves_columns(absolute_height):
    rng=np.random.default_rng(20220613)
    n=40
    z=np.linspace(0,30000,56)[:,None]*np.ones((1,n))
    target=z+np.sin(np.linspace(0,np.pi,56))[:,None]*rng.uniform(-100,100,n)
    weights=mapper(z,target,source_dzw=np.diff(z[:,0]),target_dzw=np.diff(z[:,0]),
                   endpoint_alignment=not absolute_height)
    water=np.zeros((55,n),np.float32)
    water[10:25]=rng.uniform(0,1e-3,(15,n))
    out,receipt=weights.conserved(water)
    assert np.all(out>=0)
    np.testing.assert_allclose(np.einsum('lc,l->c',out,weights.target_dzw),
                               np.einsum('lc,l->c',water,weights.source_dzw),rtol=5e-8)


@pytest.mark.parametrize('bottom', [-2., .25])
def test_absolute_positive_roundoff_preserves_shifted_endpoints(bottom):
    from woof.hex.moving_vertical import positive_cumulative_roundoff
    values=np.array([[bottom],[1.],[1.+2*np.finfo(float).eps],[1.]])
    fixed,receipt=positive_cumulative_roundoff(values,absolute_height=True)
    assert np.all(np.diff(fixed,axis=0)>=0)
    np.testing.assert_array_equal(fixed[[0,-1]],values[[0,-1]])
    assert receipt['changed_interfaces']==1
    values[2,0]=1.00001
    with pytest.raises(ValueError,match='FP64 arithmetic allowance'):
        positive_cumulative_roundoff(values,absolute_height=True)


def mapper(source,target,overlap=None,source_dzw=None,target_dzw=None,**options):
    source,target=np.asarray(source,float),np.asarray(target,float)
    if overlap is None:overlap=np.eye(source.shape[1])
    horizontal=ConservativeMap(csr_matrix(overlap),np.sum(overlap,axis=0),np.sum(overlap,axis=1),{})
    return HeightConservativeMap(horizontal,source,target,
        np.ones(len(source)-1) if source_dzw is None else source_dzw,
        np.ones(len(target)-1) if target_dzw is None else target_dzw,**options)


def test_parallel_subground_extension_keeps_previous_bottom_layer_words():
    from woof.hex.moving_balance import extend_below_ground
    rng=np.random.default_rng(1030)
    ns=29
    target=np.linspace(0.,30000.,56)[:,None]*np.ones((1,ns))
    source=target+(1.-target/30000.)*rng.uniform(100.,800.,ns)[None,:]
    w=mapper(source,target,source_dzw=np.diff(source[:,0]),target_dzw=np.diff(target[:,0]),
             endpoint_alignment=False)
    old=dict(rho=rng.uniform(.5,1.5,(55,ns)).astype(np.float32),
             rho_theta=rng.uniform(200.,400.,(55,ns)).astype(np.float32),
             exner=rng.uniform(.7,1.,(55,ns)).astype(np.float32),
             scalars=rng.uniform(0.,.001,(3,55,ns)).astype(np.float32))
    rho=old['rho'].astype(float)
    theta=(old['rho_theta'].astype(float)/rho)[0]
    exner=old['exner'].astype(float)[0]
    loading=(1.+sum((old['scalars'][i].astype(float) for i in range(3)),np.zeros_like(rho)))[0]
    c=w.cols
    rate=9.80616*loading[c]/(1004.5*theta[c])
    centre=w.source_centres[0,c]
    ground=w.source_zgrid[0,c]
    def pressure(z):
        return 1e5*(exner[c]+rate*(centre-z))**(1004.5/287.)
    expected=np.array([(pressure(np.minimum(height,ground))-pressure(ground))/(9.80616*loading[c])
                       for height in w.sample_height])
    extend_below_ground(w,old,[0,1,2])
    assert w.below_ground.tobytes()==expected.tobytes()


def test_height_redistribution_preserves_column_totals_and_exact_constant_density():
    weights=mapper([[0],[2],[5],[10]],[[0],[1],[7],[10]],source_dzw=[2,3,5],target_dzw=[1,6,3])
    out,budget=weights.conserved(np.full((3,1),2.,dtype=np.float32))
    np.testing.assert_array_equal(out,np.full((3,1),2.))
    assert budget['before']==budget['after']==[20.]


def test_height_endpoints_preserve_mass_when_ground_and_mesh_cell_count_change():
    weights=mapper([[0,100],[2,104],[5,110]],[[50],[53],[108]],overlap=[[2.,3.]],
                   source_dzw=[2,3],target_dzw=[2,3])
    value=np.array([[2,4],[3,1]],np.float32)
    out,budget=weights.conserved(value)
    assert np.all(out>=0)
    assert abs(budget['absolute_error'][0])<=budget['rounding_envelope'][0]+abs(budget['fp64_error'][0])+1e-12
    np.testing.assert_allclose(weights.totals(out,source=False),weights.totals(value,source=True),rtol=1e-7)


def test_limited_vertical_reconstruction_preserves_signed_momentum_and_sparse_tracer():
    weights=mapper([[0],[1],[3],[6],[10]],[[0],[2],[4],[7],[10]])
    for value in (np.array([[-3],[2],[4],[-1]],np.float32),np.array([[0],[1],[0],[0]],np.float32)):
        out,budget=weights.conserved(value)
        np.testing.assert_allclose(weights.totals(out,source=False),weights.totals(value,source=True),atol=2e-7)
        if np.all(value>=0):assert np.all(out>=0)


def test_height_map_refuses_crossed_vertical_surfaces():
    with pytest.raises(ValueError,match='increasing physical interfaces'):
        mapper([[0],[2],[1]],[[0],[1],[2]])


def test_height_remap_refuses_broken_donor_closure_before_fp32_cast():
    weights=mapper([[0],[1],[2]],[[0],[1],[2]])
    # A corrupted loaded stencil must not license its own mass loss by
    # adding the observed FP64 mapping error to the cast allowance.
    weights.areas*=1.25
    with pytest.raises(ValueError,match='FP64 integral'):
        weights.conserved(np.ones((2,1),dtype=np.float32))


def test_interface_velocity_uses_the_same_aligned_physical_heights():
    weights=mapper([[0],[2],[5],[10]],[[100],[101],[107],[110]])
    result=weights.interface_intensive(np.array([[0],[2],[5],[10]],float))
    np.testing.assert_allclose(result[:,0],[0,1,7,10])


def test_absolute_height_reports_terrain_domain_correction_and_conserves_positive_product():
    weights=mapper([[0],[2],[5],[10]],[[1],[2],[8],[10]],source_dzw=[2,3,5],
                   target_dzw=[1,6,2],endpoint_alignment=False)
    result,budget=weights.conserved(np.full((3,1),2.,np.float32))
    correction=budget['terrain_domain_reconciliation']
    assert correction['source_total']==20.
    assert correction['absolute_height_total_before_reconciliation']==18.
    np.testing.assert_allclose(result,2*20/18,rtol=1e-7)
    assert abs(budget['absolute_error'][0])<=budget['rounding_envelope'][0]+1e-12


def test_absolute_height_preserves_zero_global_signed_momentum_without_dividing_by_zero():
    weights=mapper([[0],[1],[3]],[[.5],[2],[3]],endpoint_alignment=False)
    result,budget=weights.conserved(np.array([[1],[-1]],np.float32))
    assert budget['before']==[0.]
    np.testing.assert_allclose(weights.totals(result,source=False),0.,atol=1e-7)
    assert budget['terrain_domain_reconciliation']['policy']=='uniform signed computational-density correction'


def test_absolute_height_refuses_different_physical_tops():
    with pytest.raises(ValueError,match='identical physical model top'):
        mapper([[0],[1],[3]],[[0],[2],[4]],endpoint_alignment=False)


def test_absolute_height_boundary_extension_retains_linear_hydrostatic_profile_shape():
    weights=mapper([[0],[1],[2],[3]],[[ -1],[0],[1],[3]],
                   source_dzw=[1,1,1],target_dzw=[1,1,2],endpoint_alignment=False)
    # Physical means are a decreasing linear profile, including the extra
    # below-ground target layer. The global terrain reconciliation is
    # explicitly removed when comparing the reconstruction's shape.
    result=weights.mass_intensive(np.array([[3.5],[2.5],[1.5]]))
    factor=weights.last_domain_reconciliation['factor']
    np.testing.assert_allclose(result[:,0]/factor,[4.5,3.5,2.])


@pytest.mark.parametrize('values',[
    np.array([[4,7],[3,5],[2,3],[1,1]],np.float32),
    np.array([[0,0],[1,2],[0,0],[0,1]],np.float32),
    np.array([[-3,2],[1,-2],[2,1],[-1,-1]],np.float32),
])
def test_local_domain_reconciliation_preserves_column_products_and_unchanged_columns(values):
    source=np.array([[0,0],[1,1],[2,2],[3,3],[4,4]],float)
    target=source.copy();target[0,0]=.25
    weights=mapper(source,target,endpoint_alignment=False,domain_reconciliation='local',
        domain_support_depth_m=.5)
    result,budget=weights.conserved(values)
    np.testing.assert_array_equal(result[:,1],values[:,1])
    np.testing.assert_array_equal(result[1:,0],values[1:,0])
    np.testing.assert_allclose(result.sum(axis=0),values.sum(axis=0),atol=3e-7)
    if np.all(values>=0):assert np.all(result>=0)
    receipt=budget['terrain_domain_reconciliation']
    assert receipt['unchanged_columns_carried_exactly']==1
    assert receipt['support_layers_max']<=1


def test_local_lowered_ground_extends_positive_support_without_touching_upper_layers():
    weights=mapper([[0],[1],[2],[3],[4]],[[-2],[-1],[1],[3],[4]],
        source_dzw=[1,1,1,1],target_dzw=[1,2,2,1],endpoint_alignment=False,
        domain_reconciliation='local',domain_support_depth_m=.5)
    values=np.array([[3.5],[2.5],[1.5],[.5]],np.float32)
    result,budget=weights.conserved(values)
    assert np.all(result>0)
    np.testing.assert_allclose(weights.totals(result,source=False),[8.],atol=4e-7)
    assert result[-1,0]==values[-1,0]
    assert budget['terrain_domain_reconciliation']['support_layers_max']<=2


def test_local_domain_reconciliation_keeps_overlap_column_totals_when_cell_count_changes():
    weights=mapper([[0,0],[1,1],[2,2],[3,3]],[[.25],[1],[2],[3]],overlap=[[2.,3.]],
        endpoint_alignment=False,domain_reconciliation='local',domain_support_depth_m=.5)
    values=np.array([[4,2],[3,1],[2,1]],np.float32)
    result,budget=weights.conserved(values)
    np.testing.assert_allclose(result.sum(),(2*9+3*4)/5,atol=3e-7)
    assert budget['terrain_domain_reconciliation']['unchanged_columns_carried_exactly']==0


def test_local_domain_reconciliation_does_not_hide_corrupted_donor_closure():
    weights=mapper([[0],[1],[2]],[[0],[1],[2]],endpoint_alignment=False,
        domain_reconciliation='local')
    weights.areas*=1.25
    with pytest.raises(ValueError,match='FP64 integral'):
        weights.conserved(np.ones((2,1),np.float32))


def test_local_domain_reconciliation_requires_absolute_height_transfer():
    with pytest.raises(ValueError,match='requires absolute-height'):
        mapper([[0],[1],[2]],[[0],[1],[2]],domain_reconciliation='local')


def _valley_and_plateau():
    """A valley target column fed 90 % by a valley donor and 10 % by a donor on a plateau 2 km higher.

    Thin low layers, as in production. THE BREAKAGE (2025-11-08 12Z smoke case, 240/60 km, move at 1 h):
    the absolute-height remap extrapolated the plateau donor 2 km down into the valley column, about 40
    of its lowest layer thicknesses, and the surface-pressure tendency after the move was 6.06x the fixed run.
    """
    top = 12000.
    def column(ground):
        low = ground + np.array([0., 50., 100., 160., 230., 310., 400.])
        return np.concatenate((low, np.linspace(low[-1] + 300., top, 30)))
    source = np.stack([column(0.), column(2000.)], axis=1)
    target = column(0.)[:, None]
    dzw = np.diff(source[:, 0])
    return mapper(source, target, overlap=[[9., 1.]], source_dzw=dzw, target_dzw=dzw, endpoint_alignment=False)


def test_absolute_height_interface_values_stay_within_each_donors_range():
    weights = _valley_and_plateau()
    acceleration = np.zeros_like(weights.source_zgrid)
    acceleration[1:-1, 1] = 0.5 * np.exp(-np.arange(1, len(acceleration) - 1) / 3.)  # plateau first-interface residual
    out = weights.interface_intensive(acceleration)
    assert out.min() >= 0. and out.max() <= 0.5 * 0.1 + 1e-12, (out.min(), out.max())


def test_absolute_height_sub_ground_air_keeps_the_donors_lowest_theta():
    weights = _valley_and_plateau()
    z = .5 * (weights.source_zgrid[:-1] + weights.source_zgrid[1:])
    ground = weights.source_zgrid[0]
    # A 15 K surface inversion over each donor's own ground on a 4 K/km background.
    theta = 290. + 4e-3 * z + 15. * (1. - np.exp(-(z - ground) / 150.))
    exner = 1. - 9.80616 * z / (1004.5 * 300.)
    rho = 1e5 * exner ** (1004.5 / 287.) / (287. * theta * exner)
    weights.extend_below_ground(theta, exner)
    mass = weights.mass_intensive(rho, density=rho)
    heat = weights.mass_intensive(rho * theta, density=rho)
    mapped = heat[:, 0] / mass[:, 0]
    below = .5 * (weights.target_zgrid[:-1, 0] + weights.target_zgrid[1:, 0]) < 2000.
    # Within the valley column below the plateau: a 90/10 mix of the valley donor and the plateau donor's
    # lowest theta, never colder than both (the extrapolated inversion was 30 K and more below them).
    assert np.all(mapped[below] >= theta[:, 0].min() - 0.5), mapped[below].min()
    assert np.all(mass > 0)


@pytest.mark.parametrize('kind', ['positive', 'signed', 'sparse'])
@pytest.mark.parametrize('layers', [1, 2, 9])
def test_column_workers_preserve_mass_words_and_receipts(monkeypatch, kind, layers):
    from woof.hex import host_parallel
    rng = np.random.default_rng(109)
    n = 19
    z = np.linspace(0., 10000., layers+1)[:, None]*np.ones((1, n))
    target = z+(1-z/10000.)*rng.uniform(-10., 10., n)
    weights = mapper(z, target, source_dzw=np.diff(z[:, 0]),
                     target_dzw=np.diff(z[:, 0]), endpoint_alignment=False, backend='cpu')
    density = rng.uniform(.1, 1.1, (layers, n))
    value = rng.uniform(0., 2., (layers, n))
    if kind == 'signed': value -= 1.
    if kind == 'sparse': value[value < 1.8] = 0.
    weights.extend_below_ground(np.full_like(density, 300.), np.ones_like(density))
    monkeypatch.setenv('WOOF_HEX_HOST_COPY_THREADS', '1')
    before = weights.mass_intensive(value, density=density)
    receipts = weights.last_domain_reconciliation, weights.last_positivity_roundoff
    monkeypatch.setattr(host_parallel, 'MIN_PARALLEL_ELEMENTS', 1)
    monkeypatch.setattr(host_parallel, 'MIN_BLOCK_COLUMNS', 3)
    monkeypatch.setenv('WOOF_HEX_HOST_COPY_THREADS', '4')
    after = weights.mass_intensive(value, density=density)
    assert before.tobytes() == after.tobytes()
    assert receipts == (weights.last_domain_reconciliation, weights.last_positivity_roundoff)


def test_column_roundoff_workers_keep_errors_and_adjustment_receipt(monkeypatch):
    from woof.hex import host_parallel
    from woof.hex.moving_vertical import positive_cumulative_roundoff
    values = np.tile(np.array([0., 1., 1.+2*np.finfo(float).eps, 1.])[:, None], (1, 19))
    values[:, ::3] = 0.
    monkeypatch.setenv('WOOF_HEX_HOST_COPY_THREADS', '1')
    before, receipt = positive_cumulative_roundoff(values)
    monkeypatch.setattr(host_parallel, 'MIN_PARALLEL_ELEMENTS', 1)
    monkeypatch.setattr(host_parallel, 'MIN_BLOCK_COLUMNS', 3)
    monkeypatch.setenv('WOOF_HEX_HOST_COPY_THREADS', '4')
    after, actual = positive_cumulative_roundoff(values)
    assert before.tobytes() == after.tobytes() and receipt == actual
    values[2, 17] = 1.00001
    with pytest.raises(ValueError, match='FP64 arithmetic allowance'):
        positive_cumulative_roundoff(values)
