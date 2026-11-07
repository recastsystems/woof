"""UA admission must expose the same Noah kernel family and output contract."""
import inspect
from woof.config import RunConfig
from woof.core.noah import launch_noah
from woof.core.noah_mosaic import launch_noah_mosaic

def test_ua_false_is_explicit_default_on_existing_launchers():
    for launcher in (launch_noah, launch_noah_mosaic):
        parameter = inspect.signature(launcher).parameters.get("ua_phys")
        assert parameter is not None, f"{launcher.__name__} cannot reach WRF UA_PHYS"
        assert parameter.default is False

def test_ua_config_field_is_false_by_default():
    assert RunConfig.__dataclass_fields__["ua_phys"].default is False

def test_ua_flag_reaches_toml(tmp_path):
    from woof.config import load_config
    p=tmp_path/"ua.toml"
    p.write_text('[grid]\nnx=12\nny=12\nnz=8\ndx=3000.0\ndy=3000.0\nztop=12000.0\n[run]\ndt=6.0\nrun_seconds=60.0\nsf_surface_physics=2\nsf_sfclay_physics=91\nbl_pbl_physics=1\nua_phys=true\n')
    assert load_config(p).ua_phys is True

def test_ua_namelist_import_is_not_a_substitution(tmp_path):
    from test_namelist_import import _pair,INPUT_TEXT
    from woof.namelist_import import import_namelists
    inp=INPUT_TEXT.replace("&physics","&physics\n ua_phys = .true.,",1)
    wps,nl=_pair(tmp_path,inp=inp)
    text,report=import_namelists(wps,nl)
    assert "ua_phys = true" in text
    assert all(row.key!="ua_phys" for row in report.substitutions)

def test_false_restart_echo_drops_only_default_flag():
    from woof.io.restart import _drop_default_off_run_keys
    a={"ua_phys":False,"sf_surface_physics":2};_drop_default_off_run_keys(a)
    assert "ua_phys" not in a
    b={"ua_phys":True,"sf_surface_physics":2};_drop_default_off_run_keys(b)
    assert b["ua_phys"] is True

def test_ua_column_fixture_covers_short_and_tall_burial_and_flux():
    from pathlib import Path
    import numpy as np
    from woof.core.noah import load_tables,pack_params
    from woof.verify.noah_oracle import load_noah_oracle
    p=pack_params(load_tables())
    d=Path(__file__).resolve().parents[1]/"woof/data/noah/ua_oracle"
    f=load_noah_oracle("noah-lsm-ua.csv",d)
    bottom=p.canopy[f.ivgtyp.ravel()-1,1]
    depth=f.inputs["snowh"].ravel()
    assert np.any((bottom>0.5)&(depth>bottom))
    assert np.any((bottom>0.)&(bottom<=0.5)&(depth>bottom))
    assert np.count_nonzero(f.reference["flx4"])>=20
    assert np.any(f.reference["fbur"]==1.)
    assert np.any((f.reference["fgsn"]>0.)&(f.reference["fgsn"]<1.))
    assert np.any(f.reference["fgsn"]==0.)
    assert np.any(f.reference["fgsn"]==1.)

def test_ua_float64_mirror_tracks_independent_driver_control():
    from pathlib import Path
    import numpy as np
    from woof.core.noah import load_tables,pack_params
    from woof.verify.npref import np_noah_column
    from woof.verify.noah_oracle import load_noah_oracle,FIXTURE_DT,FIXTURE_DZS
    params=pack_params(load_tables())
    directory=Path(__file__).resolve().parents[1]/"woof/data/noah/ua_oracle"
    f=load_noah_oracle("noah-lsm-ua-pc-control.csv",directory)
    checked=0
    for i,case in enumerate(f.cases):
        if case in (25,26,27,28): continue
        col={k:(v[:,0,i].copy() if v.ndim==3 else float(v[0,i]))
             for k,v in f.inputs.items()}
        col.update(ivgtyp=int(f.ivgtyp[0,i]),isltyp=int(f.isltyp[0,i]))
        result=np_noah_column(col,params,FIXTURE_DT,FIXTURE_DZS,**f.switches)
        if result["near_branch"]: continue
        checked+=1
        for field in ("flx4","fvb","fbur","fgsn"):
            want=float(f.reference[field][0,i])
            assert np.isclose(result[field],want,rtol=2.e-5,atol=(1.e-3 if field=="flx4" else 1.e-6)), (case,field,result[field],want)

    assert checked>=20

def test_ua_family_uses_shared_unflushed_compile_options():
    from woof.core.kernels import module_options
    for name in ("noah_ua","noah_mosaic_ua","noah_ua_sfcdiags"):
        assert module_options(name)==("-std=c++17","--fmad=false","--ftz=false")
    assert module_options("noah")==("-std=c++17","--fmad=false")
    assert module_options("noah_mosaic")==("-std=c++17",)

def test_public_real_case_namelist_keeps_ua_without_substitution():
    import tomllib
    from pathlib import Path
    from woof.namelist_import import import_namelists
    d=Path(__file__).resolve().parents[1]/"tools/noah_wrf461_oracle/public_namelist"
    text,report=import_namelists(d/"namelist.wps",d/"namelist.input",
                               rrtmg_variant="rte-rrtmgp",wrf_version=4)
    cfg=tomllib.loads(text)
    assert cfg["shared"]["ua_phys"] is True
    assert cfg["shared"]["sf_surface_physics"]==2
    assert all(row.key!="ua_phys" for row in report.substitutions)
