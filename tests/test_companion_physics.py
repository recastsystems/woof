import copy
import hashlib
import tomllib
from pathlib import Path

import pytest

from woof import companion_domains as editor
from woof.companion_physics import (ALL_DOMAINS, AT_RUN_PREPARATION, AT_SAVE,
                                     CHECK_FAILED, COMBINATION, NOT_IMPLEMENTED,
                                     PRECONDITION, REASON_KINDS, SHARED_SCOPE,
                                     SOURCE_ROUTE, availability, repairs)
from test_companion_domains import configured_case

#: The kinds whose door fires after the save that accepts the settings.
_AFTER_SAVE_KINDS = (SOURCE_ROUTE, PRECONDITION)

#: A draft the configuration parser refuses at save: Milbrandt-Yau with
#: Dudhia shortwave and NO longwave scheme, on a setup that runs no
#: land-surface model, so nothing integrates the downwelling longwave the
#: active radiation slot would publish.  Every installed microphysics
#: scheme couples to both radiation variants (this fixture used to be
#: mp=9 against RTE+RRTMGP, a refusal that no longer exists), so the
#: refusal exercised here is a radiation pairing whose one-field repairs
#: are a longwave scheme beside Dudhia, or radiation off.
DRAFT = {"mp_physics": 9, "moist": True, "moist_cq": True,
         "ra_lw_physics": 0, "ra_sw_physics": 1}

#: The two one-field repairs of ``DRAFT``, as (field, value).
ONE_FIELD_REPAIRS = {("ra_lw_physics", 1), ("ra_sw_physics", 0)}


def draft(source, grid_id=0):
    return {"schema": editor.REQUEST_SCHEMA, "config_path": str(source),
            "expected_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "action": {"kind": "set_physics", "grid_id": grid_id,
                       "settings": dict(DRAFT)}}


def test_repairs_keep_unverified_runnable_options_and_publish_nothing(tmp_path):
    source, raw = configured_case(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    request = draft(source)
    result = repairs(request)
    assert result["valid"] is False
    assert "no longwave scheme" in result["error"]
    # No registry rule owns this refusal, so the generic sentence stands;
    # the parser's own words travel in ``error`` and the repairs below
    # say what changes.
    assert result["summary"] == "These physics settings cannot run together. Choose a compatible replacement below."
    assert result["created"] is result["forecast_started"] is False
    # The closest repairs change exactly one field each and sort first.
    closest = [option for option in result["options"] if len(option["changes"]) == 1]
    assert result["options"][:len(closest)] == closest
    assert {(option["changes"][0]["field"], option["changes"][0]["after"]) for option in closest} == ONE_FIELD_REPAIRS
    for option in closest:
        change = option["changes"][0]
        expected = dict(request["action"]["settings"], **{change["field"]: change["after"]})
        assert expected.items() <= option["action"]["settings"].items()
    assert any(row["maturity"] == "implemented-unverified" for option in result["options"] for row in option["evidence"])
    # Every advertised result really passes the parser with all domain overrides.
    for option in result["options"]:
        candidate = copy.deepcopy(raw)
        editor._apply(candidate, option["action"], source)
        editor._build(candidate, source)
        assert option["validation"]["forecast_run"] == "not_run"
    assert any("Longwave radiation will be disabled." in option["effects"] for option in result["options"])
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}


def test_the_summary_and_the_first_repair_come_from_the_registry_table(
        tmp_path, monkeypatch):
    """No per-scheme branch, and no substring of the parser's prose.

    Both used to be code paths in generic code: one matched "mp_physics=9"
    and "cloud-optics" in the error to choose a hand-written summary, the
    other to offer the minimal repair.  A reworded refusal lost the repair
    and the next scheme refused the same way never had one.  The rule
    carries its reason and its remedy now, so both are read off the table.

    NO RULE FIRES ON THIS DRAFT TODAY.  The rule this began with,
    Milbrandt-Yau against RTE+RRTMGP, retired with the defect it
    described: the adapter carries the scheme's own cloud-optics row and
    every implemented scheme radiates under both variants.  The table is
    not empty -- microphysics off is refused on an HRRR start, whose
    analyzed condensate it cannot keep, and the turbulence closures carry
    the diffusion-selector rules (the 1.5-order closure beside a PBL under
    diff_opt = 2, diff_opt = 1 without coordinate coefficients) -- but the
    first is scoped
    to its source and none of the others holds for this draft, which runs
    Milbrandt-Yau at the default diffusion.  So the door is measured
    both ways: against the tracked registry, where no rule fires and the
    generic sentence stands, and against a rule handed to it, where the
    summary and the first repair must be the rule's own words and the
    rule's own edit.
    """
    from woof import physics_compat
    from woof.companion_physics import GENERIC_SUMMARY
    from woof.physics_compat import conditional_refusals_for
    from woof.physics_registry import _conditional_refusals, physics_registry

    registry = physics_registry()
    rules = [rule for component in registry["components"].values()
             for option in component["options"].values()
             for rule in _conditional_refusals(option.get("constraints", {}))]
    assert any(rule.get("sources") for rule in rules), rules
    assert conditional_refusals_for(dict(DRAFT)) == []
    source, _ = configured_case(tmp_path)
    result = repairs(draft(source))
    assert result["valid"] is False
    assert result["summary"] == GENERIC_SUMMARY
    # Every domain was measured against the table, and the table was empty.
    assert result["unmeasured"] == []

    # A rule that owns the refusal: its sentence is the summary and its
    # remedy is the repair offered first, the smallest edit that clears the
    # refusal with every other draft setting kept.
    rule = {"component": "radiation", "option": "dudhia-shortwave",
            "reason": ("Dudhia shortwave with no longwave scheme is refused "
                       "here. Set ra_lw_physics=1."),
            "remedy_label": "Set ra_lw_physics=1.",
            "remedy_settings": {"ra_lw_physics": 1}}
    monkeypatch.setattr(physics_compat, "conditional_refusals_for",
                        lambda settings: [dict(rule)])
    owned = repairs(draft(source))
    assert owned["valid"] is False
    assert owned["summary"] == rule["reason"]
    assert owned["summary"] != GENERIC_SUMMARY
    closest = owned["options"][0]
    assert closest["label"] == rule["remedy_label"]
    assert [change["field"] for change in closest["changes"]] == ["ra_lw_physics"]
    assert all(closest["action"]["settings"][key] == value
               for key, value in rule["remedy_settings"].items())
    assert owned["unmeasured"] == []

    # A refusal no rule owns keeps the generic sentence rather than
    # borrowing the words of a rule that did not fire.
    monkeypatch.setattr(physics_compat, "conditional_refusals_for",
                        lambda settings: [])
    request = draft(source)
    request["action"]["settings"] = {"mp_physics": 6, "bl_pbl_physics": 2,
                                     "sf_sfclay_physics": 1}
    other = repairs(request)
    assert other["valid"] is False
    assert other["summary"] == GENERIC_SUMMARY


def test_single_domain_repair_does_not_change_siblings(tmp_path):
    source, raw = configured_case(tmp_path)
    raw["shared"].update(moist=True, moist_cq=True, mp_physics=9)
    from woof.toml_document import emit_experiment_toml
    source.write_text(emit_experiment_toml(raw), encoding="utf-8")
    request = draft(source, 2)
    # These defaults are shared; keep the request within per-domain scope.
    request["action"]["settings"].pop("moist")
    request["action"]["settings"].pop("moist_cq")
    result = repairs(request)
    assert result["valid"] is False
    closest = result["options"][0]
    assert len(closest["changes"]) == 1
    change = closest["changes"][0]
    assert (change["field"], change["after"]) in ONE_FIELD_REPAIRS
    candidate = copy.deepcopy(raw)
    editor._apply(candidate, closest["action"], source)
    assert candidate["domain"][0] == raw["domain"][0]
    assert candidate["domain"][2] == raw["domain"][2]
    assert candidate["shared"] == raw["shared"]
    repaired = dict(request["action"]["settings"], **{change["field"]: change["after"]})
    assert candidate["domain"][1]["mp_physics"] == 9
    assert repaired.items() <= candidate["domain"][1].items()
    # Only the repair's own action reaches the row (a radiation option
    # carries its aggregate selector beside the two spectra).
    assert set(candidate["domain"][1]) - set(raw["domain"][1]) <= set(closest["action"]["settings"])


def test_valid_draft_and_stale_source(tmp_path):
    source, _ = configured_case(tmp_path)
    request = draft(source)
    request["action"]["settings"]["ra_lw_physics"] = 1
    result = repairs(request)
    assert result["valid"] is True
    assert result["options"] == []
    source.write_bytes(source.read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match="configuration changed"):
        repairs(request)


def shipped(tmp_path, name):
    """A configuration this release actually ships, copied where it can be read.

    The kinds are claims about the installed engine, so they are measured
    against configurations that ship with it and not only against a
    fixture written to produce them.
    """

    configs = Path(__file__).resolve().parents[1] / "configs"
    source = tmp_path / name
    source.write_bytes((configs / name).read_bytes())
    namelist = configs / (source.stem + ".namelist.wps")
    if namelist.exists():
        (tmp_path / "namelist.wps").write_bytes(namelist.read_bytes())
    return source


def stage_wif(monkeypatch, tmp_path):
    """Put WRF's monthly aerosol climatology where every search rung sees it.

    An mp_physics=28 domain with EXTERNAL lateral boundaries
    (``specified``) needs that dataset or a deliberate synthetic source, so whether this door opens the option is
    a question about a 225 MB file.  These two helpers make it a decision
    of the test rather than of the machine the test runs on.  The
    precondition decodes the FIRST record group -- an empty file used to
    satisfy it and then die in the ingest lane, which is a refusal after
    step 0 -- so the fixture is the smallest structurally valid WPS
    intermediate file (tests/wif_intermediate_stub.py), not an empty one.
    The remaining bytes are the ingest lane's business, and a run that
    reads them is a different test.
    """

    from woof.ingest import wif_climatology
    from wif_intermediate_stub import write_minimal_wif_intermediate

    dataset = tmp_path / wif_climatology.WIF_CLIMATOLOGY_FILE
    write_minimal_wif_intermediate(dataset)
    monkeypatch.setenv(wif_climatology.WIF_CLIMATOLOGY_PATH_ENV, str(dataset))
    return dataset


def unstage_wif(monkeypatch, tmp_path):
    """The state a user without the climatology is in, on any machine."""

    from woof.ingest import wif_climatology, wif_dataset

    monkeypatch.delenv(wif_climatology.WIF_CLIMATOLOGY_PATH_ENV, raising=False)
    monkeypatch.delenv(wif_climatology.WIF_CLIMATOLOGY_ROOT_ENV, raising=False)
    empty = tmp_path / "no-staged-wif"
    empty.mkdir(exist_ok=True)
    monkeypatch.setenv(wif_dataset.WIF_DATASET_ROOT_ENV, str(empty))
    # The last rung is the working directory, so it has to be one the
    # other helper has not written the dataset into.
    monkeypatch.chdir(empty)
    return empty


def ask(source, grid_id=0, settings=None):
    return availability({"schema": editor.REQUEST_SCHEMA, "config_path": str(source),
                         "expected_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                         "action": {"kind": "set_physics", "grid_id": grid_id,
                                    "settings": dict(settings or {})}})


def option_row(result, component, option_id):
    rows = [c for c in result["components"] if c["id"] == component]
    assert len(rows) == 1, component
    matches = [option for option in rows[0]["options"] if option["id"] == option_id]
    assert len(matches) == 1, option_id
    return matches[0]


def every_option(result):
    return [(component["id"], option) for component in result["components"]
            for option in component["options"]]


def kinds_of(option):
    return [reason["kind"] for reason in option["reasons"]]


def test_availability_answers_every_installed_option_in_the_doors_own_words(tmp_path):
    source, _ = configured_case(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    result = ask(source, settings=dict(DRAFT, ra_lw_physics=1))
    assert result["schema"] == "arwen.companion-physics-availability.v1"
    assert result["created"] is result["forecast_started"] is False
    assert result["draft"]["available"] is True and result["draft"]["reasons"] == []
    # Milbrandt-Yau is running; Dudhia shortwave with no longwave scheme is
    # closed on this setup in the engine's own words, and both RTE+RRTMGP
    # variants are open to the scheme (it radiates its own radii under the
    # modern arm; the refusal that once closed the default variant is gone).
    closed = option_row(result, "radiation", "dudhia-shortwave")
    assert closed["available"] is False
    assert "no longwave scheme" in closed["reasons"][0]["reason"]
    assert option_row(result, "radiation", "rte-rrtmgp:rte-rrtmgp")["available"] is True
    assert option_row(result, "radiation", "rte-rrtmgp:rrtmg_legacy")["available"] is True
    for component, option in every_option(result):
        where = f"{component}/{option['id']}"
        # Open means no reasons, and closed means at least one. There is no
        # third state, and no reason without a sentence to read.
        assert option["available"] == (option["reasons"] == []), where
        for reason in option["reasons"]:
            assert reason["kind"] in REASON_KINDS, where
            assert reason["reason"].strip(), where
            assert reason["at"] in (AT_SAVE, AT_RUN_PREPARATION), where
            assert reason["remedy"] in (None, ALL_DOMAINS), where
            assert isinstance(reason["closes"], bool), where
            # The sentence a tooltip shows is the action half only: the
            # explain sentinel never reaches a reader, and the mechanism
            # half travels beside it rather than inside it.
            assert "[[explain]]" not in reason["reason"], where
            assert reason["kind"] != CHECK_FAILED, reason["reason"]
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}


def test_a_refused_draft_reports_its_own_reason_before_any_save(tmp_path):
    source, _ = configured_case(tmp_path)
    draft = ask(source, settings=DRAFT)["draft"]
    assert draft["available"] is False
    assert kinds_of(draft) == [COMBINATION]
    assert draft["reasons"][0]["at"] == AT_SAVE
    assert "no longwave scheme" in draft["reasons"][0]["reason"]
    # The draft is not an option: nothing is greyed for it, so no reason of
    # its own claims to take anything away.
    assert draft["reasons"][0]["closes"] is False


def test_a_scheme_one_domain_cannot_set_is_closed_by_the_scope_that_refuses_it(tmp_path):
    """``shared-scope`` is measured against the PER-DOMAIN APPLICATION.

    Measuring it the other way round -- claiming it only where All
    domains happens to be admitted as well -- classified this scheme,
    whose own sentence reads "These settings apply to all domains ...
    Select All domains to change them", as an ordinary combination
    refusal.  The panel then offered a cell that 1.0.2 had greyed, and
    the save failed on it.
    """

    source, raw = configured_case(tmp_path)
    noah = option_row(ask(source, 2), "land_surface", "noah")
    assert noah["available"] is False
    scope = noah["reasons"][0]
    assert scope["kind"] == SHARED_SCOPE and scope["closes"] is True
    assert "sf_surface_physics" in scope["reason"]
    # The claim is the per-domain application's own, so that application
    # really does refuse this exact edit on this exact domain.
    with pytest.raises(ValueError, match="apply to all domains"):
        editor._apply(copy.deepcopy(raw), {"kind": "set_physics", "grid_id": 2,
                                           "settings": {"sf_surface_physics": 2}}, source)
    # All domains is what that refusal offers, and here it is refused too
    # (the scheme wants a surface layer, at every scope). The remedy is not
    # claimed, and the second door is reported rather than met later.
    assert scope["remedy"] is None
    assert kinds_of(noah) == [SHARED_SCOPE, COMBINATION]
    assert "requires a surface layer" in noah["reasons"][1]["reason"]
    assert noah["reasons"][1]["closes"] is False
    assert option_row(ask(source, 0), "land_surface", "noah")["available"] is False


def test_the_all_domains_remedy_is_claimed_only_where_it_works(tmp_path):
    """Every remedy this door prints was tried, on a shipped setup."""

    source = shipped(tmp_path, "les_nest_250m_grayzone.toml")
    here, everywhere = ask(source, 2), ask(source, 0)
    scoped = [(component, option) for component, option in every_option(here)
              if any(reason["kind"] == SHARED_SCOPE for reason in option["reasons"])]
    assert scoped, "some option on this shipped setup is shared and refused at a nest"
    for component, option in scoped:
        # A shared setting is not this domain's to change, whatever else is
        # true of it, so the cell is taken away either way.
        assert option["reasons"][0]["closes"] is True
    offered = [(component, option) for component, option in every_option(here)
               if any(reason["remedy"] == ALL_DOMAINS for reason in option["reasons"])]
    assert offered, "some option here is admitted run-wide"
    for component, option in offered:
        # Without exception: where the remedy is printed, selecting All
        # domains really does open the option.
        assert option_row(everywhere, component, option["id"])["available"] is True, option["id"]
    # And the converse: no option is left without the remedy that All
    # domains would have opened.
    for component, option in every_option(here):
        if option["available"] or any(reason["remedy"] == ALL_DOMAINS for reason in option["reasons"]):
            continue
        assert option_row(everywhere, component, option["id"])["available"] is False, option["id"]

    # THE SAME PROPERTY BESIDE A DRAFT.  The remedy is measured with the
    # rest of the unsaved draft in place, so it is checked that way too:
    # every remedy printed here agrees with the All-domains answer for the
    # SAME draft, and none is missing where that answer admits the option.
    # The two paths that print it used to measure it twice, separately.
    drafted = {"mp_physics": 6, "moist": True, "moist_cq": True}
    here, everywhere = ask(source, 2, drafted), ask(source, 0, drafted)
    for component, option in every_option(here):
        offered = any(reason["remedy"] == ALL_DOMAINS
                      for reason in option["reasons"])
        if option["available"]:
            continue
        assert offered == option_row(
            everywhere, component, option["id"])["available"], option["id"]


def test_a_refusal_another_choice_would_clear_leaves_its_option_reachable(
        tmp_path, monkeypatch):
    """A combination refusal never takes the option away.

    The repair search exists to trade one scheme for another and can only
    be reached from a draft the reader was allowed to build.  Greying a
    cell because the option is refused BESIDE what is on screen shut that
    door in front of them: on a shipped three-domain setup 16 of the 19
    refusals are of that shape.
    """

    source, _ = configured_case(tmp_path)
    result = ask(source, settings=DRAFT)
    # The Eta surface layer is refused beside the PBL on screen (none),
    # not on its own account.
    surface = option_row(result, "surface_layer", "eta-similarity")
    assert kinds_of(surface) == [COMBINATION]
    assert surface["reasons"][0]["closes"] is False
    # MYJ, EEPS and BouLac are the PBLs the Eta layer is admitted with.
    assert "admitted with bl_pbl_physics=2 (MYJ)" in surface["reasons"][0]["reason"]
    # Measured, not assumed: another installed choice really does open it,
    # which is why the cell that reaches it stays live.
    assert option_row(ask(source, settings={"moist": True, "bl_pbl_physics": 2}),
                      "surface_layer", "eta-similarity")["available"] is True
    # And the spectrum the draft's own refusal is about is not taken away
    # either: the default RTE+RRTMGP pair is open beside Milbrandt-Yau.
    assert option_row(result, "radiation", "rte-rrtmgp:rte-rrtmgp")["available"] is True
    # And a whole shipped setup where every refusal is of that shape keeps
    # every one of its options reachable.  The dataset is staged first, so
    # that shape is a decision of the test: an unstaged machine adds a
    # PRECONDITION reason to aerosol-aware Thompson, which is a true
    # sentence about the machine and not the combination claim this
    # assertion is about.
    stage_wif(monkeypatch, tmp_path)
    conus = ask(shipped(tmp_path, "conus3km.toml"), 0)
    closed = [option for _, option in every_option(conus) if not option["available"]]
    assert closed, "this shipped setup refuses some options"
    for option in closed:
        assert kinds_of(option) == [COMBINATION], option["id"]
        assert not any(reason["closes"] for reason in option["reasons"]), option["id"]


def test_a_door_no_installed_choice_escapes_closes_the_option(tmp_path,
                                                              monkeypatch):
    """The walls, on shipped setups, measured before they are claimed.

    There were two, and both are retired.  The first was the nested
    km_opt=2 refusal: TKE 1.5-order on the shipped LES tree put a km_opt=2
    child under a km_opt=2 parent, which raised NotImplementedError for
    want of a run although the child cold-starts its own TKE either way,
    as WRF's does.  That tree is admitted with a warning now, and so is
    SASE on a nest, so both cells are open at every scope; the closing
    rule itself is measured with a staged NotImplementedError in
    test_a_door_one_other_choice_escapes_is_reported_but_closes_nothing.
    The second was a SOURCE-ROUTE wall: aerosol-aware
    Thompson was closed whenever the configured source was the native
    HRRR route, on the premise that the route carried no aerosol lateral
    boundary condition.  It carries one -- the WIF climatology ingest
    supplies nwfa/nifa and every route's boundary snapshots couple them
    -- so that wall is retired and what remains in its place is a
    DATASET precondition, asked of every source and answered by the
    machine rather than by the route, which is where the two halves below
    measure it.  It reports at RUN PREPARATION, not at save: the
    configuration parser deliberately does not ask whether a 225 MB file
    is installed, because that battery is also the namelist importer's and
    an import writes a TOML and reads no dataset.  The panel asks the run
    door's own inventory (woof.config.run_preparation_preconditions), so
    a greyed cell and the refusal a prepared run meets are one sentence.
    """

    les = shipped(tmp_path, "les_nest_250m_grayzone.toml")
    for grid_id in (0, 1, 2, 3):
        result = ask(les, grid_id)
        for option_id in ("tke-1.5-order", "closure-supplied"):
            row = option_row(result, "turbulence", option_id)
            assert row["available"] is True, (grid_id, option_id,
                                              row["reasons"])
    hrrr = shipped(tmp_path, "hrrr_native_3km_demo.toml")
    unstage_wif(monkeypatch, tmp_path)
    route = option_row(ask(hrrr, 0), "microphysics", "thompson-aerosol-mp28")
    assert SOURCE_ROUTE not in kinds_of(route), route["reasons"]
    assert route["available"] is False
    assert PRECONDITION in kinds_of(route), route["reasons"]
    # And it is NOT a wall: the way out is a per-domain parameter of this
    # very option, which the wall measurement (which varies other
    # components) cannot see, so the cell stays live and the sentence says
    # what a prepared run will meet.
    assert not any(reason["closes"] for reason in route["reasons"])
    said = " ".join(reason["reason"] for reason in route["reasons"])
    assert "QNWFA_QNIFA_SIGMA_MONTHLY.dat" in said, said
    assert all(reason["at"] == AT_RUN_PREPARATION
               for reason in route["reasons"])
    # Stage the dataset and that sentence is GONE from the same option on
    # the same setup, which is what makes it a precondition rather than a
    # second wall: nothing about the route decides it.
    stage_wif(monkeypatch, tmp_path)
    opened = option_row(ask(hrrr, 0), "microphysics", "thompson-aerosol-mp28")
    assert SOURCE_ROUTE not in kinds_of(opened), opened["reasons"]
    staged_said = " ".join(reason["reason"] for reason in opened["reasons"])
    assert "QNWFA_QNIFA_SIGMA_MONTHLY.dat" not in staged_said, staged_said
    # Deliberately NOT "and therefore available": whether every other
    # allow-list on this route admits mp=28 is a different question, and
    # answering it here would make this measurement turn over on a change
    # that never touched an aerosol boundary condition.  The capability
    # itself -- that the route door and the PREPARATION door now give the
    # same answer for mp=28, which is what the retired wall was hiding --
    # is measured in tests/test_hrrr_configured_physics.py
    # (test_the_route_door_and_the_preparation_door_agree_about_mp28).
    # A sibling of the once-walled scheme, on the same setup, is untouched.
    assert option_row(ask(hrrr, 0), "microphysics", "thompson-mp8")["available"] is True


def test_no_reason_claims_a_door_it_did_not_come_from(tmp_path, monkeypatch):
    """Each reason reports the door that produced it, on a shipped setup.

    This used to be one instance: aerosol-aware Thompson, closed on the
    native HRRR route by a gate that fires at namelist emission while a
    save accepted the very same edit.  That gate is retired, and no
    emission route refuses a physics tuple today -- so what survives is
    the invariant the instance was evidence for.  A run-preparation
    ``at`` belongs to the two kinds whose door fires after the save --
    the emission route's gate and the machine's own preconditions -- and
    to nothing else; every other reason came from the configuration
    parser, which IS the door a save passes through.  The route table is
    empty, so the dataset precondition is what exercises the branch here:
    the dataset is unstaged on purpose, which makes this a decision of the
    test rather than of the machine it runs on.
    """

    source = shipped(tmp_path, "hrrr_native_3km_demo.toml")
    unstage_wif(monkeypatch, tmp_path)
    seen = set()
    for _, option in every_option(ask(source, 0)):
        for reason in option["reasons"]:
            seen.add(reason["kind"])
            assert reason["at"] == (AT_RUN_PREPARATION
                                    if reason["kind"] in _AFTER_SAVE_KINDS
                                    else AT_SAVE)
    assert PRECONDITION in seen, sorted(seen)
    # Measured, not assumed: the configuration parser -- the door a save
    # passes through -- really does admit an edit this panel offers.
    candidate = tomllib.loads(source.read_bytes().decode("utf-8-sig"))
    editor._apply(candidate, {"kind": "set_physics", "grid_id": 0,
                              "settings": {"mp_physics": 6}}, source)
    editor._build(candidate, source)


def test_no_emission_route_walls_a_physics_option_today(tmp_path, monkeypatch):
    """The route gate is a TABLE, and its one row was retired.

    The row read: aerosol-aware Thompson is closed when the configured
    source is hrrr.  It was SOURCE-keyed while the breakage it named is
    DATASET-keyed, so it was wrong in both directions -- it refused hrrr
    runs that had the climatology, and it left every other laterally
    forced source (era5, gfs, mapped, a bare run) walking into the same
    aerosol depletion unasked.  Both halves are measured here: the two
    sources that once disagreed about this option now answer alike, and
    neither answers with a route.
    """

    source, _ = configured_case(tmp_path)
    stage_wif(monkeypatch, tmp_path)
    answers = {}
    for forcing in ("gfs", "hrrr"):
        table = "[fetch]" + chr(10) + 'source = "' + forcing + '"' + chr(10)
        source.write_bytes(source.read_bytes().rsplit(b"[fetch]", 1)[0] + table.encode())
        result = ask(source, 0)
        assert result["forcing_source"] == forcing
        aerosol = option_row(result, "microphysics", "thompson-aerosol-mp28")
        answers[forcing] = (aerosol["available"], kinds_of(aerosol))
        assert option_row(result, "microphysics", "thompson-mp8")["available"] is True
    # gfs never had a row, so its answer is the unwalled one; hrrr had the
    # only row there was, so the two agreeing IS the retirement.
    assert answers["gfs"][0] is True, answers
    assert answers["hrrr"] == answers["gfs"], answers
    assert SOURCE_ROUTE not in answers["hrrr"][1], answers


def test_availability_refuses_a_stale_source(tmp_path):
    source, _ = configured_case(tmp_path)
    request = draft(source)
    source.write_bytes(source.read_bytes() + b"# changed" + chr(10).encode())
    with pytest.raises(ValueError, match="configuration changed"):
        availability(request)


def test_a_run_wide_only_refusal_is_not_labelled_a_scope_problem(tmp_path, monkeypatch):
    """A refusal that is not the scope's keeps the parser's own kind.

    The first spelling of this door read the kind off the setting names,
    so a scheme the loader accepts only run-wide -- refused at a nest in
    its own words -- came back as a combination problem, and the front
    end printed "not compatible with the other schemes here" over a
    sentence that said nothing of the sort.  The remedy is still measured
    and still offered; only the claim about WHY changed.
    """

    source, _ = configured_case(tmp_path)
    forbidden = {"mp_physics": 6}
    original_apply = editor._apply

    def refuse_everywhere(candidate, action, where):
        result = original_apply(candidate, action, where)
        if all(action["settings"].get(key) == value for key, value in forbidden.items()):
            raise ValueError("this closure is selected run-wide in [shared], never per nest")
        return result

    monkeypatch.setattr(editor, "_apply", refuse_everywhere)
    row = option_row(ask(source, 2), "microphysics", "wsm6-mp6")
    assert row["available"] is False
    assert kinds_of(row) == [COMBINATION]
    assert row["reasons"][0]["closes"] is False
    assert row["reasons"][0]["remedy"] is None
    assert "run-wide" in row["reasons"][0]["reason"]


def test_a_door_one_other_choice_escapes_is_reported_but_closes_nothing(tmp_path, monkeypatch):
    """The same door, walling or not, decided by measurement alone.

    Both halves of the rule in one test, because the front end reads only
    ``closes``: a scheme nothing admits is taken away, and a scheme one
    other pick admits is reported and kept.  Reading this off the KIND
    instead greyed Milbrandt-Yau for a refusal that names the radiation
    variant sitting beside it.
    """

    source, _ = configured_case(tmp_path)
    original_build = editor._build
    pairing = {}

    def unimplemented(candidate, output):
        experiment = original_build(candidate, output)
        for domain in experiment.domains:
            if domain.run.mp_physics != 6:
                continue
            if not pairing or all(getattr(domain.run, key, None) == value
                                  for key, value in pairing.items()):
                raise NotImplementedError("WSM6 is not implemented here")
        return experiment

    monkeypatch.setattr(editor, "_build", unimplemented)
    walled = option_row(ask(source, 0), "microphysics", "wsm6-mp6")
    assert kinds_of(walled) == [NOT_IMPLEMENTED]
    assert walled["reasons"][0]["closes"] is True
    # Now the same door refuses WSM6 only beside one radiation variant, so
    # a different variant clears it and the cell that reaches it stays live.
    pairing["ra_rrtmg_variant"] = "rte-rrtmgp"
    combination = option_row(ask(source, 0, {"ra_lw_physics": 4, "ra_sw_physics": 4,
                                             "ra_rrtmg_variant": "rte-rrtmgp"}),
                             "microphysics", "wsm6-mp6")
    assert kinds_of(combination) == [COMBINATION]
    assert combination["reasons"][0]["closes"] is False
    assert "not implemented here" in combination["reasons"][0]["reason"]


def test_one_option_that_explodes_costs_only_its_own_row(tmp_path, monkeypatch):
    """A surprise from one option never blanks the other forty.

    The door answers a whole panel in one call, so an exception the
    checker did not anticipate used to fail the call and leave every
    option without a reason.  It is reported, not treated as a refusal:
    a check that did not run is not a reason to take a cell away.
    """

    source, _ = configured_case(tmp_path)
    original_apply = editor._apply

    def explode_on_one(candidate, action, where):
        if action["settings"].get("mp_physics") == 6:
            raise RuntimeError("the registry row for this option is malformed")
        return original_apply(candidate, action, where)

    monkeypatch.setattr(editor, "_apply", explode_on_one)
    result = ask(source, 0)
    broken = option_row(result, "microphysics", "wsm6-mp6")
    assert broken["available"] is False and kinds_of(broken) == [CHECK_FAILED]
    assert broken["reasons"][0]["closes"] is False
    assert "malformed" in broken["reasons"][0]["reason"]
    others = [option for _, option in every_option(result) if CHECK_FAILED not in kinds_of(option)]
    assert any(option["available"] for option in others)
    assert result["draft"]["available"] is True


def test_a_layered_refusal_reaches_a_tooltip_as_its_action_half(tmp_path, monkeypatch):
    """The mechanism paragraph travels beside the sentence, not inside it.

    Refusals in this project are layered (action half, then the ``why``
    half behind the explain sentinel). Carried whole into a tooltip they
    arrived as ~950 characters of prose with the sentinel and WRF source
    citations in the middle of them.
    """

    from woof.explain import layered

    source, _ = configured_case(tmp_path)
    original_apply = editor._apply

    def layered_refusal(candidate, action, where):
        if action["settings"].get("mp_physics") == 6:
            raise ValueError(layered("WSM6 is refused here.  Choose another scheme.",
                                     "phys/module_physics_init.F:3213-3219 rejects the pairing."))
        return original_apply(candidate, action, where)

    monkeypatch.setattr(editor, "_apply", layered_refusal)
    reason = option_row(ask(source, 0), "microphysics", "wsm6-mp6")["reasons"][0]
    assert reason["reason"] == "WSM6 is refused here.  Choose another scheme."
    assert reason["detail"] == "phys/module_physics_init.F:3213-3219 rejects the pairing."


def test_a_domain_the_registry_table_could_not_be_read_on_is_reported(
        tmp_path, monkeypatch):
    """A measurement that failed says so instead of disappearing.

    The rule lookup is wrapped because a read-only panel must not raise a
    traceback out of a door a user opened -- but a swallowed failure left
    the reader with the generic sentence, no tailored repair, and no way
    to tell that answer apart from "no rule owns this refusal".  The
    failure is now a row in the payload, named, beside a summary that is
    accurate about being generic.
    """
    from woof import physics_compat
    from woof.companion_physics import GENERIC_SUMMARY

    def unreadable(settings):
        raise RuntimeError("the registry table could not be evaluated here")

    monkeypatch.setattr(physics_compat, "conditional_refusals_for", unreadable)
    source, _ = configured_case(tmp_path)
    result = repairs(draft(source))
    # The door still answers: the repair search is untouched by the miss.
    assert result["valid"] is False
    assert result["summary"] == GENERIC_SUMMARY
    assert result["options"]
    assert result["unmeasured"]
    assert all(set(row) == {"domain_index", "error"}
               for row in result["unmeasured"])
    assert "could not be evaluated" in result["unmeasured"][0]["error"]
    # One row per distinct failure, not one per domain in scope.
    assert len(result["unmeasured"]) == 1
