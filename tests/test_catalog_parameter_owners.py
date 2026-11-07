"""Shared controls stay inert when none of their declared readers runs."""
from woof import physics_catalog as pc

def test_shared_control_aliases_name_existing_equal_settings():
    menu = {family: {row["registry_option_id"]: row for row in rows.values()} for family, rows in pc._menu().items()}
    for family, rows in pc.table()["parameter_owners"].items():
        for option, keys in rows.items():
            for key, owner in keys.items():
                assert menu[family][option]["settings"][key] == menu[family][owner]["settings"][key]

def test_shared_control_is_live_for_each_reader_and_inert_elsewhere():
    menu = {family: {row["registry_option_id"]: row for row in rows.values()} for family, rows in pc._menu().items()}
    for family, rows in pc.table()["parameter_owners"].items():
        for option, keys in rows.items():
            for key, owner in keys.items():
                switches = {key: menu[family][option]["settings"][key]}
                assert key not in pc._inert(switches, {family: option})
                assert key not in pc._inert(switches, {family: owner})
                inactive = next(k for k, row in menu[family].items() if key not in row["settings"])
                assert key in pc._inert(switches, {family: inactive})
