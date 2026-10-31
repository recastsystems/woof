all: libtransport.so
libtransport.so: $(SOURCE_DIR)/oracle_bridge.rs
	rustc --edition=2021 --crate-type=cdylib -O $< -o $@

native_defect_fixture: $(SOURCE_DIR)/native_defect_fixture.rs
	rustc --edition=2021 -O $< -o $@
